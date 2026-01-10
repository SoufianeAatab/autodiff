from ops.param import Param, grads
import numpy as np
class Interpreter:
    def __init__(self, ops, input, output, params):
        self.ops = ops
        self.mem = {}
        self.total_mem = 0
        self.require_grads = [param for param in params if param.require_grads]
        self.input_param = input
        self.output_param = output
        self.add_var(input)
        self.add_var(output)
        self.mem_buffer_size = 0;
        for param in params:
            self.add_var(param)

        # print(self.mem)

    def add_var(self, var):
        if var.id not in self.mem:
            #print("Adding memory for variable", var)
            self.mem[var.id] = self.total_mem // 4
            size = self.compute_linear_size(var)
            self.total_mem += size

    def compute_linear_size(self, var):
        dims = var 
        if isinstance(var, Param):
            dims = list(var.shape)
        elif isinstance(var, int):
            return 4
        size = 4
        for dim in dims:
            size *= dim

        return size
    
    def gen_code(self):
        supported_ops = ['matmul', 'conv2d', 'sum', 'reshape', 'transpose', 'mul', 'add', 'sub', 'exp', 'assign', 'ones_like', 'sigmoid', 'log_softmax', 'sigmoid_diff', 'nll_loss', 'mse', 'const', 'assign', 'max_pool2d', 'max_pool2d_grad', 'binary_cross_entropy', 'binary_cross_entropy_diff']
        print("Memory needed for initializing", self.total_mem // 4)
        self.mem_buffer_size = self.total_mem
        code = ""
        for out in self.ops:
            # print(self.mem_buffer_size)
            if not out._op:
                print(f"op not implemented for {out}")
                continue
            op = out._op
            print(op.op_name)
            if op.op_name in supported_ops:
                if op.op_name in ['transpose', 'reshape']:
                    child = op.a
                    self.mem[out.id] = self.mem[child.id]
                elif op.op_name == "assign":
                    child = op.child[0]
                    if isinstance(op.data, np.ndarray):
                        data = op.data.reshape(-1)
                        # code = "buf[" + str(self.mem[child.id]) +"] = {"
                        code += f"float temp_{child.id}["+ str(len(data)) +"] = {"
                        for x in data:
                            code += str(x) + ", "
                        code += "};\n"
                        code += "memcpy(&buf["+ str(self.mem[child.id]) + f"], temp_{child.id}, sizeof(float) * {len(data)} );\n"
                        #code += str(operator)
                    self.mem[out.id] = self.mem[child.id]
                elif op.op_name == "ones_like":
                    size = 1
                    for d in op.a.shape:
                        size *= d
                    code += "for(uint32_t k=0;k<"+str(size)+";++k){\n"
                    self.mem[out.id] = self.mem_buffer_size // 4
                    code += f"\tbuf[{self.mem[out.id]} + k] = 1.0f;"
                    code +="}\n"
                    self.mem_buffer_size += self.compute_linear_size(out.shape)

                elif op.op_name == "const":
                    size = 1
                    for d in op.x.shape:
                        size *= d
                    code += "for(uint32_t k=0;k<"+str(size)+";++k){\n"
                    code += f"\tbuf[{self.mem[out.id]} + k] = {op.x.data};\n"
                    code += "}\n"
                    self.mem[out.id] = self.mem_buffer_size // 4
                    self.mem_buffer_size += self.compute_linear_size(out.shape)
                else:
                    vars = []
                    for child in op.child:
                        if child is None: # Using Conv2d in backward pass receivs no bias, we need to check if any child is none. 
                            continue
                        if child.id in self.mem:
                            vars.append(self.mem[child.id])
                            # if op.op_name == 'matmul':
                                # print(child.id,'=>',self.mem[child.id])
                        else:
                            assert 1==-1, f'var not found in memory {child.id} not in memory'
                    # self.add_var(out)
                    #print(f"INTERPRETER::{op.op_name}", vars)
                    self.mem[out.id] = self.mem_buffer_size // 4
                    #print(f"INTERPRETER::{op.op_name}", self.mem)
                    vars.append(self.mem[out.id])
                    #print("#####", self.compute_linear_size(out.shape), self.mem_buffer_size)
                    self.mem_buffer_size += self.compute_linear_size(out.shape)
                    #print("######", self.mem_buffer_size, out.shape)
                    code += f"{op.get_inference_code(out, vars)} // {out.shape} {out.id}\n"
            else:
                print('Missing:', op.op_name)

        print("memory needed after running interpreter", self.mem_buffer_size)

        full_code = self.gen_init_params()
        full_code += f"//buf[{self.mem[self.input_param.id]}] = input;\n"
        full_code += f"//buf[{self.mem[self.output_param.id]}] = output;\n"
        full_code += code
        if len(self.require_grads) > 0:
            full_code += self.gen_sgd()
        return full_code

        #print("Memory footprint", mem_buffer_size, "bytes", mem_buffer_size / 1024, 'kb')

    def gen_sgd(self):
        print(self.require_grads)
        code = ""
        for g in self.require_grads:
            grad = grads[g.id]
            grad_id = grad.id
            print(self.mem)
            ptr = self.mem[grad_id]
            size = 1
            for d in grad.shape:
                size *= d
            code += f"// sgd for {grad.id}\n"
            code += "for (uint32_t k=0;k<"+str(size)+";++k){\n"
            code += f"\tbuf[{self.mem[g.id]} + k] -= buf[{ptr} + k] * lr;\n"
            code += "}\n"
        return code

    def gen_init_params(self):
        code = "//===================================================\n"
        code += "float lr = 0.01;\n";
        print("init params", self.mem_buffer_size//4)
        code += f"float* buf = (float*)calloc({self.mem_buffer_size//4}, sizeof(float));\n"
        for param in self.require_grads:
            # conv param
            if len(param.shape) == 4:
                k = 1.0 / (param.shape[-1] * param.shape[-2] * param.shape[-3])
            elif len(param.shape) == 2:
                k = 1.0 / param.shape[1]
            ptr = self.mem[param.id]
            size = 1
            for d in param.shape:
                size *= d

            code += f"init_weights(&buf[{ptr}], {size}, {k}); // {param.shape} {param.id}\n"

        code += "//===================================================\n\n"
        return code

    # def gen_accuracy_test(self, y_pred, y):
    #     size = 1
    #     for d in y_pred.shape:
    #         size *= d
    #     pred_ptr  = self.mem[y_pred.id]
    #     true_ptr = self.mem[y.id]
    #     fmt = """
    #         uint32_t predmax = 0, truemax = 0;
    #         for(uint32_t k=0;k<"""+str(size)+""";++k){
    #             if (buf["""+pred_ptr+""" + predmax] < buf["""+pred_ptr+"""+k]) {
    #                 predmax = k;
    #             }
    #             if (y_ptr[l*10 + truemax] < y_ptr[l*10+k]) {
    #                 truemax = k;
    #             }
    #         }
    #         if (predmax == truemax) correct += 1;
    #         """
