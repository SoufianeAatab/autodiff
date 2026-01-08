from ops.param import Param, grads
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

        print(self.mem)

    def add_var(self, var):
        self.mem[var.id] = self.total_mem // 4
        size = self.compute_linear_size(var)
        self.total_mem += size

    def gen_torch_code(self):
        for operator in self.ops:
            if not operator._op:
                continue
            op = operator._op
            if op.op_name == 'matmul':
                # op.data = a.data @ b.data
                print(f"var_{operator.id}=var_{op.a.id}@var_{op.b.id}")
            elif op.op_name == 'transpose':
                # op.data = a.data.t()
                print(f"var_{operator.id}=var_{op.a.id}.t()")
            elif op.op_name == 'sigmoid':
                # op.data = torch.nn.functional.sigmoid(a.data)
                print(f"var_{operator.id}=torch.nn.functional.sigmoid(var_{op.x.id})")
            elif op.op_name == 'mse':
                # a_tensor = a.data
                # b_tensor = b.data
                # assert a_tensor.shape == b_tensor.shape, f"mse two different shape tensors {a_tensor.shape},{b_tensor.shape}"
                # op.data = torch.mean(torch.pow(a.data - b.data, 2), dim=-1, keepdim=True)
                print(f"var_{operator.id}=torch.nn.functional.mse_loss(var_{op.y.id}, var_{op.y_pred.id})")
            elif op.op_name == 'mul':
                # a_tensor = a.data
                # b_tensor = b.data
                # if not isinstance(a.data, torch.Tensor):
                #     a_tensor = torch.tensor(a.data).expand(a.shape)
                # if not isinstance(b.data, torch.Tensor):
                #     b_tensor = torch.tensor(b.data).expand(b.shape)
                # assert a_tensor.shape == b_tensor.shape, f"multipying two different shape tensors {a_tensor.shape},{b_tensor.shape}"
                # op.data = a_tensor * b_tensor
                print(f"var_{operator.id}=var_{op.a.id} * var_{op.b.id}")
            elif op.op_name == 'add':
                # a_tensor = a.data
                # b_tensor = b.data
                # if not isinstance(a.data, torch.Tensor):
                #     a_tensor = torch.tensor(a.data).expand(a.shape)
                # if not isinstance(b.data, torch.Tensor):
                #     b_tensor = torch.tensor(b.data).expand(b.shape)
                # assert a_tensor.shape == b_tensor.shape, f"add two different shape tensors {a_tensor.shape},{b_tensor.shape}"
                # op.data = a_tensor + b_tensor
                print(f"var_{operator.id}=var_{op.a.id} + var_{op.b.id}")

            elif op.op_name == 'sub':
                # a_tensor = a.data
                # b_tensor = b.data
                # if not isinstance(a.data, torch.Tensor):
                #     a_tensor = torch.tensor(a.data).expand(a.shape)
                # if not isinstance(b.data, torch.Tensor):
                #     b_tensor = torch.tensor(b.data).expand(b.shape)
                # assert a_tensor.shape == b_tensor.shape, f"add two different shape tensors {a_tensor.shape},{b_tensor.shape}"
                # op.data = a_tensor - b_tensor
                print(f"var_{operator.id}=var_{op.a.id} - var_{op.b.id}")
            elif op.op_name == 'ones_like':
                print(f"var_{operator.id}=torch.ones_like(var_{op.a.id})")
            elif op.op_name == 'sigmoid_diff':
                print(f"var_{operator.id}=torch.nn.functional.sigmoid(var_{op.x.id}) * (1.0-torch.nn.functional.sigmoid(var_{op.x.id})) * var_{op.grad.id}")
            elif op.op_name == 'mse_grad':
                pass
            elif op.op_name == 'assign':
                print(f"var_{operator.id} = torch.tensor({op.a}).expand({operator.shape})")
            elif op.op_name == 'conv2d':
                if op.bias is not None:
                    print(f"var_{operator.id}=torch.nn.functional.conv2d(var_{op.x.id}, var_{op.kernels.id}, bias=var_{op.bias.id})")
                else:
                    print(f"var_{operator.id}=torch.nn.functional.conv2d(var_{op.x.id}, var_{op.kernels.id})")
            elif op.op_name == 'conv2d_transpose':
                print(f"var_{operator.id}=torch.nn.functional.conv2d_transpose(var_{op.x.id}, var_{op.kernels.id})")
            elif op.op_name == 'nll_loss':
                print(f"var_{operator.id}=torch.nn.functional.nll_loss(var_{op.input.id}, var_{op.target.id})")
            elif op.op_name == 'log_softmax':
                print(f"var_{operator.id}=torch.nn.functional.log_softmax(var_{op.x.id}, dim=1)")
            elif op.op_name == 'exp':
                print(f"var_{operator.id}=torch.exp(var_{op.a.id})")
            elif op.op_name == 'sum':
                print(f"var_{operator.id}=torch.sum(var_{op.x.id}, dim={op.dim}) # {op.x.shape}")
            elif op.op_name == 'reshape':
                print(f"var_{operator.id}=var_{op.a.id}.reshape({op.shape})")
            elif op.op_name == 'const':
                print(f"var_{operator.id}=var_{op.x} #({operator.shape})")
            else:
                print(f"Op {op.op_name} not defined!")

    def compute_linear_size(self, var):
        shape = 0
        if isinstance(var, Param):
            shape = var.shape
        elif isinstance(shape, int):
            return 4
        dims = list(shape)
        size = 4
        for dim in dims:
            size *= dim

        return size
    
    def gen_code(self):
        supported_ops = ['matmul', 'conv2d', 'sum', 'reshape', 'transpose', 'mul', 'add', 'sub', 'exp', 'assign', 'ones_like', 'sigmoid', 'log_softmax', 'sigmoid_diff', 'nll_loss', 'mse', 'const', 'assign', 'max_pool2d', 'max_pool2d_grad', 'binary_cross_entropy', 'binary_cross_entropy_diff']
        self.mem_buffer_size = self.total_mem
        code = ""
        for out in self.ops:
            if not out._op:
                print(f"op not implemented for {out}")
                continue
            op = out._op
            if op.op_name in supported_ops:
                if op.op_name in ['transpose', 'reshape']:
                    child = op.a
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
                    self.mem[out.id] = self.mem_buffer_size // 4
                    code += f"\tbuf[{self.mem[out.id]} + k] = {op.x.data};\n"
                    code += "}\n"
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
                    self.mem[out.id] = self.mem_buffer_size // 4
                    vars.append(self.mem[out.id])
                    self.mem_buffer_size += self.compute_linear_size(out.shape)
                    code += f"{op.get_inference_code(out, vars)} // {out.shape} {out.id}\n"
            else:
                print('Missing:', op.op_name)

        full_code = self.gen_init_params()
        full_code += f"buf[{self.mem[self.input_param.id]}] = input;\n"
        full_code += f"buf[{self.mem[self.output_param.id]}] = output;\n"
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

            code += f"set_weights(&buf[{ptr}], {size}, {k}); // {param.shape} {param.id}\n"

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
