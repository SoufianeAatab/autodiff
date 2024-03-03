from ops.op import Op
class Matmul(Op):
    def __init__(self, a, b):
        super().__init__('matmul')
        self.a = a
        self.b = b
        self.child = (self.a, self.b)

    def get_inference_code(self, out, child_vars):
        a_rows = self.a.shape[0]
        a_cols = self.a.shape[1]
        b_rows = self.b.shape[0]
        b_cols = self.b.shape[1]

        a_var = child_vars[0]
        b_var = child_vars[1]
        out_var = child_vars[-1]
        # return f"mat_mul(v_{self.a.id}, v_{self.b.id}, v_{out.id}, {a_rows}, {a_cols}, {a_cols}, 1, {b_rows}, {b_cols}, {b_cols}, 1);"
        return f"mat_mul(&buf[{a_var}] /* {self.a.shape} */, &buf[{b_var}] /* {self.b.shape} */, &buf[{out_var}] /* {out.shape} */, {a_rows}, {a_cols}, {self.a.stride[0]}, {self.a.stride[1]}, {b_rows}, {b_cols}, {self.b.stride[0]}, {self.b.stride[1]});"

class Add(Op):
    def __init__(self, a, b):
        super().__init__('add')
        self.a = a
        self.b = b
        self.child = (self.a, self.b)

    def get_inference_code(self, operator, child_vars):
        dims = list(self.a.shape)
        size = 1
        for dim in dims:
            size *= dim

        a_var = child_vars[0]
        b_var = child_vars[1]
        out_var = child_vars[-1]
        # return f"add(v_{self.a.id}, v_{self.b.id}, v_{operator.id}, {size});"
        return f"add(&buf[{a_var}], &buf[{b_var}], &buf[{out_var}], {size});"
    
class Sub(Op):
    def __init__(self, a, b):
        super().__init__('sub')
        self.a = a
        self.b = b
        self.child = (self.a, self.b)

    def get_inference_code(self, operator, child_vars):
        dims = list(self.a.shape)
        size = 1
        for dim in dims:
            size *= dim
        a_var = child_vars[0]
        b_var = child_vars[1]
        out_var = child_vars[-1]
        # return f"sub(v_{self.a.id}, v_{self.b.id}, v_{operator.id}, {size});"
        return f"sub(&buf[{a_var}], &buf[{b_var}], &buf[{out_var}], {size});"
    
class Mul(Op):
    def __init__(self, a, b):
        super().__init__('mul')
        self.a = a
        self.b = b
        self.child = (self.a, self.b)

    def get_inference_code(self, operator, child_vars):
        dims = list(self.a.shape)
        size = 1
        for dim in dims:
            size *= dim
        
        a_var = child_vars[0]
        b_var = child_vars[1]
        out_var = child_vars[-1]
        # return f"mul(v_{self.a.id}, v_{self.b.id}, v_{operator.id}, {size});"
        return f"mul(&buf[{a_var}], &buf[{b_var}], &buf[{out_var}], {size});"
    
class Transpose(Op):
    def __init__(self, a):
        super().__init__("transpose")
        self.a = a
        self.child = (self.a,)

    def get_inference_code(self, operator, vars):
        return "//transpose();"
        # return f"v_{operator.id} = v_{self.a.id}; // reshaping or transposing, not an operator actually."
    
class Exp(Op):
    def __init__(self, a):
        super().__init__("exp")
        self.a = a
        self.child = (self.a,)

    def get_inference_code(self, operator, child_var):
        dims = list(self.a.shape)
        size = 1
        for dim in dims:
            size *= dim
        return f"exp(&buf[{child_var[0]}], &buf[{child_var[-1]}], {size});"
    
class Reshape(Op):
    def __init__(self, a, shape):
        super().__init__("reshape")
        self.a = a
        self.shape = shape
        self.child = (self.a,)

    def get_inference_code(self, operator):
        return f"v_{operator.id} = v_{self.a.id}; // reshaping or transposing, not an operator actually."
    
class Assign(Op):
    def __init__(self, a):
        super().__init__("assign")
        self.a = a
        self.child = (self.a,)

    def get_inference_code(self, operator, child_vars):
        return f"buf[{child_vars[-1]}] = {self.a}; // {operator.shape}"

class OnesLike(Op):
    def __init__(self, a):
        super().__init__("ones_like")
        self.a = a
        self.child = (self.a,)

    def get_inference_code(self, operator, child_vars):
        return f"ones_like(buf[{child_vars[0]}], buf[{child_vars[-1]}]); // {operator.shape}"
    
class Sigmoid(Op):
    def __init__(self, x):
        super().__init__("sigmoid")
        self.x = x
        self.child = (self.x,)

    def get_inference_code(self, operator, child_vars):
        dims = list(self.x.shape)
        size = 1
        for dim in dims:
            size *= dim

        x_var = child_vars[0]
        out_var = child_vars[-1]
        return f"sigmoid(&buf[{x_var}] /* {self.x.shape}*/ , &buf[{out_var}] /*{operator.shape}*/, {size});"
    
class Mse(Op):
    def __init__(self, y, y_pred):
        super().__init__("mse")
        self.y = y
        self.y_pred = y_pred
        self.child = (self.y, self.y_pred)

    def get_inference_code(self, out, child_vars):
        dims = list(self.y.shape)
        size = 1
        for dim in dims:
            size *= dim
        y_var = child_vars[0]
        ypred_var = child_vars[1]
        out_var = child_vars[-1]
        return f"buf[{out_var}] = mse(&buf[{y_var}], &buf[{ypred_var}], {size});"
    
class SigmoidDiff(Op):
    def __init__(self, x, grad):
        super().__init__("sigmoid_diff")
        self.x = x
        self.grad = grad
        self.child = (self.x, self.grad)
    
    def get_inference_code(self, operator, child_vars):
        a_var = child_vars[0]
        b_var = child_vars[1]
        out_var = child_vars[-1]
        dims = list(self.x.shape)
        size = 1
        for dim in dims:
            size *= dim
        return f"sigmoid_diff(&buf[{a_var}], &buf[{b_var}], &buf[{out_var}], {size});"
    
class Sum(Op):
    def __init__(self, x, dim=0):
        super().__init__("sum")
        self.x = x
        self.dim = dim
        self.child = (self.x,)

    def get_inference_code(self, operator, child_vars):
        x = child_vars[0]
        out = child_vars[-1]
        return f"sum(&buf[{x}], &buf[{out}], {self.dim}, {self.x.shape[0]}, {self.x.shape[1]});"

class Conv2d(Op):
    def __init__(self, x, kernels, bias, kernel_size, stride, padding):
        super().__init__('conv2d')
        self.x = x
        self.kernels = kernels
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.child = (self.x, self.kernels, self.bias)

    def get_inference_code(self, out, child_vars):
        pad_x = self.padding[0]
        pad_y = self.padding[0]
        stride_x = self.stride[0]
        stride_y = self.stride[1]
        # Todo: 
        out_activation_min = -6
        out_activation_max = 6
        input_batches = 1
        from ops.functional import settings, ConvOrder
        if settings['CONV_ORDER'] == ConvOrder.OCWH:
            input_x = self.x.shape[2]
            input_y = self.x.shape[3]
            input_ch = self.x.shape[1]
        else:
            input_x = self.x.shape[1]
            input_y = self.x.shape[2]
            input_ch = self.x.shape[3]  
        kernel_x = self.kernel_size[0]
        kernel_y = self.kernel_size[1]
        rhs_cols = input_ch * kernel_x * kernel_y
        import math
        w_out = (math.floor((input_x + 2 * pad_x - kernel_x) / stride_x) + 1)
        h_out = (math.floor((input_y + 2 * pad_y - kernel_y) / stride_y) + 1)
        ch_out = self.kernels.shape[0]

        # print(input_x, input_y, kernel_x, kernel_y, w_out , w_out , ch_out , rhs_cols)
        assert w_out > 0 and w_out > 0 and ch_out > 0 and rhs_cols > 0
        x = child_vars[0]
        filters = child_vars[1]
        out = child_vars[-1]
        # (1, 3, 3, 8) [1, 28, 28, 1] (8, 26, 26, 1)
        return f"arm_convolve_NHWC( ctx, {pad_x}, {pad_y}, {stride_x}, {stride_y},{out_activation_min}, {out_activation_max}, {input_batches}, {input_x}, {input_y}, {input_ch}, {kernel_x}, {kernel_y}, {rhs_cols}, &buf[{x}], &buf[{filters}], {w_out}, {h_out}, {ch_out},&buf[{out}]);"
        
class MaxPool2d(Op):
    def __init__(self, x, kernel_size, stride, padding):
        super().__init__('max_pool2d')
        self.x = x
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.child = (x, )

    def get_inference_code(self, operator, child_vars):
        stride_x = self.stride[0]
        stride_y = self.stride[1]
        pad_x = self.padding[0]
        pad_y = self.padding[1]
        act_min = -6
        act_max = 6
        batch_cnt = self.x.shape[0]
        from ops.functional import settings, ConvOrder
        if settings['CONV_ORDER'] == ConvOrder.OCWH:
            input_x = self.x.shape[2]
            input_y = self.x.shape[3]
            channel_in = self.x.shape[1]
        else:
            input_x = self.x.shape[1]
            input_y = self.x.shape[2]
            channel_in = self.x.shape[3]

        kernel_x = self.kernel_size[0]
        kernel_y = self.kernel_size[1]
        import math
        output_x = math.floor((input_x - kernel_x + 2 * pad_x) // stride_x ) + 1
        output_y = math.floor((input_y - kernel_y + 2 * pad_y) // stride_y ) + 1

        src = child_vars[0]
        dst = child_vars[-1]

        return f"arm_max_pool_s16({stride_x}, {stride_y}, {pad_x}, {pad_y}, {act_min},  {act_max}, {batch_cnt}, {input_x}, {input_y}, {channel_in},{output_x}, {output_y}, {kernel_x}, {kernel_y}, &buf[{src}],&buf[{dst}]);"
    
class MaxPool2dGrad(Op):
    def __init__(self, x, kernel_size, stride, padding, grad):
        super().__init__('max_pool2d_grad')
        self.x = x
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.grad = grad
        self.child = (self.x, self.grad)

    def get_inference_code(self, operator, child_vars):
        x = child_vars[0]
        grad_out = child_vars[1]
        output = child_vars[-1]
        print('MAXPOOL_BCK',self.grad.shape, self.x.shape)
        from ops.functional import settings, ConvOrder
        if settings['CONV_ORDER'] == ConvOrder.OCWH:
            input_x = self.grad.shape[2]
            input_y = self.grad.shape[3]
            channel_in = self.grad.shape[1]
            output_x = self.x.shape[2]
            output_y = self.x.shape[3]
        else:
            input_x = self.grad.shape[1]
            input_y = self.grad.shape[2]
            channel_in = self.grad.shape[3]
            output_x = self.x.shape[1]
            output_y = self.x.shape[2]

        stride_x = self.stride[0]
        stride_y = self.stride[1]
        pad_x = self.padding[0]
        pad_y = self.padding[1]
        kernel_x = self.kernel_size[0]
        kernel_y = self.kernel_size[1]
        
        return f"max_pool_backward(&buf[{grad_out}], &buf[{x}], &buf[{output}],  {input_x},  {input_y},  {channel_in}, {output_x}, {output_y}, {kernel_x},  {kernel_y}, {stride_x}, {stride_y}, {pad_x}, {pad_y});"

class Conv2dTranspose(Op):
    def __init__(self, x, kernels, bias, kernel_size, stride, padding):
        super().__init__('conv2d_transpose')
        self.x = x
        self.kernels = kernels
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.child = (self.x, self.kernels, self.bias)

    def get_inference_code(self):
        if self.bias is not None:
            return f"conv2d_transpose(v_{self.x.id}, v_{self.kernels.id}, v_{self.bias.id}, kernel_size={self.kernel_size}, {self.stride}, {self.padding})"
        else:
            return f"conv2d_transpose(var_{self.x.id}, var_{self.kernels.id}, None, kernel_size={self.kernel_size}, {self.stride}, {self.padding})"
        
class NLLLoss(Op):
    def __init__(self, input, target):
        super().__init__("nll_loss")
        self.input = input
        self.target = target
        self.child = (self.input, self.target)

    def get_inference_code(self, operator, child_var):
        dims = list(self.input.shape)
        size = 1
        for dim in dims:
            size *= dim
        return f"buf[{child_var[-1]}] = nll_loss(&buf[{child_var[0]}], &buf[{child_var[1]}], {size});"
    
class LogSoftmax(Op):
    def __init__(self, x):
        super().__init__("log_softmax")
        self.x = x
        self.child = (self.x, )

    def get_inference_code(self, operator, child_vars):
        dims = list(self.x.shape)
        size = 1
        for dim in dims:
            size *= dim
        return f"log_softmax(&buf[{child_vars[0]}], &buf[{child_vars[-1]}], {size});"

class Const(Op):
    def __init__(self, x):
        super().__init__('const')
        self.x = x
        self.child = ()

    def get_inference_code(self, out, child_vars):
        dims = list(self.x.shape)
        size = 1
        for dim in dims:
            size *= dim
        return f"buf[{child_vars[-1]}] = const({self.x});"