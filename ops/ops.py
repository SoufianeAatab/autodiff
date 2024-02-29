from ops.op import Op
class Matmul(Op):
    def __init__(self, a, b):
        super().__init__('matmul')
        self.a = a
        self.b = b
        self.child = (self.a, self.b)

    def get_inference_code(self, out):
        a_rows = self.a.shape[0]
        a_cols = self.a.shape[1]
        b_rows = self.b.shape[0]
        b_cols = self.b.shape[1]

        return f"mat_mul(v_{self.a.id}, v_{self.b.id}, v_{out.id}, {a_rows}, {a_cols}, {a_cols}, 1, {b_rows}, {b_cols}, {b_cols}, 1);"

class Add(Op):
    def __init__(self, a, b):
        super().__init__('add')
        self.a = a
        self.b = b
        self.child = (self.a, self.b)

    def get_inference_code(self, operator):
        dims = list(self.a.shape)
        size = 1
        for dim in dims:
            size *= dim

        return f"add(v_{self.a.id}, v_{self.b.id}, v_{operator.id}, {size});"
    
class Sub(Op):
    def __init__(self, a, b):
        super().__init__('sub')
        self.a = a
        self.b = b
        self.child = (self.a, self.b)

    def get_inference_code(self, operator):
        dims = list(self.a.shape)
        size = 1
        for dim in dims:
            size *= dim
            
        return f"sub(v_{self.a.id}, v_{self.b.id}, v_{operator.id}, {size});"
    
class Mul(Op):
    def __init__(self, a, b):
        super().__init__('mul')
        self.a = a
        self.b = b
        self.child = (self.a, self.b)

    def get_inference_code(self, operator):
        dims = list(self.a.shape)
        size = 1
        for dim in dims:
            size *= dim
            
        return f"mul(v_{self.a.id}, v_{self.b.id}, v_{operator.id}, {size});"
    
class Transpose(Op):
    def __init__(self, a):
        super().__init__("transpose")
        self.a = a
        self.child = (self.a,)

    def get_inference_code(self, operator):
        return f"v_{operator.id} = v_{self.a.id}; // reshaping or transposing, not an operator actually."
    
class Exp(Op):
    def __init__(self, a):
        super().__init__("exp")
        self.a = a
        self.child = (self.a,)

    def get_inference_code(self, operator):
        dims = list(self.a.shape)
        size = 1
        for dim in dims:
            size *= dim
        return f"exp(v_{self.a.id}, v_{operator.id}, {size});"
    
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

    def get_inference_code(self, operator):
        return f"v_{operator.id} = {self.a}; // {operator.shape}"

class OnesLike(Op):
    def __init__(self, a):
        super().__init__("ones_like")
        self.a = a
        self.child = (self.a,)

    def get_inference_code(self, operator):
        return f"ones_like(v_{self.a.id}); // {operator.shape}"
    
class Sigmoid(Op):
    def __init__(self, x):
        super().__init__("sigmoid")
        self.x = x
        self.child = (self.x,)

    def get_inference_code(self, operator):
        dims = list(self.x.shape)
        size = 1
        for dim in dims:
            size *= dim
        return f"sigmoid(v_{self.x.id}, v_{operator.id}, {size});"
    
class Mse(Op):
    def __init__(self, y, y_pred):
        super().__init__("mse")
        self.y = y
        self.y_pred = y_pred
        self.child = (self.y, self.y_pred)

    def get_inference_code(self):
        return f"mse(var_{self.y.id}, var_{self.y_pred.id})"
    
class SigmoidDiff(Op):
    def __init__(self, x, grad):
        super().__init__("sigmoid_diff")
        self.x = x
        self.grad = grad
        self.child = (self.x, self.grad)
    
    def get_inference_code(self, operator):
        return f"sigmoid_diff(v_{self.x.id}, v_{self.grad.id}, v_{operator.id});"
    
class Sum(Op):
    def __init__(self, x, dim=0):
        super().__init__("sum")
        self.x = x
        self.dim = dim
        self.child = (self.x,)

    def get_inference_code(self, operator):
        return f"sum(v_{self.x.id}, v_{operator.id}, {self.dim}, {self.x.shape[0]}, {self.x.shape[1]});"

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

    def get_inference_code(self, out):
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
            input_x = self.x.shape[1]
            input_y = self.x.shape[2]
            input_ch = self.x.shape[0]
        else:
            input_x = self.x.shape[0]
            input_y = self.x.shape[1]
            input_ch = self.x.shape[2]  
        print(input_x, input_y, input_ch)
        kernel_x = self.kernel_size[0]
        kernel_y = self.kernel_size[1]
        rhs_cols = input_ch * kernel_x * kernel_y
        import math
        w_out = (math.floor((input_x + 2 * pad_x - kernel_x) / stride_x) + 1)
        h_out = (math.floor((input_y + 2 * pad_y - kernel_y) / stride_y) + 1)
        ch_out = self.kernels.shape[0]
        return f"arm_convolve_NHWC( ctx_buf,{pad_x}, {pad_y}, {stride_x}, {stride_y},{out_activation_min}, {out_activation_max}, {input_batches}, {input_x}, {input_y}, {input_ch}, {kernel_x}, {kernel_y}, {rhs_cols}, v_{self.x.id}, v_{self.kernels.id}, {w_out}, {h_out}, {ch_out},v_{out.id});"
        
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

    def get_inference_code(self, operator):
        return f"nll_loss(v_{self.input.id}, v_{self.target.id}, v_{operator.id});"
    
class LogSoftmax(Op):
    def __init__(self, x):
        super().__init__("log_softmax")
        self.x = x
        self.child = (self.x, )

    def get_inference_code(self, operator):
        dims = list(self.x.shape)
        size = 1
        for dim in dims:
            size *= dim
        return f"log_softmax(v_{self.x.id}, v_{operator.id}, {size});"
