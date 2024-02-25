from ops.op import Op
class Matmul(Op):
    def __init__(self, a, b):
        super().__init__('matmul')
        self.a = a
        self.b = b
        self.child = (self.a, self.b)

    def get_inference_code(self):
        return f"matmul(var_{self.a.id}, var_{self.b.id})"

class Add(Op):
    def __init__(self, a, b):
        super().__init__('add')
        self.a = a
        self.b = b
        self.child = (self.a, self.b)

    def get_inference_code(self):
        return f"var_{self.a.id} + var_{self.b.id}"
    
class Sub(Op):
    def __init__(self, a, b):
        super().__init__('sub')
        self.a = a
        self.b = b
        self.child = (self.a, self.b)

    def get_inference_code(self):
        return f"var_{self.a.id} - var_{self.b.id}"
    
class Mul(Op):
    def __init__(self, a, b):
        super().__init__('mul')
        self.a = a
        self.b = b
        self.child = (self.a, self.b)

    def get_inference_code(self):
        return f"var_{self.a.id} * var_{self.b}"
    
class Transpose(Op):
    def __init__(self, a):
        super().__init__("transpose")
        self.a = a
        self.child = (self.a,)

    def get_inference_code(self):
        return f"var_{self.a.id}.t()"
    
class Exp(Op):
    def __init__(self, a):
        super().__init__("exp")
        self.a = a
        self.child = (self.a,)

    def get_inference_code(self):
        return f"var_{self.a.id}.exp()"
    
class Reshape(Op):
    def __init__(self, a, shape):
        super().__init__("reshape")
        self.a = a
        self.shape = shape
        self.child = (self.a,)

    def get_inference_code(self):
        return f"var_{self.a.id}.reshape({self.shape})"
    
class Assign(Op):
    def __init__(self, a):
        super().__init__("assign")
        self.a = a
        self.child = (self.a,)

    def get_inference_code(self):
        return f"{self.a.data}"

class OnesLike(Op):
    def __init__(self, a):
        super().__init__("ones_like")
        self.a = a
        self.child = (self.a,)

    def get_inference_code(self):
        return f"ones_like(var_{self.a.id})"
    
class Sigmoid(Op):
    def __init__(self, x):
        super().__init__("sigmoid")
        self.x = x
        self.child = (self.x,)

    def get_inference_code(self):
        return f"sigmoid(var_{self.x.id})"
    
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
    
    def get_inference_code(self):
        return f"sigmoid(var_{self.x.id}) * (1-sigmoid(var_{self.x.id})) * var_{self.grad.id}"
    
class Sum(Op):
    def __init__(self, x, dim=0):
        super().__init__("sum")
        self.x = x
        self.dim = dim
        self.child = (self.x,)

    def get_inference_code(self):
        return f"sum({self.x.id}, dim={self.dim})"

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

    def get_inference_code(self):
        if self.bias is not None:
            return f"conv2d(var_{self.x.id}, var_{self.kernels.id}, var_{self.bias.id}, kernel_size={self.kernel_size}, {self.stride}, {self.padding})"
        else:
            return f"conv2d(var_{self.x.id}, var_{self.kernels.id}, None, kernel_size={self.kernel_size}, {self.stride}, {self.padding})"
        
class CrossEntropy(Op):
    def __init__(self, input, target):
        super().__init__("cross_entropy")
        self.input = input
        self.target = target
        self.child = (self.input, self.target)

    def get_inference_code(self):
        return f"cross_entropy(var_{self.input.id}, var_{self.target.id})"
    
class LogSoftmax(Op):
    def __init__(self, x):
        super().__init__("log_softmax")
        self.x = x
        self.child = (self.x, )

    def get_inference_code(self):
        return f"log_softmax(var_{self.x})"