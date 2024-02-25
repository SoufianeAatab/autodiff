from ops.param import Param, Op, grads
from ops.ops import Matmul, Sigmoid, OnesLike, Mse, SigmoidDiff, Conv2d, Sum, CrossEntropy, LogSoftmax

ops = {}
def get_diff_op(op_name):
    assert op_name in ops, f"No diff operation found for {op_name}"
    return ops[op_name]

def register_diff_op(op_name, fnc):
    ops[op_name] = fnc

def ones_like(a):
    p = Param(None, children=(a,), shape=a.shape)
    op = OnesLike(a) #Op('ones_like', a, b=None)
    p._op = op
    def backward():
        grads[op.a.id] = grads[op.id]
        op.a = grads[op.id]
        return op.a
    p._backward = backward
    return p

def matmul(a, b):
    a.other = b
    b.other = a
    shape = (a.shape[0], b.shape[1])
    p = Param(None, children=(a, b), shape=shape, )
    z = Matmul(a, b)
    #Op('matmul', a, b)
    p._op = z
    p._backward_op = get_diff_op('matmul')
    def backward():
        # matmul(x, wT)
        # x @ wT, we're interested in grads wrt to weights and input x
        grad_out = grads[p.id] # read the current noe output gradient
        dw, dx = p._backward_op(p._op, grad_out)
        # assert dw.shape[0] == z.b.data.data.shape[1] and dw.shape[1] == z.b.data.data.shape[0]
        # assert dx.shape == z.a.data.data.shape, f"{dx.shape}, {z.a.data.data.shape}"
        grads[z.a.id] = dx
        grads[z.b.id] = dw
        # because z.b is transposed, we are computing xW^t + b = z, in the forward pass
    p._backward = backward
    return p

def sigmoid(x):
    p = Param(None, (x,), shape=x.shape, var_name='sigmoid_', print_init=False)
    z = Sigmoid(x)#Op('sigmoid', x, b=None)
    p._op = z
    p._backward_op = get_diff_op('sigmoid')
    def backward():
        grad_out = grads[p.id] # read grad out
        grads[z.x.id] = p._backward_op((z.x, p), grad_out) # call grad_fn and save the output
    p._backward = backward
    return p

def mse(y, y_pred):
    p = Param(None, shape=y_pred.shape, children=(y, y_pred), var_name='mse_')
    p._op = Mse(y, y_pred) #Op('mse', y, b=y_pred)
    p._backward_op = get_diff_op('mse')
    def backward():
        grad_out = grads[p.id]
        op = p._op
        dldy, dldy_hat = p._backward_op((op.y, op.y_pred), grad_out)
        grads[op.y.id] = dldy
        grads[op.y_pred.id] = dldy_hat
    p._backward = backward
    return p

# def mse_grad(y, y_pred, grad):
#     dldy = Param(None, shape=y_pred.shape, children=(y, y_pred))
#     dldy_hat = Param(None, shape=y_pred.shape, children=(y, y_pred))
#     op = Op('mse_grad', y, y_pred)
#     dldy._op = op
#     dldy_hat._op = op

#     dldy = 2 * dldy * grad
#     dldy_hat = -2 * dldy_hat * grad
#     return dldy, dldy_hat

def sigmoid_grad(x, grad):
    dx = Param(None, shape=x.shape, children=(x, grad))
    op = SigmoidDiff(x, grad)#Op('sigmoid_grad', x, grad)
    dx._op = op
    return dx

def conv2d(x, kernels, bias, kernel_size = 3, stride = 0, padding=0):
    # out_size = (math.floor((input_size[0] + 2 * padding[0] - kernel_size[0]) / stride[0]) + 1)
    out = Param(None, children=(x, kernels, bias))
    out._op = Conv2d(x, kernels, bias, kernel_size, stride, padding)
    out._backward_op = get_diff_op('conv2d')
    def backward():
        grad_out = grads[out.id]
        op = out._op
        dx, dw, dz = out._backward_op((op.x, op.kernels, op.bias, op.kernel_size, op.stride, op.padding), grad_out)
        grads[op.x.id] = dx
        grads[op.kernels.id] = dw
        grads[op.bias.id] = dz
    out._backward = backward
    return out

def sum(x, dim=0):
    z = Param(None, children=(x,), shape=(x.shape[dim]))
    op = Sum(x, dim) # Op('transpose', self, b=None)
    z._op = op
    def backward():
        grads[op.a.id] = grads[z.id]
    z._backward = backward
    return z

def cross_entropy(input, target):
    z = Param(None, children=(input, target), shape=(input.shape))
    op = CrossEntropy(input, target) # Op('transpose', self, b=None)
    z._op = op
    z._backward_op = get_diff_op('cross_entropy')
    def backward():
        grad_out = grads[z.id]
        dldy, dldy_hat = z._backward_op((op.input, op.target), grad_out)
        grads[op.input.id] = dldy_hat
        grads[op.target.id] = dldy
    z._backward = backward
    return z

def log_softmax(x):
    z = Param(None, children=(x,), shape=(x.shape))
    op = LogSoftmax(x) # Op('transpose', self, b=None)
    z._op = op
    def backward():
        grads[op.x.id] = grads[z.id]
    z._backward = backward
    return z
