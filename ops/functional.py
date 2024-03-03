from ops.param import Param, Op, grads
from ops.ops import Conv2dTranspose, Matmul, Sigmoid, OnesLike, Mse, SigmoidDiff, Conv2d, Sum, NLLLoss, LogSoftmax, Const, MaxPool2d, MaxPool2dGrad
from enum import Enum
import math

class ConvOrder(Enum):
    OHWC = 1
    OCWH = 2

settings = {}

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
    p = Param(None, (x,), shape=x.shape, )
    z = Sigmoid(x)
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

def sigmoid_grad(x, grad):
    dx = Param(None, shape=x.shape, children=(x, grad))
    op = SigmoidDiff(x, grad)#Op('sigmoid_grad', x, grad)
    dx._op = op
    return dx

def conv2d(x, kernels, bias, stride = (1,1), padding=(0,0)):
    assert 'CONV_ORDER' in settings, "Please set convolution order"

    shape = list(x.shape)
    if len(shape) == 3:
        shape = [1] + shape # append batch
        
    x.shape = shape
    if settings['CONV_ORDER'] == ConvOrder.OCWH:
        B, H, W, C = shape[-4], shape[-1], shape[-2], shape[-3]
        kernel_size = kernels.shape[2], kernels.shape[3]
    else:
        # OWHC
        B, H, W, C = shape[-4], shape[-3], shape[-2], shape[-1]
        kernel_size = kernels.shape[1], kernels.shape[2]

    w_out = (math.floor((H + 2 * padding[0] - kernel_size[0]) / stride[0]) + 1)
    h_out = (math.floor((W + 2 * padding[1] - kernel_size[1]) / stride[1]) + 1)
    if settings['CONV_ORDER'] == ConvOrder.OCWH:
        out = Param(None, children=(x, kernels, bias), shape=(B, kernels.shape[0], h_out, w_out))
    else:
        out = Param(None, children=(x, kernels, bias), shape=(B, h_out, w_out, kernels.shape[0]))

    if bias is None:
        print((B, h_out, w_out, kernels.shape[0]), x.shape, kernels.shape)

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

def conv2d_transpose(x, kernels, bias, kernel_size = (3,3), stride = (1,1), padding=(0,0)):
    assert 'CONV_ORDER' in settings, "Please set convolution order"
    if settings['CONV_ORDER'] == ConvOrder.OCWH:
        W, H = x.shape[-2], x.shape[-1]
    else:
        # OWHC
        W, H = x.shape[-3], x.shape[-2]
    w_out = ((H - 1) * stride[0] - 2 * padding[0] + kernel_size[0])
    h_out = ((W - 1) * stride[1] - 2 * padding[1] + kernel_size[1])
    out = Param(None, children=(x, kernels, bias), shape=(kernels.shape[0], h_out, w_out))
    out._op = Conv2dTranspose(x, kernels, bias, kernel_size, stride, padding)
    return out

def max_pool2d(x, kernel_size = (2,2), stride = (1,1), padding=(0,0)):
    assert 'CONV_ORDER' in settings, "Please set convolution order"

    shape = list(x.shape)
    if len(shape) == 3:
        shape = [1] + shape # append batch
        
    x.shape = shape
    if settings['CONV_ORDER'] == ConvOrder.OCWH:
        B, H, W, C = shape[-4], shape[-1], shape[-2], shape[-3]
    else:
        # OWHC
        B, H, W, C = shape[-4], shape[-3], shape[-2], shape[-1]

    w_out = (math.floor((H + 2 * padding[0] - kernel_size[0]) / stride[0]) + 1)
    h_out = (math.floor((W + 2 * padding[1] - kernel_size[1]) / stride[1]) + 1)
    if settings['CONV_ORDER'] == ConvOrder.OCWH:
        out = Param(None, children=(x, ), shape=(B, x.shape[1], h_out, w_out))
    else:
        out = Param(None, children=(x, ), shape=(B, h_out, w_out, x.shape[-1]))

    out._op = MaxPool2d(x, kernel_size, stride, padding)
    out._backward_op = get_diff_op('max_pool2d')
    def backward():
        grad_out = grads[out.id]
        op = out._op
        grad_in = out._backward_op((op.x, op.kernel_size, op.stride, op.padding), grad_out)
        grads[op.x.id] = grad_in
    out._backward = backward
    return out

def max_pool2d_diff(x, kernel_size, stride, padding, grad):
    assert 'CONV_ORDER' in settings, "Please set convolution order"

    shape = list(x.shape)
    if len(shape) == 3:
        shape = [1] + shape # append batch
        
    x.shape = shape
    if settings['CONV_ORDER'] == ConvOrder.OCWH:
        B, H, W, C = shape[-4], shape[-1], shape[-2], shape[-3]
    else:
        # OWHC
        B, H, W, C = shape[-4], shape[-3], shape[-2], shape[-1]

    w_out = W
    h_out = H
    if settings['CONV_ORDER'] == ConvOrder.OCWH:
        out = Param(None, children=(x, grad), shape=(B, x.shape[1], h_out, w_out))
    else:
        out = Param(None, children=(x, grad), shape=(B, h_out, w_out, x.shape[-1]))

    out._op = MaxPool2dGrad(x, kernel_size, stride, padding, grad)
    return out

def sum(x, dim=0):
    z = Param(None, children=(x,), shape=(x.shape[dim],))
    op = Sum(x, dim)
    z._op = op
    # def backward():
    #     grads[op.a.id] = grads[z.id]
    # z._backward = backward
    return z

def nll_loss(input, target):
    z = Param(None, children=(input, target), shape=(input.shape))
    op = NLLLoss(input, target)
    z._op = op
    z._backward_op = get_diff_op('nll_loss')
    def backward():
        grad_out = grads[z.id]
        dldy, dldy_hat = z._backward_op((op.input, op.target), grad_out)
        grads[op.input.id] = dldy_hat
        grads[op.target.id] = dldy
    z._backward = backward
    return z

def log_softmax(x):
    out = Param(None, children=(x,), shape=(x.shape))
    op = LogSoftmax(x)
    out._op = op
    out._backward_op = get_diff_op('log_softmax')
    def backward():
        grad_out = grads[out.id]
        grads[op.x.id] = out._backward_op((op.x, out), grad_out)
    out._backward = backward
    return out


def const(x, shape):
    out = Param(x, children=(), shape=(shape))
    op = Const(out)
    out._op = op
    return out
