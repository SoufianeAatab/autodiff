from ops.functional import matmul, sigmoid_grad, conv2d, sum
from ops.ops import Conv2d, Sum

def linear_diff_op(args, grad):
    w, x = args.b, args.a
    # print(f"%{w.grad.id}=matmul(%{grad.grad.id}.t(),%{x.data.id})")
    # print(f"%{x.grad.id}=matmul(%{grad.grad.id}, %{w.data.id}.t())")
    
    # grad_t = Op('transpose', grad, b=None, )
    # dw = Op(grad_t, )
    # w, x, grad = w.data.data, x.data.data, grad.grad.data
    
    # dw = grad.t() @ x # seems correct to me
    # da = grad @ w.t()
    # print('diff', w, x, grad)
    g_t = grad.t()
    # print(g_t._op, g_t._prev)
    dw = matmul(g_t, x)
    # print(dw._op, dw._prev)
    w_t = w.t() 
    # print(w_t._op, w_t._prev)
    da = matmul(grad, w_t)
    # print(da._op, da._prev)
    return dw, da

def conv2d_diff_op(args, grad):
    x, w, b, kernel_size, stride, padding = args
    C = w.shape[0]
    dz = grad.reshape((C, -1))
    dz = sum(dz, dim=0)
    dw = conv2d(x, grad, None, kernel_size, stride, padding )
    dx = x # Conv2d(x, grad, None, kernel_size, stride, padding )
    return dx, dw, dz

def mse_diff_op(args, grad):
    y, y_hat = args
    yyhat = y - y_hat
    dldy = 2 * yyhat * grad
    dldy_hat = -2 * yyhat * grad
    return dldy, dldy_hat

def sigmoid_diff_op(args, grad):
    z, a = args
    dldz = sigmoid_grad(z, grad)
    return dldz

def cross_entropy_diff(args, grad):
    y, y_hat = args
    """ Gradient of log_softmax
    dldz = -grad_nll + (-torch.exp(output) * grad[: y[arg_max]])
    If we optimize and reduce ops, i.e the negative operators
    dldz = grad_nll - torch.exp(output)) * grad[: y[arg_max]]  

    NLL = - sum(y[i] * log(p(x[i])))
        NLL = - sum(y[i] * log(softmax(x[i])))
        NLL = - sum(y[i] * log_softmax(x[i])); when using log softmax as activation layer
        NLL = - sum(y[i] * a[i]); a as the activation values of the last layer
        NLL = -a[y] => when using sparse y_true. y = y_true
    """
    dldy = -y + y_hat.exp() * grad
    dldy_hat = -y + y_hat.exp() * grad
    return dldy, dldy_hat