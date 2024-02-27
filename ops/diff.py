from ops.functional import matmul, sigmoid_grad, conv2d, sum, conv2d_transpose

def linear_diff_op(args, grad):
    w, x = args.b, args.a
    g_t = grad.t()
    w_t = w.t() 
    dw = matmul(g_t, x)
    da = matmul(grad, w_t)
    return dw, da

def conv2d_diff_op(args, grad):
    x, w, b, kernel_size, stride, padding = args
    C = w.shape[0]
    # todo fix: bias
    dz = grad.reshape((C, -1))
    dz = sum(dz, dim=1)
    x_r = x.reshape((x.shape[0],1, x.shape[1], x.shape[2])) 
    grad_r = grad.reshape((C, 1, grad.shape[1], grad.shape[2]))
    dw = conv2d(x_r, grad_r, None, kernel_size, stride, padding )
    # TODO: remember to permute dim 0 with 1
    dx = conv2d_transpose(grad, w, None, kernel_size, stride, padding ) 
    # Conv2d(x, grad, None, kernel_size, stride, padding )
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
    y_hat, y = args
    """ Gradient of log_softmax
    dldz = -grad_nll + (-torch.exp(output) * grad[: y[arg_max]])
    If we optimize and reduce ops, i.e the negative operators
    dldz = grad_nll - torch.exp(output)) * grad[: y[arg_max]]  

    NLL = - sum(y[i] * log(p(x[i])))
        NLL = - sum(y[i] * log(softmax(x[i])))
        NLL = - sum(y[i] * log(softmax(x[i]))); when using log softmax as activation layer
        NLL = - sum(y[i] * a[i]); a as the activation values of the last layer, and y is one hot vector
        NLL = -a[y] => when using sparse y_true. y index of y_true
    """
    dldy = -grad * y_hat
    dldy_hat = -grad * y
    return dldy, dldy_hat

def log_softmax_diff(args, grad):
    x, a = args
    # assuming we're using nll as a loss function
    dldz = a.exp() + grad 
    return dldz