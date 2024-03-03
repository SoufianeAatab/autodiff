from ops.functional import matmul, sigmoid_grad, conv2d, sum, conv2d_transpose, settings, ConvOrder, const, max_pool2d_diff

def linear_diff_op(args, grad):
    w, x = args.b, args.a
    g_t = grad.t()
    w_t = w.t() 
    dw = matmul(g_t, x)
    da = matmul(grad, w_t)
    return dw, da

def conv2d_diff_op(args, grad):
    x, w, b, kernel_size, stride, padding = args
    assert 'CONV_ORDER' in settings, "Please set convolution order"
    shape = list(x.shape)
    if len(shape) == 3:
        shape = [1] + shape # append batch
    O_C = w.shape[0]
    if settings['CONV_ORDER'] == ConvOrder.OCWH:
        W, H, I_C = shape[-2], shape[-1], shape[-3]
        GRAD_W, GRAD_H = grad.shape[2], grad.shape[3]
        x_r = x.reshape((I_C, 1, H, W)) 
        grad_r = grad.reshape((O_C,  1, GRAD_W, GRAD_H))
        kernel_size = w.shape[2], w.shape[3]
    else:
        # OWHC
        W, H, I_C = shape[-3], shape[-2], shape[-1]
        x_r = x.reshape((I_C, H, W, 1)) 
        GRAD_W, GRAD_H = grad.shape[1], grad.shape[2]
        grad_r = grad.reshape((O_C, GRAD_W, GRAD_H, 1))
        kernel_size = w.shape[1], w.shape[2]
        print(x_r, grad_r)
    # compute conv2d output dimmensions, to avoid using -1 for shape inference. 
    import math
    w_out = (math.floor((H + 2 * padding[0] - kernel_size[0]) / stride[0]) + 1)
    h_out = (math.floor((W + 2 * padding[1] - kernel_size[1]) / stride[1]) + 1)

    dz = grad.reshape((O_C, w_out*h_out))
    dz = sum(dz, dim=1)
    dw = conv2d(x_r, grad_r, None, stride, padding )
    dx = conv2d_transpose(grad, w, None, stride, padding ) 
    return dx, dw, dz

def max_pool2d_diff_op(args, grad):
    x, kernel_size, stride, padding = args
    grad_in = max_pool2d_diff(x, kernel_size, stride, padding, grad)
    return grad_in

def mse_diff_op(args, grad):
    y, y_hat = args
    yyhat = y - y_hat
    dldy = const(2, y.shape) * yyhat * grad
    dldy_hat = const(-2, y.shape) * yyhat * grad
    return dldy, dldy_hat

def sigmoid_diff_op(args, grad):
    z, a = args
    dldz = sigmoid_grad(z, grad)
    return dldz


def nll_loss(args, grad):
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
