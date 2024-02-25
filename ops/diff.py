from ops.functional import matmul, sigmoid_grad


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

def mse_diff_op(args, grad):
    y, y_hat = args
    # print(f"%{y.grad.id}, %{y_hat.grad.id}=mse_diff(%{y.data.id}, %{y_hat.data.id}) * %{grad.grad.id}")
    # print(f"%{y.grad.id} = 2 * (%{y.data.id} - %{y_hat.data.id}) * %{grad.grad.id}")
    # print(f"%{y_hat.grad.id} = -2 * (%{y.data.id} - %{y_hat.data.id}) * %{grad.grad.id}")
    # y, y_hat, grad = y.data.data, y_hat.data.data, grad.grad.data
    # dldy = 2 * (y - y_hat) * grad
    # dldy_hat = -2 * (y - y_hat) * grad
    # dldy, dldy_hat = mse_grad()
    # -2(y-yhat)
    yyhat = y - y_hat
    dldy = 2 * yyhat * grad
    dldy_hat = -2 * yyhat * grad
    return dldy, dldy_hat

def sigmoid_diff_op(args, grad):
    z, a = args
    # print(f"%{z.grad.id}=sigmoid_diff(%{a.data.id}) * %{grad.grad.id}")
    # print(f"%{z.grad.id} = %{a.data.id} * (1.0 - %{a.data.id}) * %{grad.grad.id}")
    # z, a, grad = z.data.data, a.data.data, grad.grad.data
    # dldz = a * (1.0-a) * grad
    dldz = sigmoid_grad(z, grad)
    return dldz