from ops.param import Param, Op, grads

ops = {}
def get_diff_op(op_name):
    assert op_name in ops, f"No diff operation found for {op_name}"
    return ops[op_name]

def register_diff_op(op_name, fnc):
    ops[op_name] = fnc

def ones_like(a):
    p = Param(None, children=(a,), shape=(a.shape[1], a.shape[0]), var_name=a.var_name + '_t', print_init=False)
    op = Op('ones_like', a, b=None)
    p._op = op
    # ops.append(op)
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
    z = Op('matmul', a, b)
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
    z = Op('sigmoid', x, b=None)
    p._op = z
    p._backward_op = get_diff_op('sigmoid')
    def backward():
        grad_out = grads[p.id] # read grad out
        grads[z.a.id] = p._backward_op((z.a, p), grad_out) # call grad_fn and save the output
    p._backward = backward
    return p

# def mse(y, y_pred):
#     p = Param(None, shape=y_pred.shape, children=(y, y_pred), var_name='mse_',print_init=False)
#     p._op = Op('mse', y_pred, b=y)
#     p._backward_op = get_diff_op('mse')

#     def backward():
#         grad_out = grads[z.id]

#         dldy, dldy_hat = z._backward_op((z.a, z.b), grad_out)
#         grads[z.a.id] = dldy
#         grads[z.b.id] = dldy_hat
#         # z.a = dldy
#         # z.b = dldy_hat
#     p._backward = backward
#     return p

# def mse_grad(y, y_pred):
#     dldy = Param(None, shape=y_pred.shape, children=(y, y_pred))
#     dldy_hat = Param(None, shape=y_pred.shape, children=(y, y_pred))
#     op = Op('mse_grad', y, y_pred)
#     dldy._op = op
#     dldy_hat._op = op
#     return dldy, dldy_hat

def sigmoid_grad(x, a, grad):
    dx = Param(None, shape=x.shape, children=(a, x))
    op = Op('sigmoid_grad', x, grad)
    dx._op = op
    return dx