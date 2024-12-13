from ops.param import Param, grads
from ops.interpreter import Interpreter
from ops.functional import *
from ops.graph import backward

def mnist_example():
    # Data placeholder
    input = Param(None, var_name='input', shape=(1,28*28))
    output = Param(None, var_name='output', shape=(1,10))

    w1 = Param(None, shape=(32,28*28), var_name='w1', print_init=True)
    w2 = Param(None, shape=(10,32), var_name='w2', print_init=True)

    w_t = w1.t()
    z = matmul(input,w_t)
    a = sigmoid(z)
    z2 = matmul(a, w2.t())
    a2 = log_softmax(z2)
    loss = nll_loss(a2, output)

    # backward function receives the last operator of the model and its inputs
    graph = backward(loss, [w1.id, w2.id])
    dx_graph = graph.build()

    interpreter = Interpreter(dx_graph, [w1, w2, input, output])
    # interpreter.gen_torch_code()
    interpreter.gen_init_params([w1, w2])
    interpreter.gen_code()
    param_grads = [w1.id, w2.id]
    interpreter.gen_sgd(grads, param_grads)
