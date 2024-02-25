from ops.param import Param, grads
from ops.interpreter import Interpreter
from ops.functional import *
from ops.graph import backward

def conv_example():
    # Data placeholder
    input = Param(None, var_name='input', shape=(3, 32, 32))
    output = Param(None, var_name='output', shape=(1,10))

    # OIHW
    kernels = Param(None, shape=(8, 3, 3, 3), var_name='kernels')
    bias = Param(None, shape=(1, 8), var_name='bias')
    w1 = Param(None, shape=(10, 8*30*30), var_name='w1')

    z1 = conv2d(input, kernels, bias, 3, 0, 0)
    a1 = sigmoid(z1)
    z2 = matmul(a1, w1.t())
    a2 = log_softmax(z2)
    loss = cross_entropy(a2, output)
    graph = backward(loss, [w1.id, kernels.id, bias.id, input.id])
    dx_graph = graph.build()
    interpreter = Interpreter(dx_graph)
    interpreter.gen_torch_code()