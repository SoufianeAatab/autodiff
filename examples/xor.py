from ops.param import Param, grads
from ops.interpreter import Interpreter
from ops.functional import *
from ops.graph import backward

def xor_example():
    # Data placeholder
    input = Param(None, var_name='input', shape=(1,1))
    output = Param(None, var_name='output', shape=(1,1))
    w1 = Param(None, shape=(8,1), var_name='w1', print_init=True)
    w2 = Param(None, shape=(1,8), var_name='w2', print_init=True)

    b1 = Param(None, shape=(1,8), var_name='b1', print_init=True)
    b2 = Param(None, shape=(1,1), var_name='b2', print_init=True)

    print(input, output, w1, w2, b1, b2)

    w_t = w1.t()
    z = matmul(input,w_t)
    z = z + b1
    a = sigmoid(z)
    z2 = matmul(a, w2.t())
    z2 = z2 + b2
    loss = mse(output, z2)

    # backward function receives the last operator of the model and its inputs
    graph = backward(loss, [w1.id, w2.id])
    dx_graph = graph.build()

    interpreter = Interpreter(dx_graph)
    interpreter.run()

    dw = grads[w1.id]
    dw2 = grads[w2.id]
    db1 = grads[b1.id]
    db2 = grads[b2.id]

    print("dw", dw)
    print("dw2", dw2)
    print("b1", db1)
    print("b2", db2)
    assert w1.shape == dw.shape
    assert w2.shape == dw2.shape

    # print("#"*40)
    # for dx in dx_graph:
    #     print(dx)
    # print("#"*40)