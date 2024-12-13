from ops.param import Param, grads
from ops.interpreter import Interpreter
from ops.functional import *
from ops.graph import backward

def face_detection_example():
    # Data placeholder
    input_size = 240*320
    input = Param(None, var_name='input', shape=(1,input_size))
    target = Param(None, var_name='target', shape=(1,1))

    w1 = Param(None, shape=(16,input_size), var_name='w1', print_init=True)
    w2 = Param(None, shape=(1,16), var_name='w2', print_init=True)
    #w11 = Param(None, shape=(32,1024), var_name='w3', print_init=True)

    # b1 = Param(None, shape=(1,8), var_name='b1', print_init=True)
    # b2 = Param(None, shape=(1,1), var_name='b2', print_init=True)


    #print(input, target, w1, w2, b1, b2)

    w_t = w1.t()
    z = matmul(input,w_t)
    a = sigmoid(z)
    #z11 = matmul(a, w11.t())
    #a11 = sigmoid(z11)
    z2 = matmul(a, w2.t())
    a2 = sigmoid(z2)
    loss = binary_cross_entropy(a2, target)

    # backward function receives the last operator of the model and its inputs
    graph = backward(loss, [w1.id, w2.id])
    dx_graph = graph.build()

    interpreter = Interpreter(dx_graph, [w1, w2, input, target])
    # interpreter.gen_torch_code()
    param_grads = [w1.id, w2.id]
    interpreter.gen_init_params([w1, w2])
    interpreter.gen_code()
    interpreter.gen_sgd(grads, param_grads)
