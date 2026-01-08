from ops.param import Param, grads
from ops.interpreter import Interpreter
from ops.functional import *
from ops.graph import backward

def mnist_example():
    # Data placeholder
    input = Param(None, var_name='input', shape=(1,4), require_grads=False)
    output = Param(None, var_name='output', shape=(1,4), require_grads=False)
    print(input, output)

    w1 = Param(None, shape=(4,4), var_name='w1', require_grads=False)
    #w2 = Param(None, shape=(10,32), var_name='w2', require_grads=False)

    params = [input, output, w1]

    w_t = w1.t()
    z = matmul(input,w_t) # (1,784)x(784,32) => (1,32)
    # a = sigmoid(z)
    # z2 = matmul(a, w2.t())
    # a2 = log_softmax(z2)
    # loss = nll_loss(a2, output)

    # backward function receives the last operator of the model and its inputs
    # graph = backward(loss, [w1.id, w2.id]).build()

    interpreter = Interpreter(z.build(), input, output, params)
    code = interpreter.gen_code()
    print(code)
    # TODO: grads dictionary is being shared globally, fix this.
    # interpreter.gen_sgd(grads, para_grads)
