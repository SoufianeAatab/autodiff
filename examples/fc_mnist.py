from ops.param import Param, grads
from ops.interpreter import Interpreter
from ops.functional import *
from ops.graph import backward

import numpy as np
def np_sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def mnist_example():
    # Data placeholder
    np.random.seed(42)
    x = np.random.randint(4, size=(1,4))
    w = np.random.randint(4, size=(4,4))
    w2 = np.random.randint(4, size=(17,4))
    a = np.matmul(x, w.T)
    a = np_sigmoid(a)
    a = np.matmul(a, w2.T)
    out = np_sigmoid(a)
    print("OUTPUT=", out)
    input = Param(x, shape=(1,4), require_grads=False)
    output = Param(None, var_name='output', shape=(1,4), require_grads=False)
    w1 = Param(w, shape=(4,4), require_grads=False)
    w2 = Param(w2, shape=(17,4), var_name='w2', require_grads=False)
    params = [input, output, w1, w2]

    w_t = w1.t()
    z = matmul(input, w_t)
    z = sigmoid(z)
    z = matmul(z, w2.t())
    z = sigmoid(z)
    print("Model output shape is", z.shape)

    # a = sigmoid(z)
    # z2 = matmul(a, w2.t())
    # a2 = log_softmax(z2)
    # loss = nll_loss(a2, output)

    # backward function receives the last operator of the model and its inputs
    # graph = backward(loss, [w1.id, w2.id]).build()

    interpreter = Interpreter(z.build(), input, output, params)
    code = interpreter.gen_code()
    output_size = len(out.reshape(-1))
    print(code)
    print(f"float ground[{output_size}] = ")
    elems = "{"
    for elem in out.reshape(-1):
        elems += f"{elem}, "
    print(elems)
    print("};")
    print(f"bool test = check(&buf[{interpreter.mem[z.id]}], ground, {output_size});\n")
    print('if(test) printf("OK!");else printf("NO OK!");\n')
    # TODO: grads dictionary is being shared globally, fix this.
    # interpreter.gen_sgd(grads, para_grads)
