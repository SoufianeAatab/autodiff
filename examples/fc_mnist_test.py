from ops.param import Param, grads
from ops.interpreter import Interpreter
from ops.functional import *
from ops.graph import backward
from utils import * 
import torch
import os

def torch_model_example(x, y, w1, w2):
    # Convert inputs to float32
    x = x.float()
    y = y.long()  # y should be long for nll_loss (class indices)
    w1 = w1.float()
    w2 = w2.float()
    
    a = x @ w1.T
    b = torch.nn.functional.sigmoid(a)
    c = b @ w2.T
    d = torch.nn.functional.log_softmax(c, dim=1)
    out = torch.nn.functional.nll_loss(d, y)
    print(out)
    out.retain_grad()
    d.retain_grad()
    c.retain_grad()
    b.retain_grad()
    a.retain_grad()
    out.backward()
    return a, b, c, d, out

def mnist_example():
    # Data placeholder
    torch.manual_seed(42)
    y = torch.zeros(1,10)
    x = torch.rand(1,784)
    w = torch.rand(32,784, requires_grad=True) * 0.1
    w22 = torch.rand(10,32, requires_grad=True) * 0.1
    y[0,4] = 1; y_test = torch.randint(0,10, (1,)); y_test[0] = 4

    ## MODEL
    a, b, c, d, out = torch_model_example(x, y_test, w, w22)
    
    input = Param(x.numpy(), shape=(1,784), require_grads=False)
    w1 = Param(w.detach().numpy(), shape=(32,784), require_grads=True)
    w2 = Param(w22.detach().numpy(), shape=(10,32), require_grads=True)
    output = Param(y.numpy(), shape=(1,10), require_grads=False)

    params = [input, output, w1, w2]

    z1 = matmul(input, w1.t())
    z2 = sigmoid(z1)
    z3 = matmul(z2, w2.t())
    z4 = log_softmax(z3)
    z5 = nll_loss(z4, output)

    # backward function receives the last operator of the model and its inputs
    graph = backward(z5, [w1.id, w2.id]).build()

    interpreter = Interpreter(graph, input, output, params)
    code = interpreter.gen_code()
    epochs = 5
    training_size = 60000
    code = gen_training_loop(code, interpreter, epochs, training_size)

    output_size = len(y.reshape(-1))

    tests = [(z1, a), (z2, b), (z3, c), (z4, d)]
    for i, (p_a, p_b) in enumerate(tests):
        name = f"test_{i+1}" 
        name_grads = f"{name}_grads"
        code += generate_c_buff(name, p_b)
        code += generate_c_buff(name_grads, p_b.grad)
        code += generate_c_test(interpreter.mem[p_a.id], name, p_a.shape[1])
        code += generate_c_test(interpreter.mem[grads[p_a.id].id], name_grads, len(p_b.grad.reshape(-1)))

    generate_c_file('mnist', code)
    # os.system("clang -std=c99 -O3 code/mnist.c && ./a.out")
    # TODO: grads dictionary is being shared globally, fix this.
    # interpreter.gen_sgd(grads, para_grads)
