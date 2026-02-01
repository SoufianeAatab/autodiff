from ops.param import Param, grads
from ops.interpreter import Interpreter
from ops.functional import *
from ops.graph import backward
from utils import * 
import numpy as np
import torch
import os

def np_sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def mnist_example():
    # Data placeholder
    np.random.seed(42)
    y = torch.zeros(1,17)
    x = torch.rand(1,4)
    w = torch.rand(4,4, requires_grad=True)
    w22 = torch.rand(17,4, requires_grad=True)
    y[0,4] = 1
    y_test = torch.randint(0,17, (1,))
    y_test[0] = 4
    ## MODEL
    a = x @ w.T
    b = torch.nn.functional.sigmoid(a)
    c = b @ w22.T
    d = torch.nn.functional.log_softmax(c, dim=1)
    out = torch.nn.functional.nll_loss(d, y_test) # same y but torch expect one integr
    print(out)
    out.retain_grad()
    d.retain_grad()
    c.retain_grad()
    b.retain_grad()
    a.retain_grad()
    
    out.backward()
    
    input = Param(x.numpy(), shape=(1,4), require_grads=False)
    w1 = Param(w.detach().numpy(), shape=(4,4), require_grads=True)
    w2 = Param(w22.detach().numpy(), shape=(17,4), require_grads=True)
    output = Param(y.numpy(), shape=(1,17), require_grads=False)
    params = [input, output, w1, w2]

    z1 = matmul(input, w1.t())
    z2 = sigmoid(z1)
    z3 = matmul(z2, w2.t())
    z4 = log_softmax(z3)
    z5 = nll_loss(z4, output)
    # assert z.shape == out.shape, f"Model output mismatch {z.shape}!={out.shape}"

    # backward function receives the last operator of the model and its inputs
    graph = backward(z5, [w1.id, w2.id]).build()

    interpreter = Interpreter(graph, input, output, params)
    code = interpreter.gen_code()

    output_size = len(y.reshape(-1))

    tests = [(z1, a), (z2, b), (z3, c), (z4, d)]
    for i, (p_a, p_b) in enumerate(tests):
        name = f"test_{i+1}" 
        name_grads = f"{name}_grads"
        code += generate_c_buff(name, p_b)
        code += generate_c_buff(name_grads, p_b.grad)
        code += generate_c_test(interpreter.mem[p_a.id], name, p_a.shape[1])
        code += generate_c_test(interpreter.mem[grads[p_a.id].id], name_grads, len(p_b.grad.reshape(-1)))

    generate_c_file('test2', code)
    os.system("clang -std=c99 code/test2.c && ./a.out")
    # TODO: grads dictionary is being shared globally, fix this.
    # interpreter.gen_sgd(grads, para_grads)
