from ops.param import Param, grads
from ops.interpreter import run
from ops.functional import *
from ops.graph import backward

import torch
import torch.nn as nn
class XorPytorch(nn.Module):
    def __init__(self):
        super(XorPytorch, self).__init__()

        self.l1 = nn.Linear(2,8, bias=False)
        self.l2 = nn.Linear(8,1, bias=False)
        torch.manual_seed(0)
        self.l1.weight = nn.Parameter(torch.rand(8, 2))
        self.l2.weight = nn.Parameter(torch.rand(1, 8))
  

    def forward(self, x):
        x = self.l1(x)
        x = torch.nn.functional.sigmoid(x)
        x = self.l2(x)
        return x

def xor_example():
    # Data placeholder
    input = Param(None, var_name='input', shape=(1,1))
    output = Param(None, var_name='output', shape=(1,1))
    print(input)
    print(output)
    # x = torch.rand(4,1)
    # y = torch.sin(x)

    w1 = Param(torch.rand(8,1), shape=(8,1), var_name='w1', print_init=True)
    # w2 = Param(torch.rand(1,8), var_name='w2', print_init=True)

    w_t = w1.t()
    z = matmul(input,w_t)
    a = sigmoid(z)
    # z2 = matmul(a, w2.t())
    # a2 = sigmoid(z2)
    # loss = mse(a2, output)

    # input.data = x[0].view(1,-1)
    # output.data = y[0].view(1,-1)

    graph = backward(a, [w_t.id, input.id])
    dx_graph = graph.build()
    for dx in dx_graph:
        print(dx)
    print("#"*40)
    # dw_graph = graph[1].build()
    # for dw in dw_graph:
    #     print(dw._op)
    # print(loss, loss._prev)
    # topo = loss.build()
    # for p in topo:
    #     print(p._op)
    # run(ops)
    # print(grads)
    # print(b_ops)

    # lr = 0.1
    # for epoch in range(1):
    #     running_loss = 0
    #     for i in range(1):
    #         input.data.data = x[i].view(1,-1)
    #         output.data.data = y[i].view(1,-1)
    #         run(ops)
    #         loss.backward()

    #         w1.data.data -= w1.grad.data * lr
    #         # w2.data.data -= w2.grad.data * lr

    #         # l1.b.data -= l1.b.grad.data * lr
    #         # l2.b.data -= l2.b.grad.data * lr
    #         running_loss += loss.data.data

    #     print(f'{epoch} loss: {loss.data.data.item() / 4.0}')