from ops.param import Param, grads
from ops.interpreter import Interpreter
from ops.functional import *
from ops.graph import backward
import torch

from utils import gen_training_loop

def torch_model_example(x, filters, w):
    a = torch.nn.functional.conv2d(x, filters, bias=None, padding=0)
    b = torch.nn.functional.sigmoid(a)
    b = b.reshape(1, -1)
    c = b @ w.T
    return c

def conv_example():
    torch.manual_seed(42)
    settings['CONV_ORDER'] = ConvOrder.OCHW
    in_h = 5
    in_w = 5
    in_ch = 4
    out_ch = 8
    kernel_width = kernel_height = 3
    filters = torch.randn(out_ch, in_ch, kernel_height, kernel_width)
    x = torch.randn(1, in_ch, in_h, in_w)
    w = torch.randn(10, 3 * 3 * 8)
    torch_out = torch_model_example(x, filters, w)
    print(torch_out.reshape(-1))
    # Data placeholder
    if settings['CONV_ORDER'] == ConvOrder.OHWC:
        input = Param(None, var_name='input', shape=(in_h, in_w, in_ch))
        kernels = Param(None, shape=(out_ch, kernel_height, kernel_width, in_ch), var_name='kernels')
    else:
        input = Param(x.detach().numpy(), var_name='input', shape=(in_ch, in_h, in_w), require_grads=False)
        kernels = Param(filters.detach().numpy(), shape=(out_ch, in_ch, kernel_height, kernel_width), var_name='kernels', require_grads=False)


    weights = Param(w.detach().numpy(), shape=(10, 3*3*8), require_grads=False)
    bias = Param(None, shape=(1, out_ch), var_name='bias', require_grads=False)

    a = conv2d(input, kernels, bias, (1,1), (0,0))
    b = sigmoid(a)
    c = matmul(b.reshape((1, 3*3*8)), weights.t())
    graph = c.build()
    interpreter = Interpreter(graph, input, None, [kernels, bias, weights])    
    code = interpreter.gen_code()
    code = gen_training_loop(code, interpreter, 1, 1)

    # print(code)