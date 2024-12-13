from ops.param import Param, grads
from ops.interpreter import Interpreter
from ops.functional import *
from ops.graph import backward

def conv_example():

    settings['CONV_ORDER'] = ConvOrder.OHWC

    in_h = 240
    in_w = 320
    in_ch = 1
    out_ch = 8
    # Data placeholder
    if settings['CONV_ORDER'] == ConvOrder.OHWC:
        input = Param(None, var_name='input', shape=(in_h, in_w, in_ch))
        kernels = Param(None, shape=(out_ch, 3, 3, in_ch), var_name='kernels')
    else:
        # w = permute(0,2,3,1)
        # x = permute(1,2,0)
        input = Param(None, var_name='input', shape=(in_ch, in_h, in_w))
        kernels = Param(None, shape=(out_ch, in_ch, 3, 3), var_name='kernels')


    output = Param(None, var_name='output', shape=(1,1))
    # OIHW
    bias = Param(None, shape=(1, out_ch), var_name='bias')
    w1 = Param(None, shape=(1, out_ch*119*159), var_name='w1')

    x = conv2d(input, kernels, bias, (1,1), (0,0))
    x = max_pool2d(x, kernel_size=(2,2), stride=(2,2), padding=(0,0))
    x = sigmoid(x)
    x = x.reshape((1, out_ch*119*159))
    x = matmul(x, w1.t())
    x = sigmoid(x)
    loss = binary_cross_entropy(x, output)

    graph = backward(loss, [output.id, kernels.id, bias.id, input.id, w1.id])
    
    dx_graph = graph.build()
    interpreter = Interpreter(dx_graph, [kernels, bias, input, output, w1])
    # interpreter.gen_torch_code()
    interpreter.gen_init_params([w1, kernels, bias])
    interpreter.gen_code()
    interpreter.gen_sgd(grads, [w1.id, kernels.id, bias.id])
    # print(kernels, interpreter.mem[kernels.id])

    # print(grads[kernels.id], grads[bias.id], grads[w1.id], grads[input.id])