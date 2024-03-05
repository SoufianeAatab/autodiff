from ops.functional import register_diff_op
from ops.diff import *

register_diff_op('matmul', linear_diff_op)
register_diff_op('mse', mse_diff_op)
register_diff_op('sigmoid', sigmoid_diff_op)
register_diff_op('conv2d', conv2d_diff_op)
register_diff_op('max_pool2d', max_pool2d_diff_op)
register_diff_op('nll_loss', nll_loss)
register_diff_op('log_softmax', log_softmax_diff)

# from examples.sine import sine_example
from examples.conv import conv_example

conv_example()
