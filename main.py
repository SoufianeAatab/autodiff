from ops.functional import register_diff_op
from ops.diff import *

register_diff_op('matmul', linear_diff_op)
register_diff_op('mse', mse_diff_op)
register_diff_op('sigmoid', sigmoid_diff_op)

from examples.sine import sine_example

sine_example()