import torch
import torch.nn as nn
from ops.functional import register_diff_op

torch.manual_seed(0)

from ops.diff import *

register_diff_op('matmul', linear_diff_op)
# register_diff_op('mse', mse_diff_op)
register_diff_op('sigmoid', sigmoid_diff_op)

from examples.xor import xor_example

xor_example()