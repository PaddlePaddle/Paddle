import unittest, os
import numpy as np
import paddle.v2 as paddle
from paddle.v2.framework.op import Operator
import paddle.v2.framework.core as core
from op_test import OpTest, create_op, set_input

gpu_list = "0,1,2,3"
