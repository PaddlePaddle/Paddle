import unittest, os
import numpy as np
import paddle.v2 as paddle
from paddle.v2.framework.op import Operator
import paddle.v2.framework.core as core
from op_test import OpTest, create_op

gpu_list = os.environ["NV_LIST"]

if not core.is_compile_gpu() or not gpu_list:
    exit(0)


class TestNCCLAllReduce(unittest.TestCase):
    def __init__(self):
        self.op_type = "nnclAllReduce"
        self.scope = core.Scope()
