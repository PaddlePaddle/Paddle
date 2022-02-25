from __future__ import print_function
import unittest
import paddle
import numpy as np
import sys
sys.path.append("..")

from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class,get_xpu_op_spport_types,XPUOpTestWrapper

paddle.enable_static()
SEED = 2022


class XPUTestReluOp(XPUOpTestWrapper):
    def __init__(self) -> None:
        self.op_name="relu"
        self.use_dynamic_create_class = True
    
    class TestReluOp(XPUOpTest):
        def setUp(self):
            self.set_xpu()
            self.op_type = "relu"
            self.init_dtype()
            self.init_shape()
            self.init_config()
            np.random.seed(SEED)

        def init_shape(self):
            self.shape=(3,2)

        def init_dtype(self):
            self.dtype = self.in_type

        def init_config(self):
            x =np.random.standard_normal(self.shape).astype(self.dtype)
            self.inputs = {'X': self.np_dtype_to_fluid_dtype(x)}
            self.outputs = {'Out': np.maximum(0,x)}

        def set_xpu(self):
            self.__class__.use_xpu = True
            self.__class__.no_need_check_grad = True

        def test_check_output(self):
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place)
    
    class TestReluOp1(TestReluOp):
        def init_shape(self):
            self.shape=(2)


    class TestReluOp1(TestReluOp):
        def init_shape(self):
            self.shape=(2,3,4)


    class TestReluOp1(TestReluOp):
        def init_shape(self):
            self.shape=(2,3,4,4)

    class TestReluOp1(TestReluOp):
        def init_shape(self):
            self.shape=(2,3,4,4,5)


support_types = get_xpu_op_spport_types("relu")
for stype in support_types:
    create_test_class(globals(), XPUTestReluOp, stype)
    

if __name__ == "__main__":
    unittest.main()
