#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import unittest
import sys

sys.path.append("..")
from op_test import OpTest, skip_check_grad_ci
import paddle
import paddle.fluid as fluid

paddle.enable_static()

SEED = 2022


class TestDropoutOp(OpTest):

    def setUp(self):
        self.set_mlu()
        self.init_dtype()
        self.init_inputs_shape()
        self.init_attrs()
        self.op_type = 'dropout'
        self.inputs = {'X': np.random.random(self.shape).astype(self.dtype)}
        self.attrs = {
            'dropout_prob': self.dropout_prob,
            'fix_seed': self.fix_seed,
            'is_test': self.is_test,
            'dropout_implementation': self.dropout_implementation
        }

        out = self.inputs['X'] * (1.0 - self.dropout_prob)
        if self.is_test == False:
            mask = None
            if self.dropout_prob == 0.0:
                mask = np.ones(self.shape).astype('uint8')
            elif self.dropout_prob == 1.0:
                mask = np.zeros(self.shape).astype('uint8')
            self.outputs = {'Out': out, 'Mask': mask}
        else:
            self.outputs = {'Out': out}

    def init_dtype(self):
        self.dtype = np.float32

    def init_inputs_shape(self):
        self.shape = [32, 64]

    def init_attrs(self):
        self.__class__.no_need_check_grad = False
        self.dropout_prob = 0.0
        self.fix_seed = True
        self.is_test = False
        self.dropout_implementation = "upscale_in_train"

    def set_mlu(self):
        self.__class__.use_mlu = True
        self.place = paddle.device.MLUPlace(0)

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        if hasattr(self.__class__, "no_need_check_grad"
                   ) and self.__class__.no_need_check_grad == True:
            return

        self.check_grad_with_place(self.place, ['X'], 'Out')


class TestDropoutOpInput1d(TestDropoutOp):

    def init_inputs_shape(self):
        self.shape = [2000]


class TestDropoutOp2(TestDropoutOp):

    def init_inputs_shape(self):
        self.shape = [32, 64]

    def init_attrs(self):
        self.dropout_prob = 1.0
        self.fix_seed = True
        self.is_test = False
        self.dropout_implementation = "upscale_in_train"


class TestDropoutOp3(TestDropoutOp):

    def init_inputs_shape(self):
        self.shape = [32, 64, 2]


class TestDropoutOp4(TestDropoutOp):

    def init_attrs(self):
        self.__class__.no_need_check_grad = True
        self.dropout_prob = 0.35
        self.fix_seed = True
        self.is_test = True
        self.dropout_implementation = "downgrade_in_infer"


class TestDropoutOp5(TestDropoutOp):

    def init_inputs_shape(self):
        self.shape = [32, 64, 3]

    def init_attrs(self):
        self.__class__.no_need_check_grad = True
        self.dropout_prob = 0.75
        self.fix_seed = True
        self.is_test = True
        self.dropout_implementation = "downgrade_in_infer"


class TestDropoutOp6(TestDropoutOp):

    def init_attrs(self):
        self.__class__.no_need_check_grad = True
        self.dropout_prob = 0.0
        self.fix_seed = True
        self.is_test = False
        self.dropout_implementation = "downgrade_in_infer"


class TestDropoutOpWithSeed(TestDropoutOp):
    # the seed is a Tensor
    def setUp(self):
        self.op_type = "dropout"
        self.set_mlu()
        self.dtype = np.float32
        self.inputs = {
            "X": np.random.random((32, 64)).astype(self.dtype),
            "Seed": np.asarray([125], dtype="int32")
        }
        self.attrs = {
            'dropout_prob': 0.0,
            'is_test': False,
            'dropout_implementation': 'upscale_in_train'
        }
        self.outputs = {
            'Out': self.inputs['X'],
            'Mask': np.ones((32, 64)).astype('uint8')
        }

    def set_mlu(self):
        self.__class__.use_mlu = True
        self.place = paddle.device.MLUPlace(0)

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        self.check_grad_with_place(self.place, ['X'], 'Out')


class TestDropoutOpFp16(TestDropoutOp):
    # float16
    def init_dtype(self):
        self.dtype = np.float16

    def set_mlu(self):
        self.__class__.use_mlu = True
        self.place = paddle.device.MLUPlace(0)
        self.__class__.no_need_check_grad = True


@skip_check_grad_ci(reason="For inference, check_grad is not required.")
class TestDropoutOpInference(OpTest):
    # is_test = True
    def setUp(self):
        self.op_type = "dropout"
        self.set_mlu()
        self.init_dtype()
        self.inputs = {'X': np.random.random((32, 64)).astype(self.dtype)}
        self.attrs = {
            'dropout_prob': 0.35,
            'fix_seed': True,
            'is_test': True,
            'dropout_implementation': 'upscale_in_train'
        }
        self.outputs = {'Out': self.inputs['X']}

    def init_dtype(self):
        self.dtype = np.float32

    def set_mlu(self):
        self.__class__.use_mlu = True
        self.place = paddle.device.MLUPlace(0)

    def test_check_output(self):
        self.check_output_with_place(self.place)


@skip_check_grad_ci(reason="For inference, check_grad is not required.")
class TestDropoutOpInference2(TestDropoutOpInference):

    def setUp(self):
        self.op_type = "dropout"
        self.set_mlu()
        self.init_dtype()
        self.inputs = {'X': np.random.random((32, 64, 3)).astype(self.dtype)}
        self.attrs = {
            'dropout_prob': 0.75,
            'is_test': True,
            'dropout_implementation': 'upscale_in_train'
        }
        self.outputs = {'Out': self.inputs['X']}


class TestDropoutAPI(unittest.TestCase):

    def setUp(self):
        np.random.seed(123)
        self.places = [fluid.CPUPlace(), paddle.device.MLUPlace(0)]

    def check_static_result(self, place):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            input = fluid.data(name="input", shape=[40, 40], dtype="float32")
            res1 = paddle.nn.functional.dropout(x=input,
                                                p=0.,
                                                training=False,
                                                mode='upscale_in_train')
            res2 = paddle.nn.functional.dropout(x=input,
                                                p=0.,
                                                axis=0,
                                                training=True,
                                                mode='upscale_in_train')
            res3 = paddle.nn.functional.dropout(x=input,
                                                p=0.,
                                                axis=0,
                                                training=False,
                                                mode='upscale_in_train')
            res4 = paddle.nn.functional.dropout(x=input,
                                                p=0.,
                                                axis=[0, 1],
                                                training=True,
                                                mode='upscale_in_train')
            res5 = paddle.nn.functional.dropout(x=input,
                                                p=0.,
                                                axis=[0, 1],
                                                training=False,
                                                mode='upscale_in_train')
            res6 = paddle.nn.functional.dropout(x=input,
                                                p=1.,
                                                training=True,
                                                mode='upscale_in_train')
            res7 = paddle.fluid.layers.dropout(
                x=input,
                dropout_prob=0.,
                dropout_implementation='upscale_in_train')
            res8 = paddle.nn.functional.dropout(x=input,
                                                p=0.,
                                                axis=(0, 1),
                                                training=False,
                                                mode='upscale_in_train')

            in_np = np.random.random([40, 40]).astype("float32")
            res_np = in_np
            res_np2 = np.zeros_like(in_np)

            exe = fluid.Executor(place)
            res_list = [res1, res2, res3, res4, res5, res7, res8]
            for res in res_list:
                fetches = exe.run(fluid.default_main_program(),
                                  feed={"input": in_np},
                                  fetch_list=[res])
                np.testing.assert_allclose(fetches[0], res_np)
            fetches2 = exe.run(fluid.default_main_program(),
                               feed={"input": in_np},
                               fetch_list=[res6])
            np.testing.assert_allclose(fetches2[0], res_np2)

    def test_static(self):
        for place in self.places:
            self.check_static_result(place=place)


if __name__ == '__main__':
    unittest.main()
