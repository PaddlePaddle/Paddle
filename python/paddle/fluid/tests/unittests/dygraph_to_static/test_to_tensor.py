#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import numpy
import paddle
import unittest
import os
import tempfile
import paddle.inference as paddle_infer
from paddle.fluid.framework import program_guard, Program
import numpy as np


paddle.enable_static()

@paddle.jit.to_static
def tensor_badreturn_0(x):
    a = paddle.to_tensor([1.0, 2.0, 3.0], dtype="int64")
    return a

@paddle.jit.to_static
def tensor_badreturn_1(x):
    paddle.set_default_dtype("float64")
    a = paddle.to_tensor([1.0, 2.0, 3.0])
    return a 


class TestToTensorReturnVal(unittest.TestCase):

    def _run(self, to_static):
        prog_trans = paddle.jit.ProgramTranslator()
        prog_trans.enable(to_static)
        x = paddle.to_tensor([3])
        out0 = tensor_badreturn_0(x)
        out1 = tensor_badreturn_1(x)
        return out0, out1
    
    def test_to_tensor_badreturn(self):
        dygraph_res = self._run(to_static=False)
        x = paddle.to_tensor([3])
        self.assertTrue(dygraph_res[0].dtype == tensor_badreturn_0(x).dtype,
                        msg='to_static dtype is {}, orig dtype is {}'.format(
                            dygraph_res[0].dtype, tensor_badreturn_0(x).dtype))
        self.assertTrue(dygraph_res[1].dtype == tensor_badreturn_1(x).dtype,
                        msg='to_static dtype is {}, orig dtype is {}'.format(
                            dygraph_res[0].dtype, tensor_badreturn_1(x).dtype))
        

class UnittestBase(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.init_info()

    def tearDwon(self):
        self.temp_dir.cleanup()

    def init_info(self):
        self.shapes = None
        self.save_path = None

    def infer_prog(self):
        config = paddle_infer.Config(self.save_path + '.pdmodel',
                                     self.save_path + '.pdiparams')
        predictor = paddle_infer.create_predictor(config)
        input_names = predictor.get_input_names()
        for i, shape in enumerate(self.shapes):
            input_handle = predictor.get_input_handle(input_names[i])
            fake_input = np.random.randn(*shape).astype("float32")
            input_handle.reshape(shape)
            input_handle.copy_from_cpu(fake_input)
        predictor.run()
        output_names = predictor.get_output_names()
        output_handle = predictor.get_output_handle(output_names[0])
        output_data = output_handle.copy_to_cpu()

        return output_data


class TestDropout(UnittestBase):

    def init_info(self):
        self.shapes = [[10, 10]]
        self.save_path = os.path.join(self.temp_dir.name, 'dropout')

    def test_static(self):
        main_prog = Program()
        starup_prog = Program()
        with program_guard(main_prog, starup_prog):
            fc = paddle.nn.Linear(10, 10)
            x = paddle.randn(self.shapes[0])
            x.stop_gradient = False
            feat = fc(x)
            # p is a Variable
            p = paddle.to_tensor([1])
            out = paddle.nn.functional.dropout(feat, p=p)
            sgd = paddle.optimizer.SGD()
            sgd.minimize(paddle.mean(out))
            # test _to_string
            self.assertTrue("Var[" in str(main_prog))

            exe = paddle.static.Executor()
            exe.run(starup_prog)
            res = exe.run(fetch_list=[x, out])
            # export model
            paddle.static.save_inference_model(self.save_path, [x], [out], exe)

            # Test for Inference Predictor
            infer_out = self.infer_prog()
            self.assertEqual(infer_out.shape, (10, 10))


if __name__ == '__main__':
    unittest.main()
