#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import numpy as np
import unittest
import sys
sys.path.append("..")
from op_test import OpTest, _set_use_system_allocator
import paddle
import paddle.fluid as fluid

paddle.enable_static()
SEED = 2021


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestGatherOp(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "gather"
        self.place = paddle.NPUPlace(0)
        self.init_dtype()
        self.init_input_output()

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(self.x),
            'Index': OpTest.np_dtype_to_fluid_dtype(self.index)
        }
        self.attrs = {'validate_indices': True}
        self.outputs = {'Out': self.out}

    def set_npu(self):
        self.__class__.use_npu = True

    def init_input_output(self):
        self.x = np.array([[1, 2], [3, 4], [5, 6]]).astype(self.dtype)
        self.index = np.array([1, 2]).astype(np.int)
        self.out = np.array([[3, 4], [5, 6]]).astype(self.dtype)

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestGatherAPI(unittest.TestCase):
    def test_name(self):
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name="x", shape=[3, 2], dtype="float32")
            index = paddle.static.data(name='index', shape=[1], dtype='int32')

            out = paddle.gather(x, index, name='gather')
            self.assertEqual(('gather' in out.name), True)

    def test_static(self):
        with paddle.static.program_guard(paddle.static.Program()):

            x_np = np.array([[1, 2], [3, 4], [5, 6]]).astype('float32')
            index_np = np.array([1, 2]).astype('int32')

            x = paddle.static.data(name="x", shape=[3, 2], dtype='float32')
            index = paddle.static.data(name="index", shape=[2], dtype='int32')

            z = paddle.gather(x, index)

            place = paddle.NPUPlace(0)
            exe = paddle.static.Executor(place)
            x_value, index_value, z_value = exe.run(
                feed={"x": x_np,
                      "index": index_np}, fetch_list=[x, index, z])

            z_expected = np.array([[3, 4], [5, 6]])
            self.assertEqual(
                (x_value == x_np).all(),
                True,
                msg="x_value = {}, but expected {}".format(x_value, x_np))
            self.assertEqual(
                (index_value == index_np).all(),
                True,
                msg="index_value = {}, but expected {}".format(index_value,
                                                               index_np))
            self.assertEqual(
                (z_value == z_expected).all(),
                True,
                msg="z_value = {}, but expected {}".format(z_value, z_expected))

    def test_backward(self):
        # TODO(ascendrc): Test backward after add grad npu op implemented.
        pass


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestPowNet(unittest.TestCase):
    def _test(self, run_npu=True):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        np.random.seed(SEED)

        a_np = np.random.random(size=(8192, 768)).astype('float32')
        index_np = np.random.randint(0, 8192, size=(1232, 1)).astype('int32')

        with paddle.static.program_guard(main_prog, startup_prog):
            a = paddle.static.data(name="a", shape=[8192, 768], dtype='float32')
            index = paddle.static.data(
                name="index", shape=[1232, 1], dtype='int32')
            a.stop_gradient = False
            b = paddle.gather(a, index)

            loss = fluid.layers.reduce_mean(b)
            sgd = fluid.optimizer.SGD(learning_rate=0.01)
            sgd.minimize(loss)

        if run_npu:
            place = paddle.NPUPlace(0)
        else:
            place = paddle.CPUPlace()

        exe = paddle.static.Executor(place)
        exe.run(startup_prog)

        print("Start run on {}".format(place))
        for epoch in range(100):

            pred_res, loss_res = exe.run(main_prog,
                                         feed={"a": a_np,
                                               "index": index_np},
                                         fetch_list=[b, loss])
            if epoch % 10 == 0:
                print("Epoch {} | Prediction[0]: {}, Loss: {}".format(
                    epoch, pred_res[0], loss_res[0]))

        return pred_res, loss_res

    def test_npu(self):
        npu_pred, npu_loss = self._test(True)
        cpu_pred, cpu_loss = self._test(False)

        self.assertTrue(np.allclose(npu_pred, cpu_pred))
        self.assertTrue(np.allclose(npu_loss, cpu_loss))


if __name__ == '__main__':
    unittest.main()
