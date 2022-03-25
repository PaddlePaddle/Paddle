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
from op_test import OpTest
import paddle

paddle.enable_static()

SEED = 2021
EPOCH = 100


class TestSliceOp(OpTest):
    def setUp(self):
        self.op_type = "slice"
        self.set_npu()
        self.init_dtype()
        self.config()
        self.inputs = {'Input': self.input}
        self.outputs = {'Out': self.out}
        self.attrs = {
            'axes': self.axes,
            'starts': self.starts,
            'ends': self.ends,
            'infer_flags': self.infer_flags
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype(self.dtype)
        self.starts = [1, 0, 2]
        self.ends = [3, 3, 4]
        self.axes = [0, 1, 2]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[1:3, 0:3, 2:4, :]

    def init_dtype(self):
        self.dtype = np.float32

    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(0)

    def test_check_output(self):
        if self.dtype == np.float16:
            self.check_output_with_place(self.place)
        else:
            self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        if self.dtype == np.float16:
            self.check_grad_with_place(
                self.place, ['Input'], 'Out', max_relative_error=0.02)
        else:
            self.check_grad_with_place(self.place, ['Input'], 'Out')


class TestSliceOp2(TestSliceOp):
    def config(self):
        self.input = np.random.random([10, 5, 6]).astype(self.dtype)
        self.starts = [0]
        self.ends = [1]
        self.axes = [1]
        self.infer_flags = [1]
        self.out = self.input[:, 0:1, :]


class TestSliceOpFp16(TestSliceOp):
    def init_dtype(self):
        self.dtype = np.float16

    def set_npu(self):
        self.__class__.use_npu = True
        self.__class__.no_need_check_grad = True
        self.place = paddle.NPUPlace(0)


class TestSliceOpTensor(TestSliceOp):
    def setUp(self):
        self.op_type = "slice"
        self.set_npu()
        self.init_dtype()
        self.config()
        self.inputs = {
            'Input': self.input,
            'StartsTensor': self.starts,
            'EndsTensor': self.ends
        }
        self.outputs = {'Out': self.out}
        self.attrs = {
            'axes': self.axes,
            'starts': [-1, -1, -1],
            'ends': [-1, -1, -1],
            'infer_flags': self.infer_flags
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype(self.dtype)
        self.starts = np.array([1, 0, 2]).astype('int32')
        self.ends = np.array([3, 3, 4]).astype('int32')
        self.axes = [0, 1, 2]
        self.infer_flags = [-1, -1, -1]
        self.out = self.input[1:3, 0:3, 2:4, :]


class TestSliceOpTensor2(TestSliceOpTensor):
    def setUp(self):
        self.op_type = "slice"
        self.set_npu()
        self.init_dtype()
        self.config()
        self.inputs = {
            'Input': self.input,
            'StartsTensor': self.starts,
            'EndsTensor': self.ends
        }
        self.outputs = {'Out': self.out}
        self.attrs = {
            'axes': self.axes,
            'starts': [-1],
            'ends': [-1],
            'infer_flags': self.infer_flags
        }

    def config(self):
        self.input = np.random.random([10, 5, 6]).astype(self.dtype)
        self.starts = np.array([0]).astype('int32')
        self.ends = np.array([1]).astype('int32')
        self.axes = [1]
        self.infer_flags = [-1]
        self.out = self.input[:, 0:1, :]


class TestSliceOpFp16Tensor(TestSliceOpTensor):
    def init_dtype(self):
        self.dtype = np.float16

    def set_npu(self):
        self.__class__.use_npu = True
        self.__class__.no_need_check_grad = True
        self.place = paddle.NPUPlace(0)


class TestSliceOpTensorList(TestSliceOp):
    def setUp(self):
        self.op_type = "slice"
        self.set_npu()
        self.init_dtype()
        self.config()

        self.starts_tensor_list = []
        for index, ele in enumerate(self.starts):
            self.starts_tensor_list.append(("start" + str(index), np.ones(
                (1)).astype('int32') * ele))

        self.ends_tensor_list = []
        for index, ele in enumerate(self.ends):
            self.ends_tensor_list.append(("end" + str(index), np.ones(
                (1)).astype('int32') * ele))

        self.inputs = {
            'Input': self.input,
            'StartsTensorList': self.starts_tensor_list,
            'EndsTensorList': self.ends_tensor_list
        }
        self.outputs = {'Out': self.out}
        self.attrs = {
            'axes': self.axes,
            'starts': [-1, -1, -1],
            'ends': [-1, -1, -1],
            'infer_flags': self.infer_flags
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype(self.dtype)
        self.starts = [1, 0, 2]
        self.ends = [3, 3, 4]
        self.axes = [0, 1, 2]
        self.infer_flags = [-1, -1, -1]
        self.out = self.input[1:3, 0:3, 2:4, :]


class TestSliceOpTensorList2(TestSliceOpTensorList):
    def setUp(self):
        self.op_type = "slice"
        self.set_npu()
        self.init_dtype()
        self.config()

        self.starts_tensor_list = []
        for index, ele in enumerate(self.starts):
            self.starts_tensor_list.append(("start" + str(index), np.ones(
                (1)).astype('int32') * ele))

        self.ends_tensor_list = []
        for index, ele in enumerate(self.ends):
            self.ends_tensor_list.append(("end" + str(index), np.ones(
                (1)).astype('int32') * ele))

        self.inputs = {
            'Input': self.input,
            'StartsTensorList': self.starts_tensor_list,
            'EndsTensorList': self.ends_tensor_list
        }
        self.outputs = {'Out': self.out}
        self.attrs = {
            'axes': self.axes,
            'starts': [-1],
            'ends': [-1],
            'infer_flags': self.infer_flags
        }

    def config(self):
        self.input = np.random.random([10, 5, 6]).astype(self.dtype)
        self.starts = np.array([0]).astype('int32')
        self.ends = np.array([1]).astype('int32')
        self.axes = [1]
        self.infer_flags = [-1]
        self.out = self.input[:, 0:1, :]


class TestSliceOpFp16TensorList(TestSliceOpTensorList):
    def init_dtype(self):
        self.dtype = np.float16

    def set_npu(self):
        self.__class__.use_npu = True
        self.__class__.no_need_check_grad = True
        self.place = paddle.NPUPlace(0)


class TestSliceNet(unittest.TestCase):
    def _test(self, run_npu=True):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        np.random.seed(SEED)

        batch_size = 32
        data_shape = (32, 32)
        a_np = np.random.random(size=data_shape).astype('float32')
        b_np = np.random.random(size=data_shape).astype('float32')
        label_np = np.random.randint(2, size=(batch_size, 1)).astype('int64')

        with paddle.static.program_guard(main_prog, startup_prog):
            a = paddle.static.data(name="a", shape=data_shape, dtype='float32')
            b = paddle.static.data(name="b", shape=data_shape, dtype='float32')
            label = paddle.static.data(
                name="label", shape=[batch_size, 1], dtype='int64')

            sum = paddle.add(a, b)
            z = paddle.slice(sum, axes=[0, 1], starts=[0, 0], ends=[33, 2])

            prediction = paddle.static.nn.fc(z, size=2, activation='softmax')

            cost = paddle.fluid.layers.softmax_with_cross_entropy(
                logits=prediction, label=label)
            loss = paddle.mean(cost)
            sgd = paddle.optimizer.SGD(learning_rate=0.01)
            sgd.minimize(loss)

        if run_npu:
            place = paddle.NPUPlace(0)
        else:
            place = paddle.CPUPlace()

        exe = paddle.static.Executor(place)
        exe.run(startup_prog)
        print("Start run on {}".format(place))
        for epoch in range(EPOCH):

            pred_res, loss_res = exe.run(
                main_prog,
                feed={"a": a_np,
                      "b": b_np,
                      "label": label_np},
                fetch_list=[prediction, loss])
            if epoch % 10 == 0:
                print("Epoch {} | Prediction[0]: {}, Loss: {}".format(
                    epoch, pred_res[0], loss_res))

        return pred_res, loss_res

    def test_npu(self):
        cpu_pred, cpu_loss = self._test(False)
        npu_pred, npu_loss = self._test(True)

        self.assertTrue(np.allclose(npu_pred, cpu_pred))
        self.assertTrue(np.allclose(npu_loss, cpu_loss))


class TestSliceOpDecsDim(OpTest):
    def setUp(self):
        self.op_type = "slice"
        self.set_npu()
        self.init_dtype()
        self.config()
        self.set_inputs()
        self.set_outputs()
        self.set_attrs()

    def set_inputs(self):
        self.inputs = {'Input': self.input}

    def set_outputs(self):
        self.outputs = {'Out': self.out}

    def set_attrs(self):
        self.attrs = {
            'axes': self.axes,
            'starts': self.starts,
            'ends': self.ends,
            'infer_flags': self.infer_flags,
            'decrease_axis': self.decrease_axis,
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype(self.dtype)
        self.starts = [1, 0, 2]
        self.ends = [2, 3, 4]
        self.axes = [0, 1, 2]
        self.decrease_axis = [0]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[1, 0:3, 2:4, :]

    def init_dtype(self):
        self.dtype = np.float32

    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(0)

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        if self.dtype == np.float16:
            self.check_grad_with_place(
                self.place, ['Input'], 'Out', max_relative_error=0.5)
        else:
            self.check_grad_with_place(self.place, ['Input'], 'Out')


class TestSliceOpDecsDimFp16(TestSliceOpDecsDim):
    def init_dtype(self):
        self.dtype = np.float16


class TestSliceOpDecsDim2(TestSliceOpDecsDim):
    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype(self.dtype)
        self.starts = [1, 0, 2]
        self.ends = [2, 1, 4]
        self.axes = [0, 1, 2]
        self.decrease_axis = [0, 1]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[1, 0, 2:4, :]


class TestSliceOpDecsDim3(TestSliceOpDecsDim):
    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype(self.dtype)
        self.starts = [-1, 0, 2]
        self.ends = [1000000, 1, 4]
        self.axes = [0, 1, 2]
        self.decrease_axis = [0, 1]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[-1, 0, 2:4, :]


class TestSliceOpDecsDim4(TestSliceOpDecsDim):
    def config(self):
        self.input = np.random.random([3, 4, 5, 7]).astype(self.dtype)
        self.starts = [0, 1, 2, 3]
        self.ends = [1, 2, 3, 4]
        self.axes = [0, 1, 2, 3]
        self.decrease_axis = [0, 1, 2, 3]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[0, 1, 2, 3:4]


class TestSliceOpDecsDim5(TestSliceOpDecsDim):
    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype(self.dtype)
        self.starts = [-1]
        self.ends = [1000000]
        self.axes = [3]
        self.decrease_axis = [3]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[:, :, :, -1]


class TestSliceOpDecsDim6(TestSliceOpDecsDim):
    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype(self.dtype)
        self.starts = [0, 1, 2, 3]
        self.ends = [1, 2, 3, 4]
        self.axes = [0, 1, 2, 3]
        self.decrease_axis = [0, 1, 2, 3]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[0, 1, 2, 3:4]


class TestSliceOpDecsDimStartsTensor(TestSliceOpDecsDim):
    def set_inputs(self):
        self.inputs = {
            'Input': self.input,
            "StartsTensor": np.array(
                self.starts, dtype='int32')
        }

    def set_attrs(self):
        self.attrs = {
            'axes': self.axes,
            #'starts': self.starts,
            'ends': self.ends,
            'infer_flags': self.infer_flags,
            'decrease_axis': self.decrease_axis,
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype(self.dtype)
        self.starts = [1, 0, 2]
        self.ends = [2, 3, 4]
        self.axes = [0, 1, 2]
        self.decrease_axis = [0]
        self.infer_flags = [-1, -1, -1]
        self.out = self.input[1, 0:3, 2:4, :]


class TestSliceOpDecsDimStartsTensorFP16(TestSliceOpDecsDimStartsTensor):
    def init_dtype(self):
        self.dtype = np.float16


class TestSliceOpDecsDimStartsTensorStartsAndEndsTensor(TestSliceOpDecsDim):
    def set_inputs(self):
        self.inputs = {
            'Input': self.input,
            "StartsTensor": np.array(
                self.starts, dtype='int64'),
            "EndsTensor": np.array(
                self.ends, dtype='int32')
        }

    def set_attrs(self):
        self.attrs = {
            'axes': self.axes,
            #'starts': self.starts,
            #'ends': self.ends,
            'infer_flags': self.infer_flags,
            'decrease_axis': self.decrease_axis,
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype(self.dtype)
        self.starts = [1, 0, 2]
        self.ends = [2, 1, 4]
        self.axes = [0, 1, 2]
        self.decrease_axis = [0, 1]
        self.infer_flags = [-1, -1, -1]
        self.out = self.input[1, 0, 2:4, :]


class TestSliceOpDecsDimStartsTensorStartsAndEndsTensorFP16(
        TestSliceOpDecsDimStartsTensorStartsAndEndsTensor):
    def init_dtype(self):
        self.dtype = np.float16


class TestSliceOpDecsDimStartsListTensor(TestSliceOpDecsDim):
    def set_inputs(self):
        starts_tensor = []
        for index, ele in enumerate(self.starts):
            starts_tensor.append(("x" + str(index), np.ones(
                (1)).astype('int32') * ele))

        self.inputs = {'Input': self.input, 'StartsTensorList': starts_tensor}

    def set_attrs(self):
        self.attrs = {
            'axes': self.axes,
            'starts': self.starts_infer,
            'ends': self.ends,
            'infer_flags': self.infer_flags,
            'decrease_axis': self.decrease_axis,
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype(self.dtype)
        self.starts = [1, 0, 2]
        self.ends = [2, 3, 4]
        self.axes = [0, 1, 2]
        self.decrease_axis = [0]
        self.infer_flags = [1, -1, 1]
        self.out = self.input[1, 0:3, 2:4, :]

        self.starts_infer = [1, -1, 2]


class TestSliceOpDecsDimStartsListTensor2(TestSliceOpDecsDimStartsListTensor):
    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype(self.dtype)
        self.starts = [-1]
        self.ends = [1000000]
        self.axes = [3]
        self.decrease_axis = [3]
        self.infer_flags = [-1]
        self.out = self.input[:, :, :, -1]

        self.starts_infer = [-1]


class TestSliceOpDecsDimStartsListTensorFP16(
        TestSliceOpDecsDimStartsListTensor):
    def init_dtype(self):
        self.dtype = np.float16


class TestSliceOpInt64(OpTest):
    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(0)

    def setUp(self):
        self.op_type = "slice"
        self.set_npu()
        self.init_dtype()
        self.config()
        self.inputs = {'Input': self.input}
        self.outputs = {'Out': self.out}
        self.attrs = {
            'axes': self.axes,
            'starts': self.starts,
            'ends': self.ends,
            'infer_flags': self.infer_flags
        }

    def config(self):
        self.input = np.random.randint(
            100, size=(3, 4, 5, 6)).astype(self.dtype)
        self.starts = [1, 0, 2]
        self.ends = [3, 3, 4]
        self.axes = [0, 1, 2]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[1:3, 0:3, 2:4, :]

    def init_dtype(self):
        self.dtype = np.int64

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestSliceOpTensorInt64(TestSliceOpInt64):
    def setUp(self):
        self.op_type = "slice"
        self.set_npu()
        self.init_dtype()
        self.config()
        self.inputs = {
            'Input': self.input,
            'StartsTensor': self.starts,
            'EndsTensor': self.ends
        }
        self.outputs = {'Out': self.out}
        self.attrs = {
            'axes': self.axes,
            'starts': [-1, -1, -1],
            'ends': [-1, -1, -1],
            'infer_flags': self.infer_flags
        }

    def config(self):
        self.input = np.random.randint(
            100, size=(3, 4, 5, 6)).astype(self.dtype)
        self.starts = np.array([1, 0, 2]).astype('int32')
        self.ends = np.array([3, 3, 4]).astype('int32')
        self.axes = [0, 1, 2]
        self.infer_flags = [-1, -1, -1]
        self.out = self.input[1:3, 0:3, 2:4, :]


if __name__ == '__main__':
    unittest.main()
