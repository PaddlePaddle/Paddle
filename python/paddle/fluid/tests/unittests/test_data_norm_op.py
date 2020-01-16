#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""This is unit test of Test data_norm Op."""

from __future__ import print_function

import unittest
import numpy as np
import paddle.fluid.core as core
from paddle.fluid.op import Operator
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import os
from op_test import OpTest
from paddle.fluid.framework import grad_var_name


def _reference_testing(x, batch_size, batch_sum, batch_square_sum, slot_dim=-1):
    x_shape = x.shape
    means_arr = batch_sum / batch_size
    scales_arr = np.sqrt(batch_size / batch_square_sum)
    min_precision = 1e-7
    if slot_dim <= 0:
        for i in range(x_shape[0]):
            x[i] -= means_arr
            x[i] *= scales_arr
        y = np.array(x)
    else:
        y = np.zeros(x_shape).astype(np.float32)
        for i in range(x_shape[0]):
            for j in range(0, x_shape[1], slot_dim):
                if x[i][j] <= -min_precision or x[i][j] >= min_precision:
                    for k in range(0, slot_dim):
                        y[i][j + k] = (
                            x[i][j + k] - means_arr[j + k]) * scales_arr[j + k]
    return y


def create_or_get_tensor(scope, var_name, var, place):
    tensor = scope.var(var_name).get_tensor()
    if var is not None:
        assert isinstance(var, np.ndarray)
        tensor.set_recursive_sequence_lengths([])
        tensor.set(var, place)
    return tensor


class TestDataNormOpInference(unittest.TestCase):
    """
    test class for data norm op
    test forward
    """

    def setUp(self):
        """
        init members of this class
        """
        self.dtype = np.float32
        self.use_mkldnn = False

    def __assert_close(self, tensor, np_array, msg, atol=1e-4):
        self.assertTrue(np.allclose(np.array(tensor), np_array, atol=atol), msg)

    def check_with_place(self, place, data_layout, dtype, shape, slot_dim=-1):
        """
        do forward and check

        Args:
            place(Place): CPUPlace
            data_layout(str): NCHW or NWHC
            dtype(dtype): np.float32
            shape(list): input shape
            slot_dim(int): dimension of one slot. Refer to data_norm api.


        """
        epsilon = 0.00001
        if len(shape) == 2:
            x_shape = shape
            c = x_shape[1]
        else:
            ValueError("len(shape) should be equal to 2")
        scale_shape = [c]

        x_val = np.random.random_sample(x_shape).astype(dtype)
        x_val = x_val - 0.5
        x_val[0][1] = 0.0
        x_val[1][1] = 0.0
        batch_size = np.ones(scale_shape).astype(np.float32)
        batch_size *= 1e4
        batch_sum = np.zeros(scale_shape).astype(np.float32)
        batch_square_sum = np.ones(scale_shape).astype(np.float32)
        batch_square_sum *= 1e4

        y_out = _reference_testing(x_val, batch_size, batch_sum,
                                   batch_square_sum, slot_dim).astype(dtype)

        scope = core.Scope()

        # create input
        x_tensor = create_or_get_tensor(scope, "x_val",
                                        OpTest.np_dtype_to_fluid_dtype(x_val),
                                        place)
        batch_size_tensor = create_or_get_tensor(
            scope, "batch_size",
            OpTest.np_dtype_to_fluid_dtype(batch_size), place)
        batch_sum_tensor = create_or_get_tensor(
            scope, "batch_sum",
            OpTest.np_dtype_to_fluid_dtype(batch_sum), place)
        batch_square_sum_tensor = create_or_get_tensor(
            scope, "batch_square_sum",
            OpTest.np_dtype_to_fluid_dtype(batch_square_sum), place)

        # create output
        y_tensor = create_or_get_tensor(scope, "y_out", None, place)
        mean_tensor = create_or_get_tensor(scope, "mean", None, place)
        scales_tensor = create_or_get_tensor(scope, "scales", None, place)

        data_norm_op = Operator(
            "data_norm",
            # inputs
            X="x_val",
            BatchSize="batch_size",
            BatchSum="batch_sum",
            BatchSquareSum="batch_square_sum",
            # outputs
            Y="y_out",
            Means="mean",
            Scales="scales",
            # attrs
            epsilon=epsilon,
            use_mkldnn=self.use_mkldnn,
            slot_dim=slot_dim)

        data_norm_op.run(scope, place)

        # check inference result
        self.__assert_close(
            y_tensor,
            y_out,
            "inference output are different at " + str(place) + ", " +
            data_layout + ", " + str(np.dtype(dtype)) +
            str(np.array(y_tensor)) + str(y_out),
            atol=1e-3)

    def test_check_output(self):
        """
        test check forward, check output
        """
        places = [core.CPUPlace()]
        for place in places:
            for data_format in ["NCHW", "NHWC"]:
                for slot_dim in [-1, 1]:
                    self.check_with_place(
                        place,
                        data_format,
                        self.dtype, [2, 3],
                        slot_dim=slot_dim)


class TestDataNormOp(OpTest):
    """
    test class for data norm op
    test forward and backward
    """

    def setUp(self):
        """
        init data norm op test env
        """
        self.op_type = 'data_norm'
        self.use_mkldnn = False
        epsilon = 0.00001
        x_shape = [10, 12]
        scale_shape = [12]
        tp = np.float32

        x_val = np.random.random(x_shape).astype(tp)
        batch_size = np.ones(scale_shape).astype(tp)
        batch_size *= 1e4
        batch_sum = np.zeros(scale_shape).astype(tp)
        batch_square_sum = np.ones(scale_shape).astype(tp)
        batch_square_sum *= 1e4

        y = np.array(x_val)

        mean = np.zeros(x_shape).astype(tp)
        scale = np.ones(x_shape).astype(tp)

        self.inputs = {
            "X": x_val,
            "BatchSize": batch_size,
            "BatchSum": batch_sum,
            "BatchSquareSum": batch_square_sum
        }
        self.outputs = {"Y": y, "Means": mean, "Scales": scale}
        self.attrs = {"epsilon": epsilon, "use_mkldnn": self.use_mkldnn}

    def test_check_output(self):
        """
        test check forward, check output
        """
        self.check_output()

    def test_check_grad(self):
        """
        test check backward, check grad
        """
        self.check_grad(['X'], 'Y', no_grad_set=set([]), check_dygraph=False)


class TestDataNormOpWithSlotDim(OpTest):
    """
    test class for data norm op
    test forward and backward
    """

    def setUp(self):
        """
        init data norm op test env
        """
        self.op_type = 'data_norm'
        self.use_mkldnn = False
        epsilon = 0.00001
        slot_dim = 1
        x_shape = [2, 50]
        scale_shape = [50]
        tp = np.float32

        x_val = np.random.uniform(-1, 1, x_shape).astype(tp)
        batch_size = np.ones(scale_shape).astype(tp)
        batch_size *= 1e4
        batch_sum = np.zeros(scale_shape).astype(tp)
        batch_square_sum = np.ones(scale_shape).astype(tp)
        batch_square_sum *= 1e4

        y = np.array(x_val)

        mean = np.zeros(x_shape).astype(tp)
        scale = np.ones(x_shape).astype(tp)

        self.inputs = {
            "X": x_val,
            "BatchSize": batch_size,
            "BatchSum": batch_sum,
            "BatchSquareSum": batch_square_sum
        }
        self.outputs = {"Y": y, "Means": mean, "Scales": scale}
        self.attrs = {
            "epsilon": epsilon,
            "use_mkldnn": self.use_mkldnn,
            "slot_dim": slot_dim
        }

    def test_check_output(self):
        """
        test check forward, check output
        """
        self.check_output()

    def test_check_grad(self):
        """
        test check backward, check grad
        """
        self.check_grad(['X'], 'Y', no_grad_set=set([]), check_dygraph=False)


class TestDataNormOpWithSyncStats(unittest.TestCase):
    """
    test class for data norm op
    test forward and backward
    """

    def test_sync_stats(self):
        if not core.is_compiled_with_cuda():
            return
        if os.name == 'nt':
            print(
                'Skip TestDataNormOpWithSyncStats because nccl is not supported on windows'
            )
            return
        x = fluid.layers.data(name='x', shape=[1], dtype='int64', lod_level=0)
        emb = layers.embedding(
            input=x,
            param_attr=fluid.ParamAttr(name="embx"),
            size=[10, 2],
            is_sparse=False)

        dn = layers.data_norm(
            input=emb,
            name="hehe",
            epsilon=1e-4,
            param_attr={
                "batch_size": 1e4,
                "batch_sum": 1e5,
                "batch_square": 1e4
            },
            summary_decay_rate=1,
            sync_stats=True)  #[-1,3]
        loss = layers.mean(dn)

        optimizer = fluid.optimizer.SGD(learning_rate=0.5)
        optimizer = fluid.optimizer.PipelineOptimizer(
            optimizer,
            cut_list=[[emb], [loss]],
            place_list=[
                fluid.CUDAPlace(0), fluid.CUDAPlace(0), fluid.CPUPlace()
            ],
            concurrency_list=[1, 1, 1],
            queue_size=1,
            sync_steps=10000000, )

        all_p = fluid.default_main_program().global_block().all_parameters()
        parameter_without_datanorm = []
        for e in all_p:
            if e.name.find("batch_size") != -1 or e.name.find(
                    "batch_sq") != -1 or e.name.find("batch_sum") != -1:
                continue
            parameter_without_datanorm.append(e.name)
        optimizer.minimize(loss, parameter_list=parameter_without_datanorm)
        place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        #prepare data
        batch_size = 1

        def binary_print(slot, fout):
            num = np.int16(len(slot) + 1)
            num.tofile(fout)
            a = np.int64(batch_size)
            a.tofile(fout)
            slot.tofile(fout)

        #batch1 = np.array([[0,1], [1,2], [2,3]]).astype("int64").reshape(batch_size,2,1)
        #batch2 = np.array([[1,2], [2,3], [3,4]]).astype("int64").reshape(batch_size,2,1)
        batch1 = np.ones(
            (batch_size, 1)).astype("int64").reshape(batch_size, 1, 1)
        batch2 = np.ones(
            (batch_size, 1)).astype("int64").reshape(batch_size, 1, 1)
        data = [batch1, batch2]
        data = [batch1]
        filelist = []
        for i in range(2):
            filelist.append("test_pipeline_input_" + str(i))
        for f in filelist:
            with open(f, "wb") as fout:
                for batch_data in data:
                    for ins in batch_data:
                        for slot in ins:
                            binary_print(slot, fout)

        dataset = fluid.DatasetFactory().create_dataset("FileInstantDataset")
        dataset.set_use_var([x])
        dataset.set_batch_size(batch_size)
        dataset.set_filelist(filelist)

        block = fluid.default_startup_program().global_block()
        block.append_op(
            type='c_comm_init_all', attrs={'ring_id': 0,
                                           'devices': [0, 1]})
        with open("main_program", "w") as fout:
            fout.write(str(fluid.default_main_program()))
        with open("startup_program", "w") as fout:
            fout.write(str(fluid.default_startup_program()))
        exe.run(fluid.default_startup_program())
        emb_t = fluid.global_scope().find_var("embx").get_tensor()
        para = np.ones((10, 2)).astype("float32")
        emb_t.set(para, place)
        for epoch in range(1):
            exe.train_from_dataset(
                fluid.default_main_program(),
                dataset,
                thread=2,
                debug=False,
                fetch_list=[],
                fetch_info=[],
                print_period=1)
        batch_size = np.array(fluid.global_scope().find_var("hehe.batch_size")
                              .get_tensor())
        self.assertEqual(batch_size[0], 10002)
        b = np.array(fluid.global_scope().find_var("hehe.batch_sum").get_tensor(
        ))
        self.assertEqual(b[0], 100002)
        c = np.array(fluid.global_scope().find_var("hehe.batch_square_sum")
                     .get_tensor())
        self.assertEqual(c[0], 10162)

        for f in filelist:
            os.remove(f)


if __name__ == '__main__':
    unittest.main()
