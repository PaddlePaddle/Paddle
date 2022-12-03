# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import numpy as np
from dist_pass_test_base import DistPassTestBase

import paddle
import paddle.distributed.fleet as fleet
import paddle.nn as nn
from paddle.distributed.passes import PassManager, new_pass

paddle.enable_static()
np.random.seed(12345)
paddle.seed(12345)


def verify_op_count(op_types, op_name, target_count):
    count = 0
    for op_type in op_types:
        if op_type == op_name:
            count += 1
    return count == target_count


class MultiFCLayer(nn.Layer):
    def __init__(self, hidden, Activation):
        super(MultiFCLayer, self).__init__()
        self.linear1 = paddle.nn.Linear(hidden, 4 * hidden)
        self.linear2 = paddle.nn.Linear(4 * hidden, hidden)
        self.linear3 = paddle.nn.Linear(hidden, hidden)

        self.relu1 = Activation()
        self.relu2 = Activation()
        self.relu3 = Activation()

    def forward(self, x, matmul_y, ele_y):
        output = self.linear1(x)
        output = self.relu1(output)
        output = self.linear2(output)

        output1 = paddle.matmul(output, matmul_y)
        output = self.linear3(output)
        output = self.relu2(output)

        output = paddle.matmul(output, matmul_y)
        output = paddle.add(output, ele_y)
        output = self.relu3(output)
        output = paddle.add(output, output1)
        return output


class TestFuseGemmEpiloguePassReluFP32(DistPassTestBase):
    def init(self):
        self.atol = 1e-3
        self.rtol = 1e-3
        self.activation = nn.ReLU
        self.act_fwd_name = 'relu'
        self.act_bwd_name = 'relu_grad'
        self.batch = 64
        self.seqlen = 128
        self.hidden = 768
        self.precision = 'FP32'  # FP32 or AMP

    def get_model(self, place):
        data = paddle.static.data(
            name="_data", shape=[-1, self.seqlen, self.hidden], dtype='float32'
        )
        matmul_y = paddle.static.data(
            name="_matmul_y",
            shape=[1, self.hidden, self.hidden],
            dtype='float32',
        )
        ele_y = paddle.static.data(
            name="_ele_y",
            shape=[
                self.hidden,
            ],
            dtype='float32',
        )

        model = MultiFCLayer(self.hidden, self.activation)
        out = model(data, matmul_y, ele_y)
        loss = paddle.mean(out)
        optimizer = paddle.optimizer.Adam(learning_rate=1e-3)

        dist_strategy = fleet.DistributedStrategy()
        dist_strategy.fuse_all_reduce_ops = False
        dist_strategy.without_graph_optimization = True
        if self.precision == 'AMP':
            dist_strategy.amp = True
            dist_strategy.amp_configs = {
                "init_loss_scaling": 32768,
                "use_dynamic_loss_scaling": True,
                "custom_white_list": ['gelu'],
            }
        fleet.init(is_collective=True, strategy=dist_strategy)
        optimizer = fleet.distributed_optimizer(optimizer)
        optimizer.minimize(loss)

        rank = paddle.distributed.get_rank()

        def reader():
            for _ in range(10):
                data_arr = (
                    np.random.random(
                        (self.batch, self.seqlen, self.hidden)
                    ).astype("float32")
                    - 0.5
                )
                matmul_y_arr = (
                    np.random.random((1, self.hidden, self.hidden)).astype(
                        "float32"
                    )
                    - 0.5
                )
                ele_y_arr = (
                    np.random.random((self.hidden,)).astype("float32") - 0.5
                )
                yield [data_arr, matmul_y_arr, ele_y_arr]

        main_program = paddle.static.default_main_program()
        startup_program = paddle.static.default_startup_program()

        fetch_list = []
        for p in model.parameters():
            grad_name = p.name + '@GRAD'
            fetch_list.append(grad_name)

        fetch_list.append(loss.name)

        return (
            main_program,
            startup_program,
            [data, matmul_y, ele_y],
            fetch_list,
            reader,
        )

    def apply_passes(self, main_prog, startup_prog):
        pass_manager = PassManager([new_pass("fuse_gemm_epilogue")])
        pass_manager.apply([main_prog], [startup_prog])
        print(pass_manager.names)

        op_type = []
        for op in main_prog.global_block().ops:
            op_type.append(op.type)
        print(op_type)
        self.assertTrue(verify_op_count(op_type, "fused_gemm_epilogue", 3))
        self.assertTrue(verify_op_count(op_type, "fused_gemm_epilogue_grad", 3))
        self.assertTrue(verify_op_count(op_type, self.act_fwd_name, 1))
        self.assertTrue(verify_op_count(op_type, self.act_bwd_name, 2))

    def test_fuse_gemm_epilogue(self):
        self.check_main()


class TestFuseGemmEpiloguePassReluFP16(TestFuseGemmEpiloguePassReluFP32):
    def init(self):
        self.atol = 1e-3
        self.rtol = 1e-3
        self.activation = nn.ReLU
        self.act_fwd_name = 'relu'
        self.act_bwd_name = 'relu_grad'
        self.batch = 64
        self.seqlen = 128
        self.hidden = 768
        self.precision = 'AMP'  # FP32 or AMP


class TestFuseGemmEpiloguePassGeluFP32(TestFuseGemmEpiloguePassReluFP32):
    def init(self):
        self.atol = 1e-3
        self.rtol = 1e-3
        self.activation = nn.GELU
        self.act_fwd_name = 'gelu'
        self.act_bwd_name = 'gelu_grad'
        self.batch = 64
        self.seqlen = 128
        self.hidden = 768
        self.precision = 'FP32'  # FP32 or AMP


class TestFuseGemmEpiloguePassGeluFP16(TestFuseGemmEpiloguePassReluFP32):
    def init(self):
        self.atol = 5e-3
        self.rtol = 1e-3
        self.activation = nn.GELU
        self.act_fwd_name = 'gelu'
        self.act_bwd_name = 'gelu_grad'
        self.batch = 64
        self.seqlen = 128
        self.hidden = 768
        self.precision = 'AMP'  # FP32 or AMP


if __name__ == "__main__":
    unittest.main()
