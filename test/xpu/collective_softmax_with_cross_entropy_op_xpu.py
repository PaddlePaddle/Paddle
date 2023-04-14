# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os
import pickle
import sys

import numpy as np
from test_collective_base_xpu import (
    DataTypeCast,
    TestCollectiveRunnerBase,
    runtime_main,
)

import paddle
from paddle.framework import core
from paddle.static import Executor, Program, data, program_guard

paddle.enable_static()


class TestCollectiveSoftmaxWithCE(TestCollectiveRunnerBase):
    def __init__(self):
        self.global_ring_id = 0
        self.batch_size = 10
        self.num_class = 1000
        self.nranks = 2
        self.ring_id = 0
        self.local_elements = int(self.num_class / self.nranks)

    def get_model(self, main_prog, startup_program, rank):
        with program_guard(main_prog, startup_program):
            logits = data(
                name="Logits",
                shape=[self.batch_size, self.local_elements],
                dtype='float32',
            )
            label = data(
                name="Label", shape=[self.batch_size, 1], dtype='int32'
            )
            softmax = main_prog.current_block().create_var(
                name="Softmax",
                dtype=logits.dtype,
                type=core.VarDesc.VarType.LOD_TENSOR,
                persistable=False,
                stop_gradient=False,
            )
            loss = main_prog.current_block().create_var(
                name="Loss",
                dtype=logits.dtype,
                type=core.VarDesc.VarType.LOD_TENSOR,
                persistable=False,
                stop_gradient=False,
            )
            loss_grad = main_prog.current_block().create_var(
                name="Loss@GRAD",
                shape=[self.batch_size, 1],
                dtype=logits.dtype,
                type=core.VarDesc.VarType.LOD_TENSOR,
                persistable=False,
                stop_gradient=False,
            )
            block = main_prog.global_block()
            with paddle.static.device_guard("xpu"):
                c_softmax_with_ce_op = block.append_op(
                    type="c_softmax_with_cross_entropy",
                    inputs={'Logits': logits, 'Label': label},
                    outputs={'Softmax': softmax, 'Loss': loss},
                    attrs={
                        'ring_id': self.ring_id,
                        'rank': rank,
                        'nranks': self.nranks,
                    },
                )
                # generate backward op_desc
                grad_op_desc_list, op_grad_to_var = core.get_grad_op_desc(
                    c_softmax_with_ce_op.desc, set(), []
                )
                for grad_op_desc in grad_op_desc_list:
                    new_op_desc = block.desc.append_op()
                    new_op_desc.copy_from(grad_op_desc)
                    for var_name in grad_op_desc.output_arg_names():
                        block.desc.var(var_name.encode("ascii"))
                    grad_op_desc.infer_var_type(block.desc)
                    grad_op_desc.infer_shape(block.desc)
                    for arg in grad_op_desc.output_arg_names():
                        grad_var = block.desc.find_var(arg.encode("ascii"))
                        grad_var.set_dtype(core.VarDesc.VarType.FP32)
                    main_prog._sync_with_cpp()

            return loss, softmax

    def run_trainer(self, args):
        train_prog = Program()
        startup_prog = Program()
        endpoints = args["endpoints"].split(",")
        rank = args["trainerid"]
        current_endpoint = args["currentendpoint"]
        self.initCommunicator(
            startup_prog, rank, self.nranks, True, current_endpoint, endpoints
        )
        np_data_type = DataTypeCast(args["data_type"])
        loss, softmax = self.get_model(train_prog, startup_prog, rank)
        device_id = int(os.getenv("FLAGS_selected_xpus", "0"))
        place = paddle.XPUPlace(device_id)
        exe = Executor(place)
        exe.run(startup_prog)

        # NOTE use uid here to assure that two xpus share the same label
        np.random.seed(os.getuid())
        label = np.random.randint(
            0,
            self.num_class,
            size=(self.batch_size, 1),
            dtype='int32',
        )
        # use FAKE loss_grad here, only to examine the correctness of grad func
        loss_grad = np.random.uniform(
            low=-10.0, high=10.0, size=(self.batch_size, 1)
        ).astype(np_data_type)

        # each xpu uses own half of logits
        np.random.seed(os.getpid())
        logits = np.random.uniform(
            low=-10.0, high=10.0, size=(self.batch_size, self.local_elements)
        ).astype(np_data_type)
        out = exe.run(
            train_prog,
            feed={'Logits': logits, 'Label': label, 'Loss@GRAD': loss_grad},
            fetch_list=[loss.name, softmax.name, 'Logits@GRAD'],
        )
        sys.stdout.buffer.write(pickle.dumps(out))


if __name__ == "__main__":
    os.environ["BKCL_PCIE_RING"] = "1"
    runtime_main(TestCollectiveSoftmaxWithCE, "softmax_with_ce", 0)
