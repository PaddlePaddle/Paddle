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
import sys

import numpy as np

sys.path.append("../legacy_test")
from op_test import convert_float_to_uint16
from test_collective_base_xpu import (
    DataTypeCast,
    TestCollectiveRunnerBase,
    dump_output,
    runtime_main,
)

import paddle
from paddle.framework import core
from paddle.static import Executor, Program, data, program_guard

paddle.enable_static()


class TestCollectiveSoftmaxWithCE(TestCollectiveRunnerBase):
    def __init__(self):
        self.global_ring_id = 0
        self.batch_size = 1
        self.seq_len = 10
        self.num_class = 1000
        self.nranks = 2
        self.ring_id = 0
        self.local_elements = int(self.num_class / self.nranks)

        self.logits_shape = [self.seq_len, self.local_elements]
        self.label_shape = [self.seq_len, 1]

    def get_model(self, main_prog, startup_program, rank, ignore_index):
        with program_guard(main_prog, startup_program):
            logits = data(
                name="Logits",
                shape=self.logits_shape,
                dtype=self.dtype,
            )
            label = data(name="Label", shape=self.label_shape, dtype='int32')
            softmax = main_prog.current_block().create_var(
                name="Softmax",
                dtype=logits.dtype,
                type=core.VarDesc.VarType.DENSE_TENSOR,
                persistable=False,
                stop_gradient=False,
            )
            loss = main_prog.current_block().create_var(
                name="Loss",
                dtype=logits.dtype,
                type=core.VarDesc.VarType.DENSE_TENSOR,
                persistable=False,
                stop_gradient=False,
            )
            loss_grad = main_prog.current_block().create_var(
                name="Loss@GRAD",
                shape=self.label_shape,
                dtype=logits.dtype,
                type=core.VarDesc.VarType.DENSE_TENSOR,
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
                        'ignore_index': ignore_index,
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
        self.dtype = args["dtype"]

        # if batch_size = 1, we treat logits/labels as 2D tensors
        # if batch_size > 1, we treat logits/labels as 3D tensors
        if self.batch_size is not None:
            self.batch_size = int(args["batch_size"])
        if self.batch_size > 1:
            self.logits_shape = [
                self.batch_size,
                self.seq_len,
                self.local_elements,
            ]
            self.label_shape = [self.batch_size, self.seq_len, 1]

        # NOTE use uid here to assure that two xpus share the same label
        np.random.seed(os.getuid())
        label = np.random.randint(
            0,
            self.num_class,
            size=self.label_shape,
            dtype='int32',
        )
        ignore_index = label[0][0]

        np_dtype = DataTypeCast(args["dtype"])
        loss, softmax = self.get_model(
            train_prog, startup_prog, rank, ignore_index
        )
        device_id = int(os.getenv("FLAGS_selected_xpus", "0"))
        place = paddle.XPUPlace(device_id)
        exe = Executor(place)
        exe.run(startup_prog)

        # use FAKE loss_grad here, only to examine the correctness of grad func
        loss_grad_fp32 = np.random.uniform(
            low=-10.0, high=10.0, size=self.label_shape
        ).astype(np.float32)
        if args["dtype"] == "bfloat16":
            loss_grad = convert_float_to_uint16(loss_grad_fp32)
        else:
            loss_grad = loss_grad_fp32.astype(np_dtype)

        # each xpu uses own half of logits
        np.random.seed(os.getpid())
        logits_fp32 = np.random.uniform(
            low=-40.0, high=40.0, size=self.logits_shape
        ).astype(np.float32)
        if args["dtype"] == "bfloat16":
            logits = convert_float_to_uint16(logits_fp32)
        else:
            logits = logits_fp32.astype(np_dtype)
        out = exe.run(
            train_prog,
            feed={'Logits': logits, 'Label': label, 'Loss@GRAD': loss_grad},
            fetch_list=[loss.name, softmax.name, 'Logits@GRAD'],
        )
        dump_output(out)


if __name__ == "__main__":
    runtime_main(TestCollectiveSoftmaxWithCE, "softmax_with_ce", 0)
