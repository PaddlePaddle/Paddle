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
import paddle
import paddle.distributed.auto_parallel as auto

from paddle.fluid import program_guard
from paddle.incubate.autograd import prim2orig, enable_prim, prim_enabled
from paddle.fluid.layer_helper import LayerHelper
from paddle.distributed.auto_parallel.utils import print_program_with_dist_attr
import paddle.distributed.auto_parallel as auto
from paddle.distributed.auto_parallel.completion import Completer
from paddle.distributed.auto_parallel.partitioner import Partitioner
from paddle.distributed.auto_parallel.utils import set_var_dist_attr
from paddle.distributed.auto_parallel.dist_context import DistributedContext, get_default_distributed_context, set_default_distributed_context

paddle.enable_static()
enable_prim()
nranks = 2
rank = 0


class TestPrimDistOp(unittest.TestCase):

    def setUp(self):
        self.main_program = paddle.static.Program()
        self.startup_program = paddle.static.Program()
        self.layer_help = LayerHelper('TestPrimDistOp')

        with paddle.static.program_guard(self.main_program,
                                         self.startup_program):
            self.init_prog()

    def init_prog(self):
        # block = self.main_program.global_block()
        # block = self.main_program.global_block()
        self.w = self.layer_help.create_parameter(dtype="float",
                                                  shape=[20],
                                                  attr=None)
        self.w_grad = paddle.static.data(name='w_grad',
                                         shape=[20],
                                         dtype='float')
        self.tmp1 = paddle.static.data(name='tmp1', shape=[20], dtype='float')
        self.tmp2 = paddle.static.data(name='tmp2', shape=[20], dtype='float')
        self.batch_reduced = paddle.static.data(name='batch_reduced',
                                                shape=[1],
                                                dtype='float')
        self.attrs = {}

        default_dist_context = get_default_distributed_context()
        _global_process_mesh = auto.ProcessMesh(list(range(nranks)))
        tensor_dist_attr = set_var_dist_attr(default_dist_context,
                                             self.tmp1, [-1],
                                             _global_process_mesh,
                                             mark_annotated=True)
        tensor_dist_attr = set_var_dist_attr(default_dist_context,
                                             self.tmp1, [-1],
                                             _global_process_mesh,
                                             mark_annotated=True)

        op = self.layer_help.append_op(type="add_p",
                                       inputs={
                                           'X': self.tmp1,
                                           'Y': self.w
                                       },
                                       outputs={'Z': self.w_grad},
                                       attrs=self.attrs)

        op = self.layer_help.append_op(type="reduce_sum_p",
                                       inputs={'X': self.tmp2},
                                       outputs={'Y': self.batch_reduced},
                                       attrs={"axis": [0]})

    def test_loss_and_grad_allreduce(self):

        dist_context = DistributedContext(self.main_program,
                                          self.startup_program)
        completer = Completer(dist_context)
        completer.complete_prim_annotation(self.main_program)
        dist_context.block_state.parse_forward_blocks(self.main_program)
        dist_context.block_state.parse_backward_blocks(self.main_program)
        dist_context.grads_params = dict()
        dist_context.grads_params[self.w_grad.name] = self.w.name
        dist_context.synced_gradient = set()
        dist_context.data_parallel_group = list(range(nranks))
        partitioner = Partitioner(dist_context, rank)
        dist_main_prog, dist_startup_prog, _ = partitioner.partition(
            self.main_program, self.startup_program, [(self.w, self.w_grad)])
        ops = dist_main_prog.global_block().ops

        self.assertTrue(ops[1].type == "c_allreduce_sum")
        self.assertTrue(ops[3].type == "c_allreduce_sum")


if __name__ == "__main__":
    unittest.main()
