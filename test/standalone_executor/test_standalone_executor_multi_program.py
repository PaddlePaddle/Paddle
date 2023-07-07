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


import unittest

import numpy as np

import paddle
from paddle.distributed.passes.pass_utils import split_program
from paddle.fluid import core
from paddle.fluid.executor import (
    _add_feed_fetch_ops,
    _as_lodtensor,
    _StandaloneExecutor,
    check_feed_shape_type,
)
from paddle.nn import TransformerEncoderLayer

paddle.enable_static()


class TestMulitProgramRun(unittest.TestCase):
    def setUp(self):
        self.place_desc = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        self.place = core.Place()
        self.place.set_place(self.place_desc)

    def build_program(self):
        batch_size = 2
        src_len = 4
        d_model = 128
        n_head = 2

        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()

        with paddle.static.program_guard(main_program, startup_program):
            enc_input = paddle.static.data(
                name="enc_input",
                shape=[batch_size, src_len, d_model],
                dtype="float32",
            )
            attn_mask = paddle.static.data(
                name="attn_mask",
                shape=[batch_size, n_head, src_len, src_len],
                dtype="float32",
            )
            encoder_layer = TransformerEncoderLayer(
                d_model, n_head, dim_feedforward=512
            )
            attn_mask = paddle.nn.layer.transformer._convert_attention_mask(
                attn_mask, enc_input.dtype
            )

            enc_output = encoder_layer(enc_input, attn_mask)

            split_op_indics = [len(main_program.block(0).ops)]

            enc_output = encoder_layer(enc_output, attn_mask)

            np.random.seed(2022)
            feed = {
                enc_input.name: np.random.rand(
                    batch_size, src_len, d_model
                ).astype(np.float32),
                attn_mask.name: np.random.rand(
                    batch_size, n_head, src_len, src_len
                ).astype(np.float32),
            }
            fetch_list = [enc_output.name]

            return (
                startup_program,
                main_program,
                split_op_indics,
                feed,
                fetch_list,
            )

    def feed_data(self, program, feed, feed_var_name, scope):
        # feed var to framework
        global_block = program.global_block()
        for op in global_block.ops:
            if op.desc.type() == 'feed':
                feed_target_name = op.desc.output('Out')[0]
                cur_feed = feed[feed_target_name]
                var = global_block.var(feed_target_name)
                if var.dtype != core.VarDesc.VarType.STRINGS:
                    if not isinstance(cur_feed, core.LoDTensor):
                        cur_feed = _as_lodtensor(
                            cur_feed, self.place_desc, var.dtype
                        )
                    check_feed_shape_type(var, cur_feed)
                idx = op.desc.attr('col')
                core.set_feed_variable(scope, cur_feed, feed_var_name, idx)
            else:
                break

    def run_program(
        self,
        startup_program,
        main_program,
        feed,
        fetch_list,
        scope,
        run_step,
        split_op_indics=None,
    ):
        paddle.seed(2022)

        startup_exe = _StandaloneExecutor(self.place, [startup_program], scope)
        startup_exe.run(scope, [], [])

        programs = [main_program]
        if split_op_indics is not None:
            programs, _, _ = split_program(main_program, split_op_indics)
            # hack add feed ops in the first program and fetch ops in the last program
            programs[0] = _add_feed_fetch_ops(
                programs[0], feed, [], "feed", "fetch"
            )
            programs[-1] = _add_feed_fetch_ops(
                programs[-1], [], fetch_list, "feed", "fetch"
            )
        else:
            programs[0] = _add_feed_fetch_ops(
                programs[0], feed, fetch_list, "feed", "fetch"
            )

        self.feed_data(programs[0], feed, "feed", scope)

        main_exe = _StandaloneExecutor(self.place, programs, scope)

        res = []
        for i in range(run_step):
            res += main_exe.run(scope, list(feed.keys()), fetch_list)
        return res

    def test_multi_program_run(self):
        (
            startup_program,
            main_program,
            split_op_indics,
            feed,
            fetch_list,
        ) = self.build_program()
        run_step = 3
        res = self.run_program(
            startup_program,
            main_program,
            feed,
            fetch_list,
            paddle.static.Scope(),
            run_step,
        )
        splited_res = self.run_program(
            startup_program,
            main_program,
            feed,
            fetch_list,
            paddle.static.Scope(),
            run_step,
            split_op_indics,
        )
        np.testing.assert_array_equal(res, splited_res)


if __name__ == "__main__":
    unittest.main()
