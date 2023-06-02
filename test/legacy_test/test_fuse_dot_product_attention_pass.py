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

np.random.seed(0)
paddle.seed(0)


def skip_unit_test():
    return (
        not paddle.is_compiled_with_cuda()
        or paddle.device.cuda.get_device_capability() != (8, 0)
        or paddle.get_cudnn_version() < 8901
    )


skip_msg = (
    "only support with cuda and CUDNN 8.9.1 or later,"
    " and only Ampere devices are supported"
)


def verify_node_count(graph, node_name, target_count):
    count = 0
    for node in graph.nodes():
        if node.name() == node_name:
            count += 1
    return count == target_count


class mha(paddle.nn.Layer):
    def __init__(
        self,
        hidden,
        num_heads,
        dropout=0.0,
        mode=None,
        fuse_qkv=False,
        num_layers=1,
    ):
        super().__init__()
        self.mha_layer = paddle.nn.MultiHeadAttention(
            hidden, num_heads, dropout=dropout, mode=mode, fuse_qkv=fuse_qkv
        )
        self.num_layers = num_layers

    def forward(self, q, kv, mask, is_qkv_same=False):
        if is_qkv_same:
            out = q
            for _ in range(self.num_layers):
                out = self.mha_layer(out, out, out, attn_mask=mask)
        else:
            out = q
            for _ in range(self.num_layers):
                out = self.mha_layer(out, kv, kv, attn_mask=mask)
        loss = paddle.mean(out)
        return loss


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFuseDotProductSelfAttention(unittest.TestCase):
    def setUp(self):
        self.run_steps = 10
        self.num_layers = 3
        self._pre_test_hook()
        self.hidden_dim = self.num_heads * self.head_size
        paddle.enable_static()
        self.place = paddle.CUDAPlace(0)
        self._create_input()
        self.init_weight = (
            np.random.normal(
                loc=0.0, scale=0.02, size=(self.hidden_dim, self.hidden_dim)
            ).astype("float32")
            * 50
        )
        self.check_fused_fwd_op_name = (
            "fused_dot_product_self_attention"
            if self.attn_mode == "self_attn"
            else "fused_dot_product_cross_attention"
        )
        self.check_fused_bwd_op_name = (
            "fused_dot_product_self_attention_grad"
            if self.attn_mode == "self_attn"
            else "fused_dot_product_cross_attention_grad"
        )

    def _set_shape(self):
        self.batch_size = 32
        self.num_heads = 12
        self.head_size = 64
        self.q_seqlen = 128
        self.kv_seqlen = 128

    def _set_mask_type(self):
        # simple_mask = True : [b, 1, 1, seqlen]
        # simple_mask = False: [b, 1, q_seqlen, kv_seqlen]
        self.use_simple_mask = True

    def _set_attn_mode(self):
        self.attn_mode = "self_attn"
        self.fuse_qkv = True

    def _pre_test_hook(self):
        self._set_shape()
        self._set_mask_type()
        self._set_attn_mode()
        if self.use_simple_mask:
            assert self.q_seqlen == self.kv_seqlen, "q_seqlen != kv_seqlen"
        assert self.attn_mode in [
            "self_attn",
            "cross_attn",
            None,
        ], "attn_mode error"
        self.dropout = 0.0
        self.atol = 1e-4
        self.rtol = 1e-4

    def _create_input(self):
        q_input = np.random.normal(
            loc=0.0,
            scale=1.0,
            size=(self.batch_size, self.q_seqlen, self.hidden_dim),
        ).astype(np.float32)
        kv_input = np.random.normal(
            loc=0.0,
            scale=1.0,
            size=(self.batch_size, self.kv_seqlen, self.hidden_dim),
        ).astype(np.float32)

        q_actual_seqlen = np.random.randint(
            low=5, high=self.q_seqlen, size=(self.batch_size,), dtype=np.int32
        )
        kv_actual_seqlen = np.random.randint(
            low=5, high=self.kv_seqlen, size=(self.batch_size,), dtype=np.int32
        )
        attn_mask_arr = np.zeros(
            shape=(self.batch_size, 1, self.q_seqlen, self.kv_seqlen),
            dtype=np.int32,
        )
        for i in range(self.batch_size):
            attn_mask_arr[i, :, : q_actual_seqlen[i], : kv_actual_seqlen[i]] = 1
        self.feed = {
            "_q_input": q_input,
            "_kv_input": kv_input,
            "_attn_mask": attn_mask_arr,
        }

        if self.use_simple_mask:
            q_actual_seqlen[:] = self.q_seqlen
            attn_mask_arr = np.zeros(
                shape=(self.batch_size, 1, 1, self.q_seqlen), dtype=np.int32
            )
            attn_mask_arr[:, :, :, : kv_actual_seqlen[0]] = 1
            self.feed = {
                "_q_input": q_input,
                "_attn_mask": attn_mask_arr,
            }

    def _reset_program_state_dict(self, model, hidden_dim):
        '''
        Set the weight of q, k, v, o proj to be the same value.
        '''
        state_dict = model.state_dict()
        reset_state_dict = {}
        kv_np = np.zeros(shape=(hidden_dim, 2 * hidden_dim), dtype="float32")
        kv_np[:, 0:hidden_dim] = self.init_weight
        kv_np[:, hidden_dim : 2 * hidden_dim] = self.init_weight
        qkv_np = np.zeros(shape=(hidden_dim, 3 * hidden_dim), dtype="float32")
        qkv_np[:, 0:hidden_dim] = self.init_weight
        qkv_np[:, hidden_dim : 2 * hidden_dim] = self.init_weight
        qkv_np[:, 2 * hidden_dim : 3 * hidden_dim] = self.init_weight
        for n, p in state_dict.items():
            if p.shape == (hidden_dim, hidden_dim):
                reset_state_dict[p.name] = self.init_weight
            elif p.shape == (hidden_dim, 2 * hidden_dim):
                reset_state_dict[p.name] = kv_np
            elif p.shape == (hidden_dim, 3 * hidden_dim):
                reset_state_dict[p.name] = qkv_np
        return reset_state_dict

    def _build_program(self, main_prog, startup_prog, mode, fuse_qkv):
        with paddle.static.program_guard(main_prog, startup_prog):
            q_input = paddle.static.data(
                name="_q_input",
                shape=[-1, -1, self.hidden_dim],
                dtype='float32',
            )
            if self.use_simple_mask:
                kv_input = None
                attn_mask = paddle.static.data(
                    name="_attn_mask",
                    shape=[-1, 1, 1, self.q_seqlen],
                    dtype='int32',
                )
            elif self.attn_mode == "self_attn":
                kv_input = None
                attn_mask = paddle.static.data(
                    name="_attn_mask",
                    shape=[-1, 1, self.q_seqlen, self.q_seqlen],
                    dtype='int32',
                )
            else:  # cross_attn
                kv_input = paddle.static.data(
                    name="_kv_input",
                    shape=[-1, -1, self.hidden_dim],
                    dtype='float32',
                )
                attn_mask = paddle.static.data(
                    name="_attn_mask",
                    shape=[-1, 1, self.q_seqlen, self.kv_seqlen],
                    dtype='int32',
                )

            model = mha(
                self.hidden_dim,
                self.num_heads,
                self.dropout,
                mode=mode,
                fuse_qkv=fuse_qkv,
                num_layers=self.num_layers,
            )
            loss = model(
                q_input,
                kv_input,
                attn_mask,
                is_qkv_same=(self.attn_mode == "self_attn"),
            )
            opt = paddle.optimizer.SGD(learning_rate=0.1)
            amp_list = paddle.static.amp.CustomOpLists(
                custom_white_list=['softmax']
            )
            # Only test AMP because cudnn v8 fmha only support half precision currently.
            opt = paddle.static.amp.decorate(
                optimizer=opt,
                amp_lists=amp_list,
                init_loss_scaling=128.0,
                use_dynamic_loss_scaling=True,
            )
            opt.minimize(loss)
        return loss, model

    def _test_ref(self):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        exe = paddle.static.Executor(self.place)

        loss, model = self._build_program(
            main_prog, startup_prog, mode=None, fuse_qkv=False
        )
        exe.run(startup_prog)
        reset_state_dict = self._reset_program_state_dict(
            model, self.hidden_dim
        )
        paddle.static.set_program_state(main_prog, reset_state_dict)
        self.reference = []
        for i in range(self.run_steps):
            loss_return = exe.run(
                main_prog, feed=self.feed, fetch_list=[loss.name]
            )
            self.reference.append(loss_return[0])

    def _test_output(self):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        exe = paddle.static.Executor(self.place)

        loss, model = self._build_program(
            main_prog, startup_prog, mode=self.attn_mode, fuse_qkv=self.fuse_qkv
        )

        exe.run(startup_prog)
        reset_state_dict = self._reset_program_state_dict(
            model, self.hidden_dim
        )
        paddle.static.set_program_state(main_prog, reset_state_dict)
        build_strategy = paddle.static.BuildStrategy()
        build_strategy.fuse_dot_product_attention = True
        program = paddle.static.CompiledProgram(main_prog)
        self.program = program.with_data_parallel(
            loss_name=loss.name,
            build_strategy=build_strategy,
            places=self.place,
        )
        self.result = []
        for i in range(self.run_steps):
            loss_return = exe.run(
                program, feed=self.feed, fetch_list=[loss.name]
            )
            self.result.append(loss_return[0])

    def test_compare(self):
        self._test_ref()
        self._test_output()
        np.testing.assert_allclose(
            self.reference,
            self.result,
            atol=self.atol,
            rtol=self.rtol,
            equal_nan=True,
            err_msg="[{}] outputs are miss-matched.".format(
                type(self).__name__
            ),
        )
        self.assertTrue(
            verify_node_count(
                self.program._graph,
                self.check_fused_fwd_op_name,
                self.num_layers,
            ),
            f"[{type(self).__name__}] The number of {self.check_fused_fwd_op_name} is miss-matched in the computing graph.",
        )
        self.assertTrue(
            verify_node_count(
                self.program._graph,
                self.check_fused_fwd_op_name,
                self.num_layers,
            ),
            f"[{type(self).__name__}] The number of {self.check_fused_fwd_op_name} is miss-matched in the computing graph.",
        )


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFuseDotProductSelfAttentionNoSimpleMask(
    TestFuseDotProductSelfAttention
):
    def _set_mask_type(self):
        self.use_simple_mask = False


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFuseDotProductSelfAttentionSeqlen64(TestFuseDotProductSelfAttention):
    def _set_shape(self):
        self.batch_size = 32
        self.num_heads = 12
        self.head_size = 64
        self.q_seqlen = 64
        self.kv_seqlen = 64


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFuseDotProductSelfAttentionSeqlen192(TestFuseDotProductSelfAttention):
    def _set_shape(self):
        self.batch_size = 32
        self.num_heads = 12
        self.head_size = 64
        self.q_seqlen = 192
        self.kv_seqlen = 192


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFuseDotProductSelfAttentionSeqlen256(TestFuseDotProductSelfAttention):
    def _set_shape(self):
        self.batch_size = 32
        self.num_heads = 12
        self.head_size = 64
        self.q_seqlen = 256
        self.kv_seqlen = 256


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFuseDotProductSelfAttentionSeqlen320(TestFuseDotProductSelfAttention):
    def _set_shape(self):
        self.batch_size = 32
        self.num_heads = 12
        self.head_size = 64
        self.q_seqlen = 320
        self.kv_seqlen = 320


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFuseDotProductSelfAttentionSeqlen384(TestFuseDotProductSelfAttention):
    def _set_shape(self):
        self.batch_size = 32
        self.num_heads = 12
        self.head_size = 64
        self.q_seqlen = 384
        self.kv_seqlen = 384


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFuseDotProductSelfAttentionSeqlen448(TestFuseDotProductSelfAttention):
    def _set_shape(self):
        self.batch_size = 32
        self.num_heads = 12
        self.head_size = 64
        self.q_seqlen = 448
        self.kv_seqlen = 448


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFuseDotProductSelfAttentionSeqlen512(TestFuseDotProductSelfAttention):
    def _set_shape(self):
        self.batch_size = 32
        self.num_heads = 12
        self.head_size = 64
        self.q_seqlen = 512
        self.kv_seqlen = 512


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFuseDotProductCrossAttention(TestFuseDotProductSelfAttention):
    def _set_attn_mode(self):
        self.attn_mode = "cross_attn"
        self.fuse_qkv = True


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFuseDotProductCrossAttentionNoSimpleMask(
    TestFuseDotProductSelfAttention
):
    def _set_mask_type(self):
        self.use_simple_mask = False

    def _set_attn_mode(self):
        self.attn_mode = "cross_attn"
        self.fuse_qkv = True

    def _set_shape(self):
        self.batch_size = 32
        self.num_heads = 12
        self.head_size = 64
        self.q_seqlen = 128
        self.kv_seqlen = 256


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFuseDotProductCrossAttentionNoSimpleMask2(
    TestFuseDotProductSelfAttention
):
    def _set_mask_type(self):
        self.use_simple_mask = False

    def _set_attn_mode(self):
        self.attn_mode = "cross_attn"
        self.fuse_qkv = True

    def _set_shape(self):
        self.batch_size = 32
        self.num_heads = 12
        self.head_size = 64
        self.q_seqlen = 192
        self.kv_seqlen = 448


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFuseDotProductCrossAttentionNoSimpleMask3(
    TestFuseDotProductSelfAttention
):
    def _set_mask_type(self):
        self.use_simple_mask = False

    def _set_attn_mode(self):
        self.attn_mode = "cross_attn"
        self.fuse_qkv = True

    def _set_shape(self):
        self.batch_size = 32
        self.num_heads = 12
        self.head_size = 64
        self.q_seqlen = 512
        self.kv_seqlen = 384


@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFuseDotProductCrossAttentionNoSimpleMask4(
    TestFuseDotProductSelfAttention
):
    def _set_mask_type(self):
        self.use_simple_mask = False

    def _set_attn_mode(self):
        self.attn_mode = "cross_attn"
        self.fuse_qkv = True

    def _set_shape(self):
        self.batch_size = 32
        self.num_heads = 12
        self.head_size = 64
        self.q_seqlen = 320
        self.kv_seqlen = 64


if __name__ == "__main__":
    unittest.main()
