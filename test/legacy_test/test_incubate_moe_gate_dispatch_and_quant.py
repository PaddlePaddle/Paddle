# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import itertools
import unittest

import numpy as np

import paddle
from paddle.incubate.nn.functional import (
    fp8,
    moe_gate_dispatch,
    moe_gate_dispatch_and_quant,
)


class Fp8MoeGateDispatchAndQuant(paddle.autograd.PyLayer):
    """Fp8MoeGateDispatchAndQuant"""

    @staticmethod
    def forward(
        ctx,
        x,
        gate_logtis,
        corr_bias,
        k,
        capacity,
        use_pad,
        use_pow2_scale=True,
    ):
        """forward"""
        (
            out_fp8,
            scale,
            combine_weights,
            scatter_index,
            expert_offset,
            expert_id,
        ) = moe_gate_dispatch_and_quant(
            x,
            gate_logtis,
            corr_bias=corr_bias,
            k=k,
            capacity=capacity,
            use_pad=use_pad,
            use_pow2_scale=use_pow2_scale,
        )
        assert out_fp8.shape[0] == scale.shape[0]

        # Maintain computational graph integrity via BF16 proxy tensors
        # Required because this operator produces FP8 outputs but BF16 gradients
        # Current framework implementation enforces gradient/base tensor dtype consistency
        fake_out = paddle.empty(out_fp8.shape, x.dtype)
        fake_out.stop_gradient = False
        combine_weights.stop_gradient = False
        scatter_index.stop_gradient = True
        expert_offset.stop_gradient = True
        expert_id.stop_gradient = True

        out_fp8.stop_gradient = True
        scale.stop_gradient = True

        ctx.k = k
        ctx.capacity = capacity
        ctx.use_pad = use_pad
        ctx.combine_weights = combine_weights
        ctx.scatter_index = scatter_index
        ctx.expert_id = expert_id
        ctx.has_corr_bias = corr_bias is not None

        return (
            fake_out,
            combine_weights,
            scatter_index,
            expert_offset,
            expert_id,
            {
                "fp8_data": out_fp8,
                "scale": scale,
            },
        )

    @staticmethod
    def backward(ctx, *grads):
        """backward"""
        out_grad, combine_weights_grad = grads[0], grads[1]
        x_grad, gate_logits_grad = paddle._C_ops.moe_gate_dispatch_grad(
            ctx.combine_weights,
            ctx.scatter_index,
            ctx.expert_id,
            out_grad,
            combine_weights_grad,
            ctx.k,
            ctx.capacity,
            ctx.use_pad,
        )
        if ctx.has_corr_bias:
            return x_grad, gate_logits_grad, None
        else:
            return x_grad, gate_logits_grad


class FakeOp(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, input, fp8_args=None):
        return fp8_args["fp8_data"].astype("bfloat16")

    @staticmethod
    def backward(ctx, output_grad):
        return output_grad + 1


class TestMoeOpsFP8(unittest.TestCase):
    def setUp(self):
        paddle.seed(42)

    def single_test(self, seq_len, expert_num, moe_k, cap):
        capacity = int(cap * seq_len // expert_num)

        hidden_sizes = [256, 512, 640, 2048]
        use_pad_options = [True, False]
        use_pow2_scale_options = [True, False]

        for hidden_size, use_pad, use_pow2_scale in itertools.product(
            hidden_sizes, use_pad_options, use_pow2_scale_options
        ):
            x = paddle.randn([seq_len, hidden_size], dtype="bfloat16")
            gate_logtis = paddle.randn([seq_len, expert_num], dtype="float32")

            (
                _,
                combine_weights,
                scatter_index,
                expert_offset,
                expert_id,
                fp8_args,
            ) = Fp8MoeGateDispatchAndQuant.apply(
                x,
                gate_logtis,
                corr_bias=None,
                k=moe_k,
                capacity=capacity,
                use_pad=use_pad,
                use_pow2_scale=use_pow2_scale,
            )

            out_fp8, scale = fp8_args["fp8_data"], fp8_args["scale"]

            (
                out_ref,
                combine_weights_ref,
                scatter_index_ref,
                expert_offset_ref,
                expert_id_ref,
            ) = moe_gate_dispatch(
                x,
                gate_logtis,
                corr_bias=None,
                k=moe_k,
                capacity=capacity,
                use_pad=use_pad,
            )

            np.testing.assert_equal(
                combine_weights._md5sum(), combine_weights_ref._md5sum()
            )
            np.testing.assert_equal(
                scatter_index._md5sum(), scatter_index_ref._md5sum()
            )
            np.testing.assert_equal(
                expert_offset._md5sum(), expert_offset_ref._md5sum()
            )
            np.testing.assert_equal(
                expert_id._md5sum(), expert_id_ref._md5sum()
            )

            out_fp8_ref, scale_ref = fp8.fp8_quant_blockwise(
                out_ref,
                quant_method="1x128",
                output_scale_transpose=False,
                using_pow2_scale=use_pow2_scale,
            )

            np.testing.assert_equal(scale.shape, scale_ref.shape)
            np.testing.assert_equal(out_fp8.shape, out_fp8_ref.shape)

            np.testing.assert_equal(scale._md5sum(), scale_ref._md5sum())
            np.testing.assert_equal(
                out_fp8.astype("float32")._md5sum(),
                out_fp8_ref.astype("float32")._md5sum(),
            )

    def test_moe_gate_dispatch_and_quant(self):
        self.single_test(seq_len=4096, expert_num=1, moe_k=1, cap=1)
        self.single_test(seq_len=4096, expert_num=64, moe_k=8, cap=8)
        self.single_test(seq_len=128, expert_num=16, moe_k=8, cap=8)

    def test_fake_pylayer(self):
        hidden_size = 256
        expert_num = 4
        moe_k = 2
        capacity = 2
        use_pad = True

        x = paddle.randn([4096, hidden_size], dtype="bfloat16")
        gate_logtis = paddle.randn([4096, expert_num], dtype="float32")

        x.stop_gradient = False
        gate_logtis.stop_gradient = False

        (
            fake_out,
            combine_weights,
            scatter_index,
            expert_offset,
            expert_id,
            fp8_args,
        ) = Fp8MoeGateDispatchAndQuant.apply(
            x,
            gate_logtis,
            corr_bias=None,
            k=moe_k,
            capacity=capacity,
            use_pad=use_pad,
        )
        fake_out.retain_grads()
        loss = FakeOp.apply(fake_out, fp8_args).sum()

        loss.backward()

        np.testing.assert_equal(
            fake_out.grad.numpy(), paddle.full_like(fake_out, 2).numpy()
        )


if __name__ == "__main__":
    unittest.main()
