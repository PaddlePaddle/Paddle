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

import time
import unittest

import numpy as np

import paddle
import paddle.nn.functional as F
from paddle.nn.functional.flash_attention import (
    flash_attention,
    flash_attn_unpadded,
    scaled_dot_product_attention,
)


def attention_naive(q, k, v, causal=False):
    qt = paddle.transpose(q, [0, 2, 1, 3])
    kt = paddle.transpose(k, [0, 2, 1, 3])
    vt = paddle.transpose(v, [0, 2, 1, 3])
    scale = 1.0 / np.sqrt(q.shape[-1])
    s = paddle.matmul(qt, paddle.transpose(kt, [0, 1, 3, 2]))
    s = paddle.scale(s, scale)
    p = (
        paddle.incubate.softmax_mask_fuse_upper_triangle(s)
        if causal
        else F.softmax(s)
    )
    o = paddle.matmul(p, vt)
    return paddle.transpose(o, [0, 2, 1, 3])


def attention_naive_with_mask(q, k, v, attn_bias):
    qt = paddle.transpose(q, [0, 2, 1, 3])
    kt = paddle.transpose(k, [0, 2, 1, 3])
    vt = paddle.transpose(v, [0, 2, 1, 3])
    scale = 1.0 / np.sqrt(q.shape[-1])
    s = paddle.matmul(qt, paddle.transpose(kt, [0, 1, 3, 2]))
    s = paddle.scale(s, scale)
    p = F.softmax(s + attn_bias)
    o = paddle.matmul(p, vt)
    return paddle.transpose(o, [0, 2, 1, 3])


class TestFlashAttentionAPI(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.shape = (2, 2048, 40, 128)
        self.kvshape = (2, 2048, 40, 128)
        self.dtype = 'float16'
        self.dropout = 0.0
        self.causal = False
        self.perf_test = False
        self.return_softmax = False
        self.use_sdp_kernel = False
        self.use_sdp_api = False

    def test_unpadded(self):
        print(
            f"Test unpadded case shape {self.shape} dtype {self.dtype} causal {self.causal} test_unpadded"
        )

        paddle.disable_static()

        query = np.random.randn(*self.shape)
        key = np.random.randn(*self.kvshape)
        value = np.random.randn(*self.kvshape)
        out_grad = np.random.randn(*self.shape)

        dout = paddle.to_tensor(
            out_grad, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        q = paddle.to_tensor(
            query, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        q_ = paddle.to_tensor(
            query, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        k = paddle.to_tensor(
            key, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        k_ = paddle.to_tensor(
            key, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        v = paddle.to_tensor(
            value, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        v_ = paddle.to_tensor(
            value, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        out_ = attention_naive(q_, k_, v_, self.causal)

        scale = 1.0 / np.sqrt(q.shape[-1])

        bs = self.shape[0]
        ms = self.shape[1]
        kvms = self.kvshape[1]
        nh = self.shape[2]
        hd = self.shape[3]
        cu_q = paddle.arange(0, (bs + 1) * ms, ms, dtype='int32')
        cu_k = paddle.arange(0, (bs + 1) * kvms, kvms, dtype='int32')
        qq = paddle.reshape(q, [bs * ms, nh, hd])
        kk = paddle.reshape(k, [bs * kvms, nh, hd])
        vv = paddle.reshape(v, [bs * kvms, nh, hd])
        out, _ = flash_attn_unpadded(
            qq,
            kk,
            vv,
            cu_q,
            cu_k,
            ms,
            kvms,
            scale,
            self.dropout,
            self.causal,
            self.return_softmax,
        )
        out_ = paddle.reshape(out_, [bs * ms, nh, hd])

        np.testing.assert_allclose(
            out.numpy(), out_.numpy(), rtol=1e-02, atol=1e-02
        )

        dout_ = paddle.reshape(dout, [bs * ms, nh, hd])
        out.backward(dout_)
        out_.backward(dout_)

        np.testing.assert_allclose(
            q.grad.numpy(), q_.grad.numpy(), rtol=1e-02, atol=1e-02
        )

        np.testing.assert_allclose(
            k.grad.numpy(), k_.grad.numpy(), rtol=1e-02, atol=1e-02
        )
        np.testing.assert_allclose(
            v.grad.numpy(), v_.grad.numpy(), rtol=1e-02, atol=1e-02
        )

    def test_all(self):
        print(
            f"Test case shape {self.shape} dtype {self.dtype} causal {self.causal} test_all"
        )
        # test dynamic
        paddle.disable_static()

        query = np.random.randn(*self.shape)
        key = np.random.randn(*self.shape)
        value = np.random.randn(*self.shape)
        out_grad = np.random.randn(*self.shape)

        dout = paddle.to_tensor(
            out_grad, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        q = paddle.to_tensor(
            query, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        k = paddle.to_tensor(
            key, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        v = paddle.to_tensor(
            value, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        q_ = paddle.to_tensor(
            query, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        k_ = paddle.to_tensor(
            key, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        v_ = paddle.to_tensor(
            value, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        if self.use_sdp_kernel:
            with paddle.nn.functional.sdp_kernel(
                enable_math=self.enable_math,
                enable_flash=self.enable_flash,
                enable_mem_efficient=self.enable_mem_efficient,
            ):
                if self.use_sdp_api:
                    out = scaled_dot_product_attention(
                        q, k, v, None, self.dropout, self.causal
                    )
                else:
                    out, _ = flash_attention(
                        q, k, v, self.dropout, self.causal, self.return_softmax
                    )

        else:
            out, _ = flash_attention(
                q, k, v, self.dropout, self.causal, self.return_softmax
            )
        out_ = attention_naive(q_, k_, v_, self.causal)

        out.backward(dout)
        out_.backward(dout)

        np.testing.assert_allclose(
            out.numpy(), out_.numpy(), rtol=1e-02, atol=1e-02
        )

        self.assertEqual(q.grad.shape, q.shape)
        self.assertEqual(q_.grad.shape, q.shape)

        np.testing.assert_allclose(
            q.grad.numpy(), q_.grad.numpy(), rtol=1e-02, atol=1e-02
        )
        np.testing.assert_allclose(
            k.grad.numpy(), k_.grad.numpy(), rtol=1e-02, atol=1e-02
        )
        np.testing.assert_allclose(
            v.grad.numpy(), v_.grad.numpy(), rtol=1e-02, atol=1e-02
        )

        for _ in range(3):
            out, _ = flash_attention(
                q, k, v, self.dropout, self.causal, self.return_softmax
            )
            out_ = attention_naive(q_, k_, v_, self.causal)

        paddle.device.synchronize()

        start = time.time()
        for _ in range(3):
            out, _ = flash_attention(
                q, k, v, self.dropout, self.causal, self.return_softmax
            )
        paddle.device.synchronize()
        end = time.time()
        print(f"flash attention fwd time {(end - start) / 3 * 1000} ms")

        start = time.time()
        for _ in range(3):
            out_ = attention_naive(q_, k_, v_, self.causal)
        paddle.device.synchronize()
        end = time.time()
        print(f"native attention fwd time {(end - start) / 3 * 1000} ms")

        start = time.time()
        for _ in range(3):
            out.backward(dout, retain_graph=True)
        paddle.device.synchronize()
        end = time.time()
        print(f"flash attention bwd time {(end - start) / 3 * 1000} ms")

        start = time.time()
        for _ in range(3):
            out_.backward(dout, retain_graph=True)
        paddle.device.synchronize()
        end = time.time()
        print(f"native attention bwd time {(end - start) / 3 * 1000} ms")


class TestFlashAttentionGQA(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.num_head = 8
        self.seq_len = 2048
        self.head_dim = 128
        self.num_group = 2
        self.dtype = 'bfloat16'

    def gen_unpadded_data(self, dtype):
        seq_len_q = np.random.randint(
            low=1, high=self.seq_len, size=[self.batch_size]
        )
        seq_len_k = np.random.randint(
            low=1, high=self.seq_len, size=[self.batch_size]
        )
        cu_seqlen_q = paddle.to_tensor(
            [0, *np.cumsum(seq_len_q).tolist()], dtype=paddle.int32
        )
        cu_seqlen_k = paddle.to_tensor(
            [0, *np.cumsum(seq_len_k).tolist()], dtype=paddle.int32
        )

        qs, ks, vs = [], [], []
        for i in range(self.batch_size):
            tmp_q = (
                paddle.randn(
                    [seq_len_q[i] * self.num_head * self.head_dim], dtype=dtype
                )
                / 1e2
            )
            tmp_k = (
                paddle.randn(
                    [
                        seq_len_k[i]
                        * self.num_head
                        * self.head_dim
                        // self.num_group
                    ],
                    dtype=dtype,
                )
                / 1e2
            )
            tmp_v = (
                paddle.randn(
                    [
                        seq_len_k[i]
                        * self.num_head
                        * self.head_dim
                        // self.num_group
                    ],
                    dtype=dtype,
                )
                / 1e2
            )
            qs.append(tmp_q)
            ks.append(tmp_k)
            vs.append(tmp_v)

        q = paddle.concat(qs, axis=0).reshape(
            [-1, self.num_head, self.head_dim]
        )
        k = paddle.concat(ks, axis=0).reshape(
            [-1, self.num_head // self.num_group, self.head_dim]
        )
        v = paddle.concat(vs, axis=0).reshape(
            [-1, self.num_head // self.num_group, self.head_dim]
        )
        return q, k, v, cu_seqlen_q, cu_seqlen_k

    def gen_test_data(self, dtype, use_unpadded):
        assert self.num_head % self.num_group == 0
        if use_unpadded:
            q, k, v, cu_seqlen_q, cu_seqlen_k = self.gen_unpadded_data(dtype)
        else:
            q = (
                paddle.randn(
                    [
                        self.batch_size,
                        self.seq_len,
                        self.num_head,
                        self.head_dim,
                    ],
                    dtype=dtype,
                )
                / 1e2
            )
            k = (
                paddle.randn(
                    [
                        self.batch_size,
                        self.seq_len,
                        self.num_head // self.num_group,
                        self.head_dim,
                    ],
                    dtype=dtype,
                )
                / 1e2
            )
            v = (
                paddle.randn(
                    [
                        self.batch_size,
                        self.seq_len,
                        self.num_head // self.num_group,
                        self.head_dim,
                    ],
                    dtype=dtype,
                )
                / 1e2
            )
            cu_seqlen_q = None
            cu_seqlen_k = None
        out_grad = paddle.randn(q.shape, dtype=dtype) / 1e2
        return q, k, v, cu_seqlen_q, cu_seqlen_k, out_grad

    def clone_tensor(self, tensor):
        if tensor is None:
            return None
        elif isinstance(tensor, (list, tuple)):
            return [self.clone_tensor(t) for t in tensor]
        else:
            tensor = tensor.detach().clone()
            tensor.stop_gradient = False
            return tensor

    @paddle.no_grad()
    def convert_dtype(self, tensors):
        ret = []
        for t in tensors:
            if t.dtype in [paddle.float16, paddle.bfloat16]:
                t = t.astype(paddle.float32)
            t = t.numpy()
            ret.append(t)
        return ret

    def calc_fa(
        self, q, k, v, cu_seqlen_q, cu_seqlen_k, out_grad, causal, use_unpadded
    ):
        q, k, v = self.clone_tensor([q, k, v])
        if use_unpadded:
            scale = self.head_dim ** (-0.5)
            out = flash_attn_unpadded(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlen_q,
                cu_seqlens_k=cu_seqlen_k,
                max_seqlen_q=self.seq_len,
                max_seqlen_k=self.seq_len,
                scale=scale,
                causal=causal,
            )
        else:
            out = flash_attention(q, k, v, causal=causal)
        out = out[0]
        out.backward(out_grad)
        return self.convert_dtype([out, q.grad, k.grad, v.grad])

    def calc_raw_attn(
        self, q, k, v, cu_seqlen_q, cu_seqlen_k, out_grad, causal, use_unpadded
    ):
        q, k, v = self.clone_tensor([q, k, v])
        if use_unpadded:
            qq, q_mask = self.pad(q, cu_seqlen_q, self.seq_len)
            kk, k_mask = self.pad(k, cu_seqlen_k, self.seq_len)
            vv, _ = self.pad(v, cu_seqlen_k, self.seq_len)
            qk_mask = paddle.matmul(q_mask, k_mask, transpose_y=True)
            qk_mask = qk_mask.reshape(
                [self.batch_size, 1, self.seq_len, self.seq_len]
            )
            qk_mask[qk_mask == 0] = -1e6
            qk_mask[qk_mask == 1] = 0
        else:
            qq, kk, vv = q, k, v

        assert len(qq.shape) == 4, qq.shape
        assert len(kk.shape) == 4, kk.shape
        assert len(vv.shape) == 4, vv.shape
        perm = [0, 2, 1, 3]
        qq = paddle.transpose(qq, perm)
        kk = paddle.transpose(kk, perm)
        kk = paddle.stack([kk] * self.num_group, axis=2).reshape(qq.shape)
        vv = paddle.transpose(vv, perm)
        vv = paddle.stack([vv] * self.num_group, axis=2).reshape(qq.shape)
        scale = self.head_dim ** (-0.5)
        weight = paddle.matmul(qq * scale, kk, transpose_y=True)
        if use_unpadded:
            weight += qk_mask
        if causal:
            shape = weight.shape[-2:]
            mask = paddle.full(shape, -np.inf, dtype=weight.dtype)
            mask = paddle.triu(mask, diagonal=1)
            weight += mask

        weight = weight.astype(paddle.float32)
        weight = F.softmax(weight)
        out = paddle.matmul(weight.astype(vv.dtype), vv)
        out = paddle.transpose(out, perm)
        if use_unpadded:
            out = self.unpad(out, cu_seqlen_q)
        out.backward(out_grad)
        return self.convert_dtype([out, q.grad, k.grad, v.grad])

    def pad(self, x, cu_seqlen, max_seqlen):
        cu_seqlen_cpu = cu_seqlen.numpy()
        split_sections = []
        for i in range(len(cu_seqlen_cpu) - 1):
            split_sections.append(cu_seqlen_cpu[i + 1] - cu_seqlen_cpu[i])

        tmp_xs = paddle.split(x, split_sections)
        batch_size = len(tmp_xs)
        tmp_masks = []
        tmp_x_pads = []
        for i in range(batch_size):
            tmp_mask = paddle.ones([max_seqlen], dtype=x.dtype)
            tmp_mask[split_sections[i] :] = 0
            tmp_mask = tmp_mask.reshape([1, -1, 1])
            tmp_masks.append(tmp_mask)

            tmp_shape = tmp_xs[i].shape
            tmp_pad = paddle.zeros(
                [max_seqlen - tmp_shape[0], *list(tmp_shape[1:])], dtype=x.dtype
            )
            tmp_x = paddle.concat([tmp_xs[i], tmp_pad]).unsqueeze(0)
            tmp_x_pads.append(tmp_x)

        x_pad = paddle.concat(tmp_x_pads)
        mask = paddle.concat(tmp_masks)
        return x_pad, mask

    def unpad(self, x, cu_seqlen):
        cu_seqlen_cpu = cu_seqlen.numpy()
        xs = paddle.split(x, x.shape[0])
        tmp_xs = []
        for i in range(len(cu_seqlen_cpu) - 1):
            tmp = xs[i].squeeze(0)[: cu_seqlen_cpu[i + 1] - cu_seqlen_cpu[i]]
            tmp_xs.append(tmp)
        unpad_x = paddle.concat(tmp_xs)
        return unpad_x

    def test_main(self):
        for causal in [False]:
            for use_unpadded in [True]:
                (
                    q,
                    k,
                    v,
                    cu_seqlen_q,
                    cu_seqlen_k,
                    out_grad,
                ) = self.gen_test_data(self.dtype, use_unpadded)
                fa_out = self.calc_fa(
                    q,
                    k,
                    v,
                    cu_seqlen_q,
                    cu_seqlen_k,
                    out_grad,
                    causal,
                    use_unpadded,
                )
                raw_out = self.calc_raw_attn(
                    q,
                    k,
                    v,
                    cu_seqlen_q,
                    cu_seqlen_k,
                    out_grad,
                    causal,
                    use_unpadded,
                )
                assert len(fa_out) == len(raw_out)
                for t1, t2 in zip(fa_out, raw_out):
                    np.testing.assert_allclose(t1, t2, atol=1e-2, rtol=1e-2)


if __name__ == '__main__':
    unittest.main()
