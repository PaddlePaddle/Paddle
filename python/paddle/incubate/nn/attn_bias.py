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


# The following codes are from https://github.com/facebookresearch/xformers

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence

import paddle


class AttentionBias(ABC):
    @abstractmethod
    def materialize(self, shape, dtype=paddle.float32):
        raise NotImplementedError()


class LowerTriangularMask(AttentionBias):
    def materialize(self, shape, dtype=paddle.float32):
        create_as = dtype if dtype is not paddle.bfloat16 else paddle.float32
        tensor = paddle.full(
            shape=shape, fill_value=float("-inf"), dtype=create_as
        )
        return paddle.triu(tensor, diagonal=1).astype(dtype)

    def add_bias(self, bias):
        return LowerTriangularMaskWithTensorBias(bias)


class LowerTriangularMaskWithTensorBias(LowerTriangularMask):
    def __init__(self, bias):
        self._bias = bias

    def materialize(self, shape, dtype=paddle.float32):
        return super().materialize(shape, dtype) + self._bias


@dataclass
class SeqLenInfo:
    seqstart: paddle.Tensor
    max_seqlen: int
    seqstart_py: List[int]

    def intervals(self):
        yield from zip(self.seqstart_py, self.seqstart_py[1:])

    @classmethod
    def from_seqlens(cls, seqlens):
        seqstart_py = [0]
        max_seqlen = -1
        for seqlen in seqlens:
            max_seqlen = max(max_seqlen, seqlen)
            seqstart_py.append(seqstart_py[-1] + seqlen)
        seqstart = paddle.to_tensor(seqstart_py, dtype=paddle.int32)
        return cls(
            max_seqlen=max_seqlen, seqstart=seqstart, seqstart_py=seqstart_py
        )

    def split(self, x, batch_sizes=None):
        assert self.seqstart_py[-1] == x.shape[1] and x.shape[0] == 1
        if batch_sizes is None:
            batch_sizes = [1] * (len(self.seqstart_py) - 1)
        split_chunks = []
        it = 0
        for batch_size in batch_sizes:
            split_chunks.append(
                self.seqstart_py[it + batch_size] - self.seqstart_py[it]
            )
            it += batch_size
        return [
            tensor.reshape([bs, -1, *tensor.shape[2:]])
            for bs, tensor in zip(batch_sizes, x.split(split_chunks, axis=1))
        ]


@dataclass
class PaddedSeqLenInfo(SeqLenInfo):
    seqlen: paddle.Tensor
    seqlen_py: Sequence[int]

    def intervals(self):
        for (start, _), length in zip(super().intervals(), self.seqlen_py):
            yield start, start + length

    @classmethod
    def from_seqlens(cls, seqlens):
        raise NotImplementedError(
            "Please use SeqLenInfo.from_seq_lens() or PaddedSeqLenInfo.from_seq_lens_padded()."
        )

    @classmethod
    def from_seqlens_padded(cls, seqlens, padding):
        assert all(seqlen <= padding for seqlen in seqlens)
        seqstart_py = list(range(0, len(seqlens) * padding + 1, padding))
        return cls(
            seqlen=paddle.to_tensor(seqlens, dtype=paddle.int32),
            seqlen_py=seqlens,
            max_seqlen=max(seqlens),
            seqstart=paddle.to_tensor(seqstart_py, dtype=paddle.int32),
            seqstart_py=seqstart_py,
        )

    def split(self, x, batch_sizes=None):
        raise NotImplementedError()


@dataclass
class BlockDiagonalMask(AttentionBias):
    q_seqinfo: SeqLenInfo
    k_seqinfo: SeqLenInfo
    _batch_sizes: Optional[Sequence[int]] = None

    def _create_block_mask(self, shape, dtype=paddle.float32):
        return paddle.zeros(shape=shape, dtype=dtype)

    def materialize(self, shape, dtype=paddle.float32):
        assert shape[-1] == self.k_seqinfo.seqstart_py[-1]
        assert shape[-2] == self.q_seqinfo.seqstart_py[-1]
        mask = paddle.full(shape[-2:], fill_value=float('-inf'), dtype=dtype)
        for (q_start, q_end), (k_start, k_end) in zip(
            self.q_seqinfo.intervals(), self.k_seqinfo.intervals()
        ):
            sub_shape = [q_end - q_start, k_end - k_start]
            mask[q_start:q_end, k_start:k_end] = self._create_block_mask(
                sub_shape, dtype
            )
        for _ in range(len(shape) - 2):
            mask = mask.unsqueeze(0)
        return mask.expand(shape)

    @classmethod
    def from_seqlens(cls, q_seqlen, kv_seqlen=None):
        assert kv_seqlen is None or len(q_seqlen) == len(kv_seqlen)
        q_seqinfo = SeqLenInfo.from_seqlens(q_seqlen)
        if kv_seqlen is None or q_seqlen == kv_seqlen:
            k_seqinfo = q_seqinfo
        else:
            k_seqinfo = SeqLenInfo.from_seqlens(kv_seqlen)
        return cls(q_seqinfo=q_seqinfo, k_seqinfo=k_seqinfo)

    @classmethod
    def from_tensor_list(cls, tensors):
        batch_sizes = [tensor.shape[0] for tensor in tensors]
        seqlens = []
        for x in tensors:
            for _ in range(x.shape[0]):
                seqlens.append(x.shape[1])
        block_diag = cls.from_seqlens(seqlens)
        block_diag._batch_sizes = batch_sizes
        concated_tensor = paddle.concat(
            [x.reshape([1, -1, *x.shape[2:]]) for x in tensors], axis=1
        )
        return block_diag, concated_tensor

    @classmethod
    def from_tensor_lists_qkv(cls, tensors_q, tensors_k, tensors_v=None):
        assert len(tensors_q) == len(tensors_k)
        assert tensors_v is None or len(tensors_v) == len(tensors_q)
        batch_sizes = [tensor.shape[0] for tensor in tensors_q]
        q_seqlens, kv_seqlens = [], []
        for i, (q, k) in enumerate(zip(tensors_q, tensors_k)):
            assert q.shape[0] == k.shape[0]
            q_seqlens.extend([q.shape[1]] * q.shape[0])
            kv_seqlens.extend([k.shape[1]] * k.shape[0])
            assert tensors_v is None or tensors_v[i].shape[:2] == k.shape[:2]
        block_diag = cls.from_seqlens(q_seqlens, kv_seqlens)
        block_diag._batch_sizes = [x.shape[0] for x in tensors_q]
        return (
            block_diag,
            paddle.concat(
                [x.reshape([1, -1, *x.shape[2:]]) for x in tensors_q], axis=1
            ),
            paddle.concat(
                [x.reshape([1, -1, *x.shape[2:]]) for x in tensors_k], axis=1
            ),
            paddle.concat(
                [x.reshape([1, -1, *x.shape[2:]]) for x in tensors_v], axis=1
            )
            if tensors_v is not None
            else None,
        )

    def split_queries(self, tensor):
        return self.q_seqinfo.split(tensor, self._batch_sizes)

    def split_kv(self, tensor):
        return self.k_seqinfo.split(tensor, self._batch_sizes)

    def split(self, tensor):
        assert self.q_seqinfo is self.k_seqinfo
        return self.q_seqinfo.split(tensor, self._batch_sizes)

    def make_causal(self):
        return BlockDiagonalCausalMask(
            q_seqinfo=self.q_seqinfo,
            k_seqinfo=self.k_seqinfo,
            _batch_sizes=self._batch_sizes,
        )


@dataclass
class BlockDiagonalCausalMask(BlockDiagonalMask):
    def _create_block_mask(self, shape, dtype=paddle.float32):
        return LowerTriangularMask().materialize(shape=shape, dtype=dtype)


@dataclass
class BlockDiagonalCausalWithOffsetPaddedKeysMask(AttentionBias):
    q_seqinfo: SeqLenInfo
    k_seqinfo: PaddedSeqLenInfo
    causal_diagonal: Optional[paddle.Tensor] = None

    def _create_block_mask(self, shape, offset=0, dtype=paddle.float32):
        create_as = dtype if dtype is not paddle.bfloat16 else paddle.float32
        tensor = paddle.full(shape, dtype=create_as, fill_value=float('-inf'))
        return paddle.triu(tensor, diagonal=1 + offset).astype(dtype)

    def materialize(self, shape, dtype=paddle.float32):
        assert shape[-1] == self.k_seqinfo.seqstart_py[-1]
        assert shape[-2] == self.q_seqinfo.seqstart_py[-1]
        mask = paddle.full(shape[-2:], dtype=dtype, fill_value=float('-inf'))
        for i, ((q_start, q_end), (k_start, k_end)) in enumerate(
            zip(self.q_seqinfo.intervals(), self.k_seqinfo.intervals())
        ):
            mask[q_start:q_end, k_start:k_end] = self._create_block_mask(
                (q_end - q_start, k_end - k_start),
                offset=0
                if self.causal_diagonal is None
                else int(self.causal_diagonal[i].item()),
                dtype=dtype,
            )
        for _ in range(len(shape) - 2):
            mask = mask.unsqueeze(0)
        return mask.expand(shape)

    @classmethod
    def from_seqlens(
        cls, q_seqlen, kv_padding, kv_seqlen, causal_diagonal=None
    ):
        assert kv_seqlen is None or len(q_seqlen) == len(kv_seqlen)
        q_seqinfo = SeqLenInfo.from_seqlens(q_seqlen)
        k_seqinfo = PaddedSeqLenInfo.from_seqlens_padded(kv_seqlen, kv_padding)
        return cls(
            q_seqinfo=q_seqinfo,
            k_seqinfo=k_seqinfo,
            causal_diagonal=causal_diagonal,
        )
