# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import numpy as np

import paddle
from paddle.distribution import distribution
from paddle.tensor import multinomial


class Categorical(distribution.Distribution):
    has_enumerate_support = True

    def __init__(
        self,
        probs=None,
        logits=None,
        validate_args: bool | None = None,
    ) -> None:
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        if probs is not None:
            if probs.dim() < 1:
                raise ValueError(
                    "`probs` parameter must be at least one-dimensional."
                )
            self._probs = probs / probs.sum(-1, keepdim=True)
            self._logits = None
            self._param = self._probs
        else:
            if logits.dim() < 1:
                raise ValueError(
                    "`logits` parameter must be at least one-dimensional."
                )
            self._logits = logits - paddle.logsumexp(
                logits, axis=-1, keepdim=True
            )
            self._probs = None
            self._param = self._logits

        self._num_events = self._param.shape[-1]
        batch_shape = (
            tuple(self._param.shape[:-1]) if self._param.dim() > 1 else ()
        )
        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def logits(self):
        if self._logits is None:
            eps = paddle.finfo(self.probs.dtype).eps
            probs = paddle.clip(self.probs, min=eps, max=1 - eps)
            self._logits = paddle.log(probs)
        return self._logits

    @property
    def probs(self):
        if self._probs is None:
            self._probs = paddle.nn.functional.softmax(self.logits, axis=-1)
        return self._probs

    @property
    def mode(self):
        return paddle.argmax(self.probs, axis=-1)

    def sample(self, sample_shape=(), shape=None):
        if shape is not None:
            sample_shape = shape
        sample_shape = tuple(sample_shape)
        probs_2d = self.probs.reshape([-1, self._num_events])
        samples_2d = multinomial(
            probs_2d, np.prod(np.array(sample_shape)), True
        ).T
        return samples_2d.reshape(sample_shape + self._batch_shape)

    def log_prob(self, value):
        value = paddle.cast(value, dtype='int64').unsqueeze(-1)
        log_pmf = self.logits
        output_shape = paddle.broadcast_shape(value.shape, log_pmf.shape)
        value = paddle.broadcast_to(value, [*output_shape[:-1], 1])
        log_pmf = paddle.broadcast_to(log_pmf, output_shape)
        return paddle.take_along_axis(log_pmf, value, axis=-1).squeeze(-1)

    def entropy(self):
        min_real = paddle.finfo(self.logits.dtype).min
        logits = paddle.clip(self.logits, min=min_real)
        return -(logits * self.probs).sum(-1)
