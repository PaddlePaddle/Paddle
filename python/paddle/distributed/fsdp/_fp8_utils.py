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

import math
import types

import paddle


def _quantizer(params):
    quantizers = {getattr(p, "_fsdp_fp8_quantizer", None) for p in params}
    if len(quantizers) > 1:
        raise ValueError("fp8_param_gather: a buffer group mixes quantizers")
    return quantizers.pop()


def _shape(param):
    return list(getattr(param, "original_shape", None) or param.shape)


def fp8_shard_numel(params):
    quantizer = _quantizer(params)
    if quantizer is None:
        return None
    numel = quantizer.block**2
    for p in params:
        shape = tuple(_shape(p))
        reason = quantizer.check(p, shape) if len(shape) == 3 else "not 3-D"
        if reason is not None:
            raise ValueError(f"fp8_param_gather: {p.name}: {reason}")
        numel = math.lcm(numel, shape[1] * shape[2])
    return numel


def init_fp8_buffer(params_buffer, params):
    if not params_buffer.is_sharded:
        raise ValueError(
            f"fp8_param_gather: group {params_buffer.unique_key} is not sharded"
        )
    quantizer = _quantizer(params)
    k, full = len(quantizer.orientations), params_buffer.total_buffer_size
    numels = [full] * k + [full // quantizer.block**2] * k
    dtypes = [paddle.float8_e4m3fn] * k + [paddle.float32] * k
    params_buffer.fp8 = types.SimpleNamespace(
        params=params,
        quantizer=quantizer,
        quantized=False,
        shards=[
            paddle.empty([n // params_buffer.fsdp_degree], d)
            for n, d in zip(numels, dtypes)
        ],
        fulls=None,
        weight_copy=None,
    )


def build_ckpt_weight(params_buffer, dtype):
    params_buffer.fp8.weight_copy = params_buffer.data_buffer.astype(dtype)
    return params_buffer.fp8.weight_copy


@paddle.autograd.no_grad()
def sync_ckpt_weight_to_main(params_buffer):
    fp8 = params_buffer.fp8
    weight, fp8.weight_copy = fp8.weight_copy, None
    if weight is None:
        return
    main = params_buffer.data_buffer
    if not bool(paddle.equal_all(main.astype(weight.dtype), weight)):
        paddle.assign(weight.astype(main.dtype), main)
    fp8.quantized = False


@paddle.autograd.no_grad()
def _quantize_fp8(params_buffer):
    fp8 = params_buffer.fp8
    if fp8.quantized:
        return
    quantizer = fp8.quantizer
    k, bn = len(quantizer.orientations), quantizer.block**2
    for param in fp8.params:
        shard = params_buffer.param_shard(param)
        if shard is None:
            continue
        lo, hi, _ = shard
        _, rows, cols = _shape(param)
        weights = paddle._C_ops.view_slice(params_buffer.data_buffer, lo, hi)
        weights.get_tensor()._set_dims([(hi - lo) // (rows * cols), rows, cols])
        weights = weights.astype(param.dtype)
        for i, transpose in enumerate(quantizer.orientations):
            data, scale = quantizer.quantize(weights, transpose)
            paddle.assign(data.flatten(), fp8.shards[i]._slice(lo, hi))
            paddle.assign(
                scale.flatten(), fp8.shards[k + i]._slice(lo // bn, hi // bn)
            )
    fp8.quantized = True


def get_fp8_tmp_buffer(params_buffer):
    _quantize_fp8(params_buffer)
    fp8 = params_buffer.fp8
    fp8.fulls = [
        paddle.empty([s.shape[0] * params_buffer.fsdp_degree], s.dtype)
        for s in fp8.shards
    ]
    return list(zip(fp8.fulls, fp8.shards))


def bind_fp8_param(params_buffer, param):
    fp8 = params_buffer.fp8
    k, bn = len(fp8.quantizer.orientations), fp8.quantizer.block**2
    shape = _shape(param)
    begin = params_buffer.param_offsets[param.name]
    end = begin + math.prod(shape)
    fp8.quantizer.bind(
        param,
        shape,
        [data._slice(begin, end) for data in fp8.fulls[:k]],
        [scale._slice(begin // bn, end // bn) for scale in fp8.fulls[k:]],
    )
    # Zero-byte storage: read only through the fp8 caches.
    dims = param.shape
    share_scratch(param, param.dtype)
    param._clear_to_zero_allocation()
    param.get_tensor()._set_dims(dims)


def share_scratch(param, dtype):
    stop_gradient = param.stop_gradient
    scratch = paddle.empty([1], dtype=dtype)
    param.get_tensor()._share_data_with(scratch.get_tensor())
    param.stop_gradient = stop_gradient


def clear_fp8_tmp_buffer(params_buffer):
    for param in params_buffer.fp8.params:
        params_buffer.fp8.quantizer.clear(param)
    params_buffer.fp8.fulls = None
