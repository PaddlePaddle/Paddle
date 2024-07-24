// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "flash_attn.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/flags.h"
#include "paddle/phi/kernels/empty_kernel.h"
#ifdef PADDLE_WITH_FLASHATTN
#include "paddle/phi/backends/dynload/flashattn.h"
#endif

PD_DECLARE_bool(cudnn_deterministic);

namespace phi {

#ifdef PADDLE_WITH_FLASHATTN
static std::pair<uint64_t, uint64_t> GenerateRNGState(
    const GPUContext& ctx,
    const paddle::optional<DenseTensor>& fixed_seed_offset,
    const std::string& rng_name,
    const int64_t batch_size,
    const int64_t num_heads) {
  if (fixed_seed_offset.get_ptr()) {
    const int64_t* fixed_seed_offset_data =
        fixed_seed_offset.get_ptr()->data<int64_t>();
    uint64_t seed = static_cast<uint64_t>(fixed_seed_offset_data[0]);
    uint64_t offset = static_cast<uint64_t>(fixed_seed_offset_data[1]);
    return std::make_pair(seed, offset);
  } else {
    uint64_t inc = batch_size * num_heads * 32;
    std::pair<uint64_t, uint64_t> seed_offset_pair;
    if (rng_name != "") {
      auto gen = phi::GetRandomSeedGenerator(rng_name);
      seed_offset_pair = gen->IncrementOffset(inc);
    } else {
      auto* gen = ctx.GetGenerator();
      seed_offset_pair = gen->IncrementOffset(inc);
    }
    return seed_offset_pair;
  }
}

template <typename T>
mcflashattnDataType_t McFLashAttnTypeTraits(T& _tensor) {
  phi::DataType dtype = _tensor.dtype();
  switch (dtype) {
    case phi::DataType::FLOAT16:
      return MCFLASHATTN_DATATYPE_FP16;
    case phi::DataType::BFLOAT16:
      return MCFLASHATTN_DATATYPE_BF16;
    case phi::DataType::INT8:
      return MCFLASHATTN_DATATYPE_INT8;
    case phi::DataType::INT32:
      return MCFLASHATTN_DATATYPE_INT32;
    case phi::DataType::INT64:
      return MCFLASHATTN_DATATYPE_INT64;
    case phi::DataType::FLOAT32:
      return MCFLASHATTN_DATATYPE_FP32;
    case phi::DataType::FLOAT64:
      return MCFLASHATTN_DATATYPE_FP64;
    default:
      return MCFLASHATTN_DATATYPE_NONE;
  }
}

template <typename T>
Tensor_t DenseTensorToMcFLashAttnTensor(T& _tensor) {
  PADDLE_ENFORCE_EQ(
      _tensor.meta().is_contiguous(),
      true,
      phi::errors::InvalidArgument("McFLashAttnTensor must be contiguous."));
  DDim tensor_dim = _tensor.dims();
  mcflashattnDataType_t _dtype = McFLashAttnTypeTraits(_tensor);
  int dim_num = tensor_dim.size();
  void* data_ptr = const_cast<void*>(_tensor.data());
  PADDLE_ENFORCE_NE(
      data_ptr,
      nullptr,
      phi::errors::InvalidArgument("McFLashAttnTensor must not be nullptr."));
  switch (dim_num) {
    case 1:
      return phi::dynload::make_contiguous_tensor1d(
          data_ptr, _dtype, tensor_dim[0]);
    case 2:
      return phi::dynload::make_contiguous_tensor2d(
          data_ptr, _dtype, tensor_dim[0], tensor_dim[1]);
    case 3:
      return phi::dynload::make_contiguous_tensor3d(
          data_ptr, _dtype, tensor_dim[0], tensor_dim[1], tensor_dim[2]);
    case 4:
      return phi::dynload::make_contiguous_tensor4d(data_ptr,
                                                    _dtype,
                                                    tensor_dim[0],
                                                    tensor_dim[1],
                                                    tensor_dim[2],
                                                    tensor_dim[3]);
    default:
      PADDLE_THROW(
          "McFLashAttnTensors must have dimensions more than 0 and less than "
          "5.");
      return nullptr;
  }
}

struct FlashAttnParamsBase {
  int64_t num_heads;
  int64_t num_heads_k;
  int64_t head_size;
  int64_t batch_size;
  int64_t seqlen_q;
  int64_t seqlen_k;

  Tensor_t q;
  Tensor_t k;
  Tensor_t v;
  Tensor_t out;
  Tensor_t alibi_slopes =
      nullptr;  // alibi now use atten_mask rather than alibi_slops

  float p_dropout;
  float softmax_scale;
  bool is_causal;
  int window_size_left = -1;  // do not support
  int window_size_right = -1;
  cudaStream_t stream;
  mcflashattnExtendParameter_t extend_parameter = nullptr;

  int64_t* seed_offset_data;
  FlashAttnParamsBase(const GPUContext& ctx,
                      const DenseTensor& _q,
                      const DenseTensor& _k,
                      const DenseTensor& _v,
                      const DenseTensor& _out,
                      const bool _is_test,
                      const float _p_dropout,
                      const bool _is_causal)
      : is_causal(_is_causal), stream(ctx.stream()) {
    // q, k, v [batch_size, seq_len, num_heads, head_dim]
    const auto& dims = _q.dims();
    PADDLE_ENFORCE_EQ(dims.size(),
                      4,
                      phi::errors::InvalidArgument(
                          "flash_attn receive input with dim "
                          "[batch_size, seq_len, num_heads, head_dim]"));

    batch_size = dims[0];
    seqlen_q = dims[1];
    num_heads = dims[2];
    head_size = dims[3];
    seqlen_k = _k.dims()[1];
    num_heads_k = _k.dims()[2];
    softmax_scale = 1.0f / std::sqrt(head_size);
    p_dropout = _is_test ? 0.0f : _p_dropout;
    q = DenseTensorToMcFLashAttnTensor(_q);
    k = DenseTensorToMcFLashAttnTensor(_k);
    v = DenseTensorToMcFLashAttnTensor(_v);
    out = DenseTensorToMcFLashAttnTensor(_out);
  }
  ~FlashAttnParamsBase() {
    phi::dynload::release_tensor(
        q);  // won't release tensor memory, only release info of tensor_t
    phi::dynload::release_tensor(k);
    phi::dynload::release_tensor(v);
    phi::dynload::release_tensor(out);
    phi::dynload::release_tensor(alibi_slopes);
    // phi::dynload::release_extend_para(extend_parameter);
  }
};
template <typename T>
struct FlashAttnParamsFwd : public FlashAttnParamsBase {
  Tensor_t attn_mask;
  Tensor_t p;  // return softmax
  Tensor_t rng_state;
  Tensor_t softmax_lse;

  FlashAttnParamsFwd(const GPUContext& ctx,
                     const paddle::optional<DenseTensor>& _attn_mask,
                     bool _return_softmax,
                     DenseTensor& _softmax,
                     const DenseTensor& _q,
                     const DenseTensor& _k,
                     const DenseTensor& _v,
                     DenseTensor& _out,
                     DenseTensor& _softmax_lse,
                     bool _is_test,
                     float _p_dropout,
                     bool _is_causal,
                     const paddle::optional<DenseTensor>& _fixed_seed_offset,
                     DenseTensor& _seed_offset,
                     const std::string& _rng_name)
      : FlashAttnParamsBase(
            ctx, _q, _k, _v, _out, _is_test, _p_dropout, _is_causal) {
    if (_attn_mask.get_ptr()) {
      PADDLE_ENFORCE_NE(_is_causal,
                        true,
                        phi::errors::InvalidArgument(
                            "When attn_mask is set, causal can not be true."));

      PADDLE_ENFORCE_EQ(
          _attn_mask->dtype(),
          _q.dtype(),
          phi::errors::InvalidArgument(
              "attn_mask is expected to have the same data type with q."));
      attn_mask = DenseTensorToMcFLashAttnTensor(_attn_mask.get());
    } else {
      attn_mask = nullptr;
    }
    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    int head_size_rounded = round_multiple(head_size, 32);
    int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    std::vector<int64_t> softmax_lse_dims = {
        batch_size, num_heads, seqlen_q_rounded};
    _softmax_lse.Resize(phi::make_ddim(softmax_lse_dims));
    ctx.template Alloc<float>(&_softmax_lse);
    softmax_lse = DenseTensorToMcFLashAttnTensor(_softmax_lse);
    if (_return_softmax) {
      PADDLE_ENFORCE_EQ(
          _p_dropout > 0.0f,
          true,
          phi::errors::InvalidArgument(
              "return_softmax is only supported when dropout > 0.0"));

      _softmax.Resize(
          {batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded});
      ctx.template Alloc<T>(&_softmax);
      p = DenseTensorToMcFLashAttnTensor(_softmax);
    }
    else{
      p = nullptr;
    }

    _seed_offset.Resize({2});
    seed_offset_data = ctx.template HostAlloc<int64_t>(&_seed_offset);
    std::tie(seed_offset_data[0], seed_offset_data[1]) = GenerateRNGState(
        ctx, _fixed_seed_offset, _rng_name, batch_size, num_heads);
    rng_state = phi::dynload::make_contiguous_tensor1d(
        seed_offset_data, MCFLASHATTN_DATATYPE_INT64, 2);
  }

  ~FlashAttnParamsFwd() {
    phi::dynload::release_tensor(attn_mask);
    phi::dynload::release_tensor(p);  // return softmax
    phi::dynload::release_tensor(rng_state);
    phi::dynload::release_tensor(softmax_lse);
  }
};

struct FlashAttnParamsBwd : public FlashAttnParamsBase {
  Tensor_t attn_mask;
  Tensor_t softmax_d;
  Tensor_t rng_state;
  Tensor_t dout;
  Tensor_t out;
  Tensor_t dq;
  Tensor_t dk;
  Tensor_t dv;
  Tensor_t dq_accum;
  Tensor_t softmax_lse;
  bool deterministic;
  FlashAttnParamsBwd(const GPUContext& ctx,
                     const paddle::optional<DenseTensor>& _attn_mask,
                     const DenseTensor& _dout,
                     const DenseTensor& _q,
                     const DenseTensor& _k,
                     const DenseTensor& _v,
                     const DenseTensor& _out,
                     const DenseTensor& _softmax_lse,
                     const DenseTensor& _seed_offset_data,
                     DenseTensor& _dq,
                     DenseTensor& _dk,
                     DenseTensor& _dv,
                     float _p_dropout,
                     bool _is_causal)
      : FlashAttnParamsBase(
            ctx, _q, _k, _v, _out, false, _p_dropout, _is_causal) {
    if (_attn_mask.get_ptr()) {
      PADDLE_ENFORCE_NE(_is_causal,
                        true,
                        phi::errors::InvalidArgument(
                            "When attn_mask is set, causal can not be true."));

      PADDLE_ENFORCE_EQ(
          _attn_mask->dtype(),
          _q.dtype(),
          phi::errors::InvalidArgument(
              "attn_mask is expected to have the same data type with q."));
      attn_mask = DenseTensorToMcFLashAttnTensor(_attn_mask.get());
    } else {
      attn_mask = nullptr;
    }
    dout = DenseTensorToMcFLashAttnTensor(_dout);
    dq = DenseTensorToMcFLashAttnTensor(_dq);
    dk = DenseTensorToMcFLashAttnTensor(_dk);
    dv = DenseTensorToMcFLashAttnTensor(_dv);
    out = DenseTensorToMcFLashAttnTensor(_out);
    softmax_lse = DenseTensorToMcFLashAttnTensor(_softmax_lse);
    rng_state = DenseTensorToMcFLashAttnTensor(_seed_offset_data);
    deterministic = FLAGS_cudnn_deterministic;
    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    int head_size_rounded = round_multiple(head_size, 32);
    int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    std::vector<int64_t> softmax_lse_dims = {
        batch_size, num_heads, seqlen_q_rounded};
    // gradient of softmax_lse
    _softmax_d = Empty<float>(ctx, softmax_lse_dims);
    softmax_d = DenseTensorToMcFLashAttnTensor(_softmax_d);

    // an internal gradient of q, which will be further accumulated.
    _dq_accum = Empty<float>(
        ctx, {batch_size, num_heads, seqlen_q_rounded, head_size_rounded});
    dq_accum = DenseTensorToMcFLashAttnTensor(_dq_accum);
  }

  ~FlashAttnParamsBwd() {
    phi::dynload::release_tensor(attn_mask);
    phi::dynload::release_tensor(softmax_d);
    phi::dynload::release_tensor(rng_state);
    phi::dynload::release_tensor(dout);
    phi::dynload::release_tensor(out);
    phi::dynload::release_tensor(dq);
    phi::dynload::release_tensor(dk);
    phi::dynload::release_tensor(dv);
    phi::dynload::release_tensor(dq_accum);
    phi::dynload::release_tensor(softmax_lse);
  }

 private:
  DenseTensor _softmax_d;
  DenseTensor _dq_accum;
};

static void CheckFlashAttnStatus(const mcflashattnStatus_t status) {
  PADDLE_ENFORCE_EQ(
      status,
      MCFLASHATTN_STATUS_SUCCESS,
      phi::errors::External("Error in MC-Flash-Attention, error code is %d",
                            status));
}
#endif

static void RaiseNotSupportedError() {
  PADDLE_THROW(
      phi::errors::Unimplemented("FlashAttention is unsupported, please check "
                                 "the GPU compability and CUDA Version."));
}

}  // namespace phi
