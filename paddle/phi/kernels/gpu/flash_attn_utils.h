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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"

#ifdef PADDLE_WITH_FLASHATTN
#include "paddle/phi/backends/dynload/flashattn.h"
#endif

#ifdef PADDLE_WITH_FLASHATTN_V3
#include "paddle/phi/backends/dynload/flashattnv3.h"
#endif

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

static std::vector<int64_t> GetAttnMaskDims(const DenseTensor* attn_mask) {
  std::vector<int64_t> mask_dim_4d;
  if (attn_mask) {
    const auto& origin_dims = attn_mask->dims();
    auto rank = origin_dims.size();
    PADDLE_ENFORCE_GE(
        rank,
        4,
        common::errors::InvalidArgument(
            "The number of dimensions of attn_mask is expected to be greater "
            "or equal to 4, but received %d. The shape of attn_mask is {%s}",
            rank,
            origin_dims));

    int64_t first_dim = 1;
    for (int i = 0; i < rank - 3; i++) {
      first_dim *= origin_dims[i];
    }
    mask_dim_4d = {first_dim,
                   origin_dims[rank - 3],
                   origin_dims[rank - 2],
                   origin_dims[rank - 1]};
  }
  return mask_dim_4d;
}

static std::vector<int64_t> GetAttnSparseMaskDims(
    const DenseTensor* startend_row_indices, int max_seqlen_q) {
  std::vector<int64_t> mask_dim_4d;
  if (startend_row_indices) {
    const auto& dtype = startend_row_indices->dtype();
    const auto& origin_dims = startend_row_indices->dims();
    auto rank = origin_dims.size();
    PADDLE_ENFORCE_EQ(
        dtype,
        DataType::INT32,
        common::errors::InvalidArgument("dtype of startend_row_indices must be "
                                        "int32, but received %d",
                                        dtype));
    PADDLE_ENFORCE_GE(
        rank,
        4,
        common::errors::InvalidArgument(
            "The number of dimensions of startend_row_indices is expected to "
            "be greater or equal to 4, but received %d. The shape of "
            "startend_row_indices is [%s]",
            rank,
            origin_dims));
    PADDLE_ENFORCE_EQ(origin_dims[rank - 2],
                      max_seqlen_q,
                      common::errors::InvalidArgument(
                          "The sparse_mask_dims[%d] of "
                          "attn_mask_start_row_indices is expected to be "
                          "equal to %d, but received %d.",
                          rank - 2,
                          max_seqlen_q,
                          origin_dims[2]));

    int64_t first_dim = 1;
    for (int i = 0; i < rank - 3; i++) {
      first_dim *= origin_dims[i];
    }
    mask_dim_4d = {first_dim,
                   origin_dims[rank - 3],
                   origin_dims[rank - 2],
                   origin_dims[rank - 1]};
  }

  return mask_dim_4d;
}

struct FlashAttnParamsBase {
  int version;
  bool is_fwd;

  int kBlockM;
  int batch_size;
  // for padded kernel, max_seqlen_q and seqlen_q is the same.
  int64_t max_seqlen_q;
  // for padded kernel, max_seqlen_k and seqlen_k is the same.
  int64_t max_seqlen_k;
  int num_heads;
  int num_heads_k;
  int head_size;

  int seqlen_q_rounded;
  int seqlen_k_rounded;
  int head_size_rounded;

  bool is_bf16;
  bool is_fp8;
  float softmax_scale;
  std::vector<int64_t> softmax_lse_dims;

  bool causal;
  std::vector<int64_t> mask_dims;
  const DenseTensor* attn_mask_tensor;

  const DenseTensor* startend_row_indices;
  std::vector<int64_t> startend_row_indices_dims;

  FlashAttnParamsBase(const int _version,
                      const int _is_fwd,
                      const int _batch_size,
                      const int64_t _max_seqlen_q,
                      const int64_t _max_seqlen_k,
                      const int _num_heads,
                      const int _num_heads_k,
                      const int _head_size,
                      const float _scale,
                      const bool _causal,
                      const DataType q_dtype,
                      const paddle::optional<DenseTensor>& attn_mask,
                      const paddle::optional<DenseTensor>& startend_row_indices)
      : version(_version),
        is_fwd(_is_fwd),
        batch_size(_batch_size),
        max_seqlen_q(_max_seqlen_q),
        max_seqlen_k(_max_seqlen_k),
        num_heads(_num_heads),
        num_heads_k(_num_heads_k),
        head_size(_head_size),
        softmax_scale(_scale),
        causal(_causal),
        attn_mask_tensor(attn_mask.get_ptr()),
        startend_row_indices(startend_row_indices.get_ptr()) {
    is_bf16 = q_dtype == DataType::BFLOAT16;

    // TODO(GuoxiaWang): check q, k, v dtype

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    // FLAGS_flash_attn_version
    if (_version == 3 && !_is_fwd) {
      kBlockM = head_size <= 64 ? 128 : (head_size < 256 ? 64 : 32);
      head_size_rounded = head_size <= 64 ? 64 : round_multiple(head_size, 32);
    } else {
      kBlockM = 128;
      head_size_rounded = round_multiple(head_size, 32);
    }

    seqlen_q_rounded = round_multiple(max_seqlen_q, kBlockM);
    seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    softmax_lse_dims = {batch_size, num_heads, seqlen_q_rounded};

    if (attn_mask_tensor) {
      PADDLE_ENFORCE_EQ(
          attn_mask->dtype(),
          q_dtype,
          common::errors::InvalidArgument(
              "attn_mask is expected to have the same data type with q."));

      mask_dims = GetAttnMaskDims(attn_mask_tensor);
    }

    startend_row_indices_dims = GetAttnSparseMaskDims(
        startend_row_indices ? startend_row_indices.get_ptr() : nullptr,
        max_seqlen_q);

    if (startend_row_indices.is_initialized()) {
      PADDLE_ENFORCE_EQ(
          attn_mask_tensor,
          nullptr,
          common::errors::InvalidArgument(
              "attn_mask and attn_mask_start_row_indices cannot be "
              "set at same time."));
    }
  }
};

template <typename T>
struct FlashAttnFwdParamsV2 : public FlashAttnParamsBase {
  float dropout;
  bool return_softmax;
  uint64_t seed;
  uint64_t offset;
  DenseTensor rng_state;
  DenseTensor* softmax;
  DenseTensor* softmax_lse;
  DenseTensor* seed_offset;
  DenseTensor tile_count_semaphore;

  FlashAttnFwdParamsV2(
      const GPUContext& ctx,
      const int _version,
      const int _batch_size,
      const int64_t _max_seqlen_q,
      const int64_t _max_seqlen_k,
      const int _num_heads,
      const int _num_heads_k,
      const int _head_size,
      const float _dropout,
      const float _scale,
      const bool _causal,
      const bool _return_softmax,
      const DataType q_dtype,
      const bool is_test,
      const std::string& rng_name,
      const paddle::optional<DenseTensor>& fixed_seed_offset,
      const paddle::optional<DenseTensor>& attn_mask,
      const paddle::optional<DenseTensor>& startend_row_indices,
      DenseTensor* _softmax,
      DenseTensor* _softmax_lse,
      DenseTensor* _seed_offset)
      : FlashAttnParamsBase(_version,
                            /*is_fwd=*/true,
                            _batch_size,
                            _max_seqlen_q,
                            _max_seqlen_k,
                            _num_heads,
                            _num_heads_k,
                            _head_size,
                            _scale,
                            _causal,
                            q_dtype,
                            attn_mask,
                            startend_row_indices),
        dropout(_dropout),
        return_softmax(_return_softmax),
        softmax(_softmax),
        softmax_lse(_softmax_lse),
        seed_offset(_seed_offset) {
    dropout = is_test ? 0.0f : _dropout;

    // (umiswing): There is no suitable kernel for uint64_t, allocate in int64_t
    // with the same size.
    rng_state = Empty<int64_t>(ctx, {2});

    if (_dropout > 0.0f) {
      auto seed_offset_pair = GenerateRNGState(
          ctx, fixed_seed_offset, rng_name, batch_size, num_heads);
      seed = seed_offset_pair.first;
      offset = seed_offset_pair.second;
    } else {
      seed = 0;
      offset = 0;
    }

    seed_offset->Resize({2});
    int64_t* seed_offset_data = ctx.template HostAlloc<int64_t>(seed_offset);
    seed_offset_data[0] = static_cast<int64_t>(seed);
    seed_offset_data[1] = static_cast<int64_t>(offset);

    softmax_lse->Resize(phi::make_ddim(softmax_lse_dims));
    ctx.template Alloc<float>(softmax_lse);

    if (_version == 3) {
      tile_count_semaphore = Full<int>(ctx, {1}, static_cast<int>(0));
    }

    if (return_softmax) {
      PADDLE_ENFORCE_EQ(
          dropout > 0.0f,
          true,
          common::errors::InvalidArgument(
              "return_softmax is only supported when dropout > 0.0"));

      softmax->Resize(
          {batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded});
      ctx.template Alloc<T>(softmax);
    }
  }
};

struct FlashAttnBwdParamsV2 : public FlashAttnParamsBase {
  float dropout;
  uint64_t seed;
  uint64_t offset;
  DenseTensor softmax_d;
  DenseTensor dq_accum;
  DenseTensor rng_state;

  DenseTensor softmax_lse_log2;
  DenseTensor dq_semaphore;

  FlashAttnBwdParamsV2(
      const GPUContext& ctx,
      const int _version,
      const int _batch_size,
      const int64_t _max_seqlen_q,
      const int64_t _max_seqlen_k,
      const int _num_heads,
      const int _num_heads_k,
      const int _head_size,
      const float _dropout,
      const float _scale,
      const bool _causal,
      const DataType q_dtype,
      const paddle::optional<DenseTensor>& attn_mask,
      const paddle::optional<DenseTensor>& startend_row_indices,
      const int64_t* seed_offset_data)
      : FlashAttnParamsBase(_version,
                            /*is_fwd=*/false,
                            _batch_size,
                            _max_seqlen_q,
                            _max_seqlen_k,
                            _num_heads,
                            _num_heads_k,
                            _head_size,
                            _scale,
                            _causal,
                            q_dtype,
                            attn_mask,
                            startend_row_indices),
        dropout(_dropout) {
    seed = static_cast<uint64_t>(seed_offset_data[0]);
    offset = static_cast<uint64_t>(seed_offset_data[1]);

    // (umiswing): There is no suitable kernel for uint64_t, allocate in int64_t
    // with the same size.
    rng_state = Empty<int64_t>(ctx, {2});

    // gradient of softmax_lse
    softmax_d = Empty<float>(ctx, softmax_lse_dims);

    if (_version == 3) {
      softmax_lse_log2 = Empty<float>(ctx, softmax_lse_dims);
      dq_semaphore = Empty<int>(
          ctx, {(max_seqlen_q + kBlockM - 1) / kBlockM, batch_size, num_heads});
    }

    // an internal gradient of q, which will be further accumulated.
    dq_accum = Empty<float>(
        ctx, {batch_size, num_heads, seqlen_q_rounded, head_size_rounded});
  }
};

static void CheckFlashAttnStatus(const bool status) {
  PADDLE_ENFORCE_EQ(status,
                    true,
                    common::errors::External(
                        "Error in Flash-Attention, detail information is: %s",
                        phi::dynload::flash_attn_error()));
}
#endif

static void RaiseNotSupportedError(int version = 2) {
  PADDLE_THROW(common::errors::Unimplemented(
      "FlashAttentio%d is unsupported, please check "
      "the GPU compatibility and CUDA Version.",
      version));
}

}  // namespace phi
