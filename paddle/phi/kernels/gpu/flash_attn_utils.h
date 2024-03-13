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

#ifdef PADDLE_WITH_FLASHATTN
#include "paddle/phi/backends/dynload/flashattn.h"
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
        phi::errors::InvalidArgument(
            "The number of dimenstions of attn_mask is expected to be greater "
            "or equal to 4, but recieved %d. The shape of attn_mask is {%s}",
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
    const DenseTensor* attn_mask_start_row_indices,
    int64_t attn_mask_start_row,
    int max_seqlen_q) {
  std::vector<int64_t> mask_dim_3d;
  if (attn_mask_start_row_indices) {
    const auto& dtype = attn_mask_start_row_indices->dtype();
    const auto& origin_dims = attn_mask_start_row_indices->dims();
    auto rank = origin_dims.size();
    PADDLE_ENFORCE_EQ(dtype,
                      DataType::INT32,
                      phi::errors::InvalidArgument(
                          "dtype of attn_mask_start_row_indices must be "
                          "int32, but recieved %d",
                          dtype));
    PADDLE_ENFORCE_GE(
        rank,
        3,
        phi::errors::InvalidArgument(
            "The number of dimenstions of attn_mask_start_row_indices is "
            "expected to be greater or "
            "equal to 3, but recieved %d. The shape of "
            "attn_mask_start_row_indices is [%s]",
            rank,
            origin_dims));
    PADDLE_ENFORCE_EQ(origin_dims[rank - 1],
                      max_seqlen_q,
                      phi::errors::InvalidArgument(
                          "The sparse_mask_dims[%d] of "
                          "attn_mask_start_row_indices is expected to be "
                          "equal to %d, but recieved %d.",
                          rank - 1,
                          max_seqlen_q,
                          origin_dims[2]));
    PADDLE_ENFORCE_GE(attn_mask_start_row,
                      0,
                      phi::errors::InvalidArgument(
                          "attn_mask_start_row should be greater or equal than "
                          "0 when using attn_mask_start_row_indices, "
                          "but recieved %d.",
                          attn_mask_start_row));

    int64_t first_dim = 1;
    for (int i = 0; i < rank - 2; i++) {
      first_dim *= origin_dims[i];
    }
    mask_dim_3d = {first_dim, origin_dims[rank - 2], origin_dims[rank - 1]};
  }

  return mask_dim_3d;
}

template <typename T>
struct FlashAttnFwdParamsV2 {
  int batch_size;
  // for padded kernel, max_seqlen_q and seqlen_q is the same.
  int64_t max_seqlen_q;
  // for padded kernel, max_seqlen_k and seqlen_k is the same.
  int64_t max_seqlen_k;
  int seqlen_q_rounded;
  int seqlen_k_rounded;
  int num_heads;
  int num_heads_k;
  int head_size;
  int head_size_rounded;
  float dropout;
  float scale;
  bool causal;
  bool return_softmax;
  bool is_bf16;
  uint64_t seed;
  uint64_t offset;
  std::vector<int64_t> mask_dims;
  DenseTensor rng_state;
  const DenseTensor* attn_mask_tensor;
  DenseTensor* softmax;
  DenseTensor* softmax_lse;
  DenseTensor* seed_offset;
  const DenseTensor* attn_mask_start_row_indices_tensor;
  std::vector<int64_t> attn_mask_start_row_indices_dims;
  int attn_mask_start_row;

  FlashAttnFwdParamsV2(
      const GPUContext& ctx,
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
      const int _attn_mask_start_row,
      const paddle::optional<DenseTensor>& fixed_seed_offset,
      const paddle::optional<DenseTensor>& attn_mask,
      const paddle::optional<DenseTensor>& attn_mask_start_row_indices,
      DenseTensor* _softmax,
      DenseTensor* _softmax_lse,
      DenseTensor* _seed_offset)
      : batch_size(_batch_size),
        max_seqlen_q(_max_seqlen_q),
        max_seqlen_k(_max_seqlen_k),
        num_heads(_num_heads),
        num_heads_k(_num_heads_k),
        head_size(_head_size),
        scale(_scale),
        dropout(_dropout),
        causal(_causal),
        return_softmax(_return_softmax),
        attn_mask_start_row(_attn_mask_start_row),
        softmax(_softmax),
        softmax_lse(_softmax_lse),
        seed_offset(_seed_offset),
        attn_mask_tensor(attn_mask.get_ptr()),
        attn_mask_start_row_indices_tensor(
            attn_mask_start_row_indices.get_ptr()) {
    dropout = is_test ? 0.0f : _dropout;
    is_bf16 = q_dtype == DataType::BFLOAT16;

    // (umiswing): There is no suitable kernel for uint64_t, allocate in int64_t
    // with the same size.
    rng_state = Empty<int64_t>(ctx, {2});

    auto seed_offset_pair = GenerateRNGState(
        ctx, fixed_seed_offset, rng_name, batch_size, num_heads);
    seed = seed_offset_pair.first;
    offset = seed_offset_pair.second;

    seed_offset->Resize({2});
    int64_t* seed_offset_data = ctx.template HostAlloc<int64_t>(seed_offset);
    seed_offset_data[0] = static_cast<int64_t>(seed);
    seed_offset_data[1] = static_cast<int64_t>(offset);

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    head_size_rounded = round_multiple(head_size, 32);
    seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    softmax_lse->Resize({batch_size, num_heads, max_seqlen_q});
    ctx.template Alloc<float>(softmax_lse);

    if (return_softmax) {
      PADDLE_ENFORCE_EQ(
          dropout > 0.0f,
          true,
          phi::errors::InvalidArgument(
              "return_softmax is only supported when dropout > 0.0"));

      softmax->Resize(
          {batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded});
      ctx.template Alloc<T>(softmax);
    }

    mask_dims = GetAttnMaskDims(attn_mask_tensor);
    if (attn_mask) {
      PADDLE_ENFORCE_EQ(
          attn_mask->dtype(),
          q_dtype,
          phi::errors::InvalidArgument(
              "attn_mask is expected to have the same data type with q."));
    }

    attn_mask_start_row_indices_dims = GetAttnSparseMaskDims(
        attn_mask_start_row_indices_tensor, attn_mask_start_row, max_seqlen_q);

    PADDLE_ENFORCE_NE(attn_mask && attn_mask_start_row_indices,
                      true,
                      phi::errors::InvalidArgument(
                          "attn_mask and attn_mask_start_row_indices cannot be "
                          "set at same time."));
  }
};

struct FlashAttnBwdParamsV2 {
  int batch_size;
  int64_t max_seqlen_q;
  int64_t max_seqlen_k;
  int seqlen_q_rounded;
  int seqlen_k_rounded;
  int num_heads;
  int num_heads_k;
  int head_size;
  int head_size_rounded;
  float dropout;
  float scale;
  bool causal;
  bool is_bf16;
  uint64_t seed;
  uint64_t offset;
  std::vector<int64_t> mask_dims;
  DenseTensor softmax_d;
  DenseTensor dq_accum;
  DenseTensor rng_state;
  const DenseTensor* attn_mask_tensor;
  const DenseTensor* attn_mask_start_row_indices_tensor;
  std::vector<int64_t> attn_mask_start_row_indices_dims;
  int attn_mask_start_row;

  FlashAttnBwdParamsV2(
      const GPUContext& ctx,
      const int _batch_size,
      const int64_t _max_seqlen_q,
      const int64_t _max_seqlen_k,
      const int _num_heads,
      const int _num_heads_k,
      const int _head_size,
      const float _dropout,
      const float _scale,
      const bool _causal,
      const int _attn_mask_start_row,
      const DataType q_dtype,
      const paddle::optional<DenseTensor>& attn_mask,
      const paddle::optional<DenseTensor>& attn_mask_start_row_indices,
      const int64_t* seed_offset_data)
      : batch_size(_batch_size),
        max_seqlen_q(_max_seqlen_q),
        max_seqlen_k(_max_seqlen_k),
        num_heads(_num_heads),
        num_heads_k(_num_heads_k),
        head_size(_head_size),
        dropout(_dropout),
        scale(_scale),
        causal(_causal),
        attn_mask_start_row(_attn_mask_start_row),
        attn_mask_tensor(attn_mask.get_ptr()),
        attn_mask_start_row_indices_tensor(
            attn_mask_start_row_indices.get_ptr()) {
    is_bf16 = q_dtype == DataType::BFLOAT16;
    seed = static_cast<uint64_t>(seed_offset_data[0]);
    offset = static_cast<uint64_t>(seed_offset_data[1]);

    // (umiswing): There is no suitable kernel for uint64_t, allocate in int64_t
    // with the same size.
    rng_state = Empty<int64_t>(ctx, {2});

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };

    head_size_rounded = round_multiple(head_size, 32);
    seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    softmax_d = Empty<float>(ctx, {batch_size, num_heads, seqlen_q_rounded});
    dq_accum = Empty<float>(
        ctx, {batch_size, num_heads, seqlen_q_rounded, head_size_rounded});

    mask_dims = GetAttnMaskDims(attn_mask_tensor);
    if (attn_mask) {
      PADDLE_ENFORCE_EQ(
          attn_mask->dtype(),
          q_dtype,
          phi::errors::InvalidArgument(
              "attn_mask is expected to have the same data type with q."));
    }

    attn_mask_start_row_indices_dims = GetAttnSparseMaskDims(
        attn_mask_start_row_indices_tensor, attn_mask_start_row, max_seqlen_q);

    PADDLE_ENFORCE_NE(attn_mask && attn_mask_start_row_indices,
                      true,
                      phi::errors::InvalidArgument(
                          "attn_mask and attn_mask_start_row_indices cannot be "
                          "set at same time."));
  }
};

static void CheckFlashAttnStatus(const bool status) {
  PADDLE_ENFORCE_EQ(status,
                    true,
                    phi::errors::External(
                        "Error in Flash-Attention, detail information is: %s",
                        phi::dynload::flash_attn_error()));
}

#endif

static void RaiseNotSupportedError() {
  PADDLE_THROW(
      phi::errors::Unimplemented("FlashAttention is unsupported, please check "
                                 "the GPU compability and CUDA Version."));
}

}  // namespace phi
