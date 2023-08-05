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

namespace phi {

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
  DenseTensor* softmax;
  DenseTensor* softmax_lse;
  DenseTensor* seed_offset;

  FlashAttnFwdParamsV2(const GPUContext& ctx,
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
                       DenseTensor* _softmax,
                       DenseTensor* _softmax_lse,
                       DenseTensor* _seed_offset,
                       const DenseTensor* const fixed_seed_offset_ptr)
      : batch_size(_batch_size),
        max_seqlen_q(_max_seqlen_q),
        max_seqlen_k(_max_seqlen_k),
        num_heads(_num_heads),
        num_heads_k(_num_heads),
        head_size(_head_size),
        scale(_scale),
        dropout(_dropout),
        causal(_causal),
        return_softmax(_return_softmax),
        softmax(_softmax),
        softmax_lse(_softmax_lse),
        seed_offset(_seed_offset) {
    dropout = is_test ? 0.0f : _dropout;
    is_bf16 = q_dtype == DataType::BFLOAT16;

    if (fixed_seed_offset_ptr) {
      const int64_t* fixed_seed_offset_data =
          fixed_seed_offset_ptr->data<int64_t>();
      seed = static_cast<uint64_t>(fixed_seed_offset_data[0]);
      offset = static_cast<uint64_t>(fixed_seed_offset_data[1]);
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
      seed = seed_offset_pair.first;
      offset = seed_offset_pair.second;
    }

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
      softmax->Resize(
          {batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded});
      ctx.template Alloc<T>(softmax);
    }
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
  DenseTensor softmax_d;
  DenseTensor dq_accum;

  FlashAttnBwdParamsV2(const GPUContext& ctx,
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
                       const int64_t* seed_offset_data)
      : batch_size(_batch_size),
        max_seqlen_q(_max_seqlen_q),
        max_seqlen_k(_max_seqlen_k),
        num_heads(_num_heads),
        num_heads_k(_num_heads_k),
        head_size(_head_size),
        dropout(_dropout),
        scale(_scale),
        causal(_causal) {
    is_bf16 = q_dtype == DataType::BFLOAT16;
    seed = static_cast<uint64_t>(seed_offset_data[0]);
    offset = static_cast<uint64_t>(seed_offset_data[1]);
    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };

    head_size_rounded = round_multiple(head_size, 32);
    seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    softmax_d = Empty<float>(ctx, {batch_size, num_heads, seqlen_q_rounded});
    dq_accum = Empty<float>(
        ctx, {batch_size, num_heads, seqlen_q_rounded, head_size_rounded});
  }
};
}  // namespace phi
