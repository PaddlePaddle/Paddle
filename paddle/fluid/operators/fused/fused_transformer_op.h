/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
Indicesou may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <string>

namespace paddle {
namespace operators {

template <typename T>
class FusedMHA {
  FusedMHA(int, int, int, int, float, bool, uint64_t, uint64_t*, uint64_t*);
  ~FusedMHA();

  void ComputeForward(T*, T*);
  void ComputeBackward(const T*, T*, T*);

 private:
  int batch_size;
  int max_seq_len;
  int dim_embed;

  int num_head;
  int head_size;

  float dropout;

  bool is_test;
  uint64_t seed;

  int32_t seqlen;
  int32_t* cu_seqlen;
};

template <typename T>
class FusedAttention {
 public:
  FusedAttention(int, int, int, int, float, float, bool);
  ~FusedAttention();

  void ComputeForward(T*, T*);
  void ComputeBackward();

 private:
  FusedMHA<T>* fmha;  // fused multihead attention

  int batch_size;
  int max_seq_len;
  int dim_embed;

  int num_head;
  int head_size;

  float dropout;
  T attn_dropout;

  bool normalize_pre_or_post;

  // weights and bias used in attention
  T* fattn_qkv_w;
  T* fattn_qkv_b;
  T* fattn_o_w;
  T* fattn_o_b;
  T* fattn_n_w;
  T* fattn_n_b;
  T* fattn_norm_w;
  T* fattn_norm_b;

  T* fattn_grad_qkv_w;
  T* fattn_grad_qkv_b;
  T* fattn_grad_o_w;
  T* fattn_grad_o_b;
  T* fattn_grad_n_w;
  T* fattn_grad_n_b;
  T* fattn_grad_norm_w;
  T* fattn_grad_norm_b;
};

template <typename T>
class FusedFFN {
  FusedFFN(int, int, int, int, float, std::string, bool);
  ~FusedFFN();

  void ComputeForward(T*, T*);
  void ComputeBackward();

 private:
  int batch_size;
  int max_seq_len;
  int dim_embed;
  int dim_feedforward;

  float attn_dropout;
  float act_dropout;

  bool normalize_pre_or_post;

  std::string act_method;

  // weights and bias used in ffn
  T* fffn_inter_w;
  T* fffn_inter_b;
  T* fffn_output_w;
  T* fffn_output_b;

  T* fffn_grad_inter_w;
  T* fffn_grad_inter_b;
  T* fffn_grad_output_w;
  T* fffn_grad_output_b;
};

template <typename T>
class FusedTransformerEncoderLayer {
 public:
  FusedTransformerEncoderLayer(int, int, int, int, int, float, float, float,
                               std::string, bool);
  ~FusedTransformerEncoderLayer();

  void ComputeForward(T* src, T* output);
  void ComputeBackward();

 private:
  FusedAttention<T>* fused_attn;
  FusedFFN<T>* fused_ffn;

  int batch_size;
  int max_seq_len;
  int dim_embed;
  int dim_feedforward;

  int num_head;
  int head_size;

  float dropout;
  float attn_dropout;
  float act_dropout;

  bool normalize_pre_or_post;

  std::string act_method;
};
}
}
