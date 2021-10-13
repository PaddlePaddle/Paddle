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

#include "paddle/fluid/operators/fused/fused_transformer_op.h"
#include <string>

namespace paddle {
namespace operators {

// constructor and init
template <typename T>
FusedTransformerEncoderLayer<T>::FusedTransformerEncoderLayer(
    int batch_size_, int max_seq_len_, int dim_embed_, int dim_feedforward_,
    int num_head_, float dropout_, float act_dropout_, float attn_dropout_,
    std::string act_method_, bool normalize_pre_or_post_) {
  // configurations
  batch_size = batch_size_;
  max_seq_len = max_seq_len_;
  dim_embed = dim_embed_;
  dim_feedforward = dim_feedforward_;
  num_head = num_head_;
  head_size = dim_embed_ / num_head;

  dropout = dropout_;
  act_dropout = act_dropout_;
  attn_dropout = attn_dropout_;

  act_method = act_method_;
  normalize_pre_or_post = normalize_pre_or_post_;

  // init attn
  fused_attn =
      new FusedAttention<T>(batch_size, max_seq_len, dim_embed, num_head,
                            dropout, attn_dropout, normalize_pre_or_post);

  // init ffn
  fused_ffn =
      new FusedFFN<T>(batch_size, max_seq_len, dim_embed, dim_feedforward_,
                      act_dropout, act_method, normalize_pre_or_post);
}

// deconstructor
template <typename T>
FusedTransformerEncoderLayer<T>::~FusedTransformerEncoderLayer() {
  delete fused_attn;
  delete fused_ffn;
}

// compute forward
template <typename T>
void FusedTransformerEncoderLayer<T>::ComputeForward(T* src, T* output) {
  T* output_attn;  // todo

  fused_attn->ComputeForward(src, output_attn);
  fused_ffn->ComputeForward(output_attn, output);
}

// compute backward
template <typename T>
void FusedTransformerEncoderLayer<T>::ComputeBackward() {}

// constructor and init
template <typename T>
FusedAttention<T>::FusedAttention(int batch_size_, int max_seq_len_,
                                  int dim_embed_, int num_head_, float dropout_,
                                  float attn_dropout_,
                                  bool normalize_pre_or_post_) {
  // configurations
  batch_size = batch_size_;
  max_seq_len = max_seq_len_;
  dim_embed = dim_embed_;
  num_head = num_head_;
  head_size = dim_embed_ / num_head;

  dropout = dropout_;
  attn_dropout = attn_dropout_;

  normalize_pre_or_post = normalize_pre_or_post_;

  // init fmha
  fmha = new FusedMHA<T>();
}

// compute forward
template <typename T>
void FusedAttention<T>::ComputeForward(T* src, T* output) {}

template <typename T>
FusedAttention<T>::~FusedAttention() {
  delete fmha;
}

// compute backward
template <typename T>
void FusedAttention<T>::ComputeBackward() {}

// constructor and init
template <typename T>
FusedFFN<T>::FusedFFN(int batch_size_, int max_seq_len_, int dim_embed_,
                      int dim_feedforward_, float act_dropout_,
                      std::string act_method_, bool normalize_pre_or_post_) {
  batch_size = batch_size_;
  max_seq_len = max_seq_len_;
  dim_embed = dim_embed_;
  dim_feedforward = dim_feedforward_;
  act_dropout = act_dropout_;

  act_method = act_method_;
  normalize_pre_or_post = normalize_pre_or_post_;
}

template <typename T>
FusedFFN<T>::~FusedFFN() {}

// compute forward
template <typename T>
void FusedFFN<T>::ComputeForward(T* src, T* output) {}

// compute backward
template <typename T>
void FusedFFN<T>::ComputeBackward() {}

// init
template <typename T>
FusedMHA<T>::FusedMHA(int batch_size_, int max_seq_len_, int dim_embed_,
                      int num_head_, float dropout_, bool is_test_,
                      uint64_t seed_, uint64_t* seqlen_, uint64_t* cu_seqlen_) {
  batch_size = batch_size_;
  max_seq_len = max_seq_len_;
  dim_embed = dim_embed_;
  num_head = num_head_;
  head_size = dim_embed_ / num_head;

  dropout = dropout_;
  is_test = is_test_;
  seed = seed_;
  seqlen = seqlen_;
  cu_seqlen = cu_seqlen_;
}

// compute forward
template <typename T>
void FusedMHA<T>::ComputeForward(T* output, T* softmax_mask) {}

// compute backward
template <typename T>
void FusedMHA<T>::ComputeBackward(const T* grad_output, T* softmax_mask,
                                  T* grad_x) {}
}
}