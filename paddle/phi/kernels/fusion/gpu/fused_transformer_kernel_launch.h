// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi::fusion {

struct FusedDropoutConfig {
  bool fix_seed;
  bool is_test;
  bool is_upscale_in_train;
  float probability;
  const DenseTensor* seed;
  int seed_value;
};

struct AttentionDropoutConfig {
  bool is_test;
  const std::string* implementation;
  float probability;
  bool is_upscale_in_train;
  bool fix_seed;
  int seed_value;
  const DenseTensor* seed;
};

template <typename T>
using FusedLayerNormParamType = typename dtype::MPTypeTrait<T>::Type;

template <typename T>
void LaunchLayerNormForward(const GPUContext& dev_ctx,
                            int rows,
                            int cols,
                            float epsilon,
                            const T* x,
                            const FusedLayerNormParamType<T>* scale,
                            const FusedLayerNormParamType<T>* bias,
                            T* out,
                            FusedLayerNormParamType<T>* mean,
                            FusedLayerNormParamType<T>* variance);

template <typename T>
void LaunchLayerNormBackward(const GPUContext& dev_ctx,
                             int rows,
                             int cols,
                             float epsilon,
                             const T* out_grad,
                             const T* x,
                             const FusedLayerNormParamType<T>* scale,
                             const FusedLayerNormParamType<T>* mean,
                             const FusedLayerNormParamType<T>* variance,
                             T* x_grad,
                             FusedLayerNormParamType<T>* scale_grad,
                             FusedLayerNormParamType<T>* bias_grad);

template <typename T>
void LaunchDropoutActBiasForward(const GPUContext& dev_ctx,
                                 int rows,
                                 int cols,
                                 const FusedDropoutConfig& dropout,
                                 const T* x,
                                 const T* bias,
                                 const std::string& activation,
                                 T* out,
                                 uint8_t* mask);

template <typename T>
void LaunchDropoutActBiasBackward(const GPUContext& dev_ctx,
                                  int rows,
                                  int cols,
                                  const FusedDropoutConfig& dropout,
                                  const T* out_grad,
                                  const T* x,
                                  const T* bias,
                                  const uint8_t* mask,
                                  T* x_grad,
                                  T* bias_grad,
                                  const std::string& activation);

template <typename T>
void LaunchResidualDropoutBiasForward(const GPUContext& dev_ctx,
                                      int rows,
                                      int cols,
                                      const FusedDropoutConfig& dropout,
                                      float epsilon,
                                      const T* x,
                                      const T* residual,
                                      const T* bias,
                                      T* out,
                                      uint8_t* mask);

template <typename T>
void LaunchResidualDropoutBiasBackward(const GPUContext& dev_ctx,
                                       int rows,
                                       int cols,
                                       const FusedDropoutConfig& dropout,
                                       float epsilon,
                                       const T* out_grad,
                                       const uint8_t* mask,
                                       T* x_grad,
                                       T* residual_grad,
                                       T* bias_grad);

template <typename T>
void LaunchLayernormResidualDropoutBiasForward(
    const GPUContext& dev_ctx,
    int rows,
    int cols,
    const FusedDropoutConfig& dropout,
    float epsilon,
    const T* x,
    const T* residual,
    const T* bias,
    const FusedLayerNormParamType<T>* scale,
    const FusedLayerNormParamType<T>* ln_bias,
    T* dropout_out,
    uint8_t* mask,
    T* out,
    FusedLayerNormParamType<T>* mean,
    FusedLayerNormParamType<T>* variance);

template <typename T>
void LaunchLayernormResidualDropoutBiasBackward(
    const GPUContext& dev_ctx,
    int rows,
    int cols,
    const FusedDropoutConfig& dropout,
    float epsilon,
    const T* out_grad,
    const T* dropout_out,
    const uint8_t* mask,
    const FusedLayerNormParamType<T>* scale,
    const FusedLayerNormParamType<T>* mean,
    const FusedLayerNormParamType<T>* variance,
    T* dropout_out_grad,
    FusedLayerNormParamType<T>* scale_grad,
    FusedLayerNormParamType<T>* bias_grad,
    T* x_grad,
    T* dropout_bias_grad,
    T* residual_grad);

template <typename T>
void LaunchAdd(const GPUContext& dev_ctx,
               const DenseTensor& x,
               const DenseTensor& y,
               DenseTensor* out);

template <typename T>
void LaunchAttentionLayerNormForward(const GPUContext& dev_ctx,
                                     float epsilon,
                                     int rows,
                                     int cols,
                                     const T* x,
                                     const FusedLayerNormParamType<T>* scale,
                                     const FusedLayerNormParamType<T>* bias,
                                     T* out,
                                     FusedLayerNormParamType<T>* mean,
                                     FusedLayerNormParamType<T>* variance);

template <typename T>
void LaunchAttentionLayerNormBackward(
    const GPUContext& dev_ctx,
    float epsilon,
    int rows,
    int cols,
    const T* x,
    const T* out_grad,
    const FusedLayerNormParamType<T>* scale,
    const FusedLayerNormParamType<T>* mean,
    const FusedLayerNormParamType<T>* variance,
    T* x_grad,
    FusedLayerNormParamType<T>* scale_grad,
    FusedLayerNormParamType<T>* bias_grad);

template <typename T>
void LaunchAttentionMatMulForward(const GPUContext& dev_ctx,
                                  bool trans_a,
                                  bool trans_b,
                                  int m,
                                  int n,
                                  int k,
                                  bool compute_bias,
                                  const DenseTensor* weight,
                                  const DenseTensor* input,
                                  const DenseTensor* bias,
                                  DenseTensor* output,
                                  DenseTensor* bias_out,
                                  bool fused = false);

template <typename T>
void LaunchAttentionMatMulBackward(const GPUContext& dev_ctx,
                                   bool trans_a,
                                   bool trans_b,
                                   int m,
                                   int n,
                                   int k,
                                   bool compute_bias,
                                   const DenseTensor* input,
                                   const DenseTensor* weight,
                                   const DenseTensor* output_grad,
                                   DenseTensor* input_grad,
                                   DenseTensor* weight_grad,
                                   DenseTensor* bias_grad,
                                   bool use_addto = false,
                                   bool fused = false);

template <typename T>
void LaunchFMHAForward(const GPUContext& dev_ctx,
                       int batch_size,
                       int sequence_length,
                       int num_heads,
                       int head_dim,
                       const AttentionDropoutConfig& dropout,
                       const DenseTensor& qkv,
                       const DenseTensor* cache_kv,
                       const DenseTensor* src_mask,
                       DenseTensor* transpose_out,
                       DenseTensor* cache_kv_out,
                       DenseTensor* qk_out,
                       DenseTensor* src_mask_out,
                       DenseTensor* softmax_out,
                       DenseTensor* dropout_mask_out,
                       DenseTensor* dropout_out,
                       DenseTensor* qktv_out,
                       DenseTensor* fmha_out);

template <typename T>
void LaunchFMHABackward(const GPUContext& dev_ctx,
                        int batch_size,
                        int sequence_length,
                        int num_heads,
                        int head_dim,
                        const AttentionDropoutConfig& dropout,
                        const DenseTensor& transpose_out,
                        const DenseTensor* src_mask,
                        const DenseTensor& softmax_out,
                        const DenseTensor& dropout_mask_out,
                        const DenseTensor& dropout_out,
                        const DenseTensor& qk_out,
                        const DenseTensor& src_mask_out,
                        const DenseTensor& fmha_out_grad,
                        DenseTensor* qktv_out_grad,
                        DenseTensor* dropout_out_grad,
                        DenseTensor* softmax_out_grad,
                        DenseTensor* src_mask_out_grad,
                        DenseTensor* qk_out_grad,
                        DenseTensor* transpose_out_grad,
                        DenseTensor* cache_kv_grad,
                        DenseTensor* qkv_grad);

}  // namespace phi::fusion
