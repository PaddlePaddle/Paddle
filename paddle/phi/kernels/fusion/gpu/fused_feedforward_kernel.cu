// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/fusion/gpu/fused_transformer_kernel_launch.h"

#include "paddle/phi/kernels/fusion/gpu/fused_dropout_helper.h"

namespace phi::fusion {

namespace {

DropoutParam MakeDropoutParam(const FusedDropoutConfig& config) {
  return DropoutParam(config.fix_seed,
                      0,
                      config.is_test,
                      config.is_upscale_in_train,
                      config.probability,
                      config.seed,
                      config.seed_value);
}

}  // namespace

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
                            FusedLayerNormParamType<T>* variance) {
  FusedDropoutLayerNormHelper<T, uint8_t> helper(rows, cols, epsilon);
  helper.LayerNorm(dev_ctx, x, scale, bias, out, mean, variance);
}

template <typename T>
void LaunchDropoutActBiasForward(const GPUContext& dev_ctx,
                                 int rows,
                                 int cols,
                                 const FusedDropoutConfig& dropout,
                                 const T* x,
                                 const T* bias,
                                 const std::string& activation,
                                 T* out,
                                 uint8_t* mask) {
  auto dropout_param = MakeDropoutParam(dropout);
  FusedDropoutHelper<T, uint8_t> helper(dev_ctx, rows, cols, dropout_param);
  helper.DropoutActBias(dev_ctx, x, bias, activation, out, mask);
}

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
                                      uint8_t* mask) {
  auto dropout_param = MakeDropoutParam(dropout);
  FusedDropoutLayerNormHelper<T, uint8_t> helper(
      dev_ctx, rows, cols, dropout_param, epsilon);
  helper.ResidualDropoutBias(dev_ctx, x, residual, bias, out, mask);
}

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
    FusedLayerNormParamType<T>* variance) {
  auto dropout_param = MakeDropoutParam(dropout);
  FusedDropoutLayerNormHelper<T, uint8_t> helper(
      dev_ctx, rows, cols, dropout_param, epsilon);
  helper.LayernormResidualDropoutBias(dev_ctx,
                                      x,
                                      residual,
                                      bias,
                                      scale,
                                      ln_bias,
                                      dropout_out,
                                      mask,
                                      out,
                                      mean,
                                      variance);
}

#define INSTANTIATE_FUSED_FEEDFORWARD_FORWARD(T, U)                            \
  template void LaunchLayerNormForward<T>(const GPUContext&,                   \
                                          int,                                 \
                                          int,                                 \
                                          float,                               \
                                          const T*,                            \
                                          const U*,                            \
                                          const U*,                            \
                                          T*,                                  \
                                          U*,                                  \
                                          U*);                                 \
  template void LaunchDropoutActBiasForward<T>(const GPUContext&,              \
                                               int,                            \
                                               int,                            \
                                               const FusedDropoutConfig&,      \
                                               const T*,                       \
                                               const T*,                       \
                                               const std::string&,             \
                                               T*,                             \
                                               uint8_t*);                      \
  template void LaunchResidualDropoutBiasForward<T>(const GPUContext&,         \
                                                    int,                       \
                                                    int,                       \
                                                    const FusedDropoutConfig&, \
                                                    float,                     \
                                                    const T*,                  \
                                                    const T*,                  \
                                                    const T*,                  \
                                                    T*,                        \
                                                    uint8_t*);                 \
  template void LaunchLayernormResidualDropoutBiasForward<T>(                  \
      const GPUContext&,                                                       \
      int,                                                                     \
      int,                                                                     \
      const FusedDropoutConfig&,                                               \
      float,                                                                   \
      const T*,                                                                \
      const T*,                                                                \
      const T*,                                                                \
      const U*,                                                                \
      const U*,                                                                \
      T*,                                                                      \
      uint8_t*,                                                                \
      T*,                                                                      \
      U*,                                                                      \
      U*)

INSTANTIATE_FUSED_FEEDFORWARD_FORWARD(float, float);
INSTANTIATE_FUSED_FEEDFORWARD_FORWARD(double, double);
INSTANTIATE_FUSED_FEEDFORWARD_FORWARD(phi::float16, float);

#undef INSTANTIATE_FUSED_FEEDFORWARD_FORWARD

}  // namespace phi::fusion
