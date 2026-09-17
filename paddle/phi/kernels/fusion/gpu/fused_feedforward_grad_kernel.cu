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

#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
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
                             FusedLayerNormParamType<T>* bias_grad) {
  FusedDropoutLayerNormHelper<T, uint8_t> helper(rows, cols, epsilon);
  helper.LayerNormGrad(dev_ctx,
                       out_grad,
                       x,
                       scale,
                       mean,
                       variance,
                       x_grad,
                       scale_grad,
                       bias_grad);
}

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
                                  const std::string& activation) {
  auto dropout_param = MakeDropoutParam(dropout);
  FusedDropoutHelper<T, uint8_t> helper(dev_ctx, rows, cols, dropout_param);
  helper.DropoutActBiasGrad(
      dev_ctx, out_grad, x, bias, mask, x_grad, bias_grad, activation);
}

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
                                       T* bias_grad) {
  auto dropout_param = MakeDropoutParam(dropout);
  FusedDropoutLayerNormHelper<T, uint8_t> helper(
      dev_ctx, rows, cols, dropout_param, epsilon);
  helper.ResidualDropoutBiasGrad(
      dev_ctx, out_grad, mask, x_grad, residual_grad, bias_grad);
}

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
    T* residual_grad) {
  auto dropout_param = MakeDropoutParam(dropout);
  FusedDropoutLayerNormHelper<T, uint8_t> helper(
      dev_ctx, rows, cols, dropout_param, epsilon);
  helper.LayernormResidualDropoutBiasGrad(dev_ctx,
                                          out_grad,
                                          dropout_out,
                                          mask,
                                          scale,
                                          mean,
                                          variance,
                                          dropout_out_grad,
                                          scale_grad,
                                          bias_grad,
                                          x_grad,
                                          dropout_bias_grad,
                                          residual_grad);
}

template <typename T>
void LaunchAdd(const GPUContext& dev_ctx,
               const DenseTensor& x,
               const DenseTensor& y,
               DenseTensor* out) {
  std::vector<const DenseTensor*> inputs = {&x, &y};
  std::vector<DenseTensor*> outputs = {out};
  funcs::ElementwiseKernel<T>(
      dev_ctx, inputs, &outputs, funcs::AddFunctor<T>());
}

#define INSTANTIATE_FUSED_FEEDFORWARD_BACKWARD(T, U)                       \
  template void LaunchLayerNormBackward<T>(const GPUContext&,              \
                                           int,                            \
                                           int,                            \
                                           float,                          \
                                           const T*,                       \
                                           const T*,                       \
                                           const U*,                       \
                                           const U*,                       \
                                           const U*,                       \
                                           T*,                             \
                                           U*,                             \
                                           U*);                            \
  template void LaunchDropoutActBiasBackward<T>(const GPUContext&,         \
                                                int,                       \
                                                int,                       \
                                                const FusedDropoutConfig&, \
                                                const T*,                  \
                                                const T*,                  \
                                                const T*,                  \
                                                const uint8_t*,            \
                                                T*,                        \
                                                T*,                        \
                                                const std::string&);       \
  template void LaunchResidualDropoutBiasBackward<T>(                      \
      const GPUContext&,                                                   \
      int,                                                                 \
      int,                                                                 \
      const FusedDropoutConfig&,                                           \
      float,                                                               \
      const T*,                                                            \
      const uint8_t*,                                                      \
      T*,                                                                  \
      T*,                                                                  \
      T*);                                                                 \
  template void LaunchLayernormResidualDropoutBiasBackward<T>(             \
      const GPUContext&,                                                   \
      int,                                                                 \
      int,                                                                 \
      const FusedDropoutConfig&,                                           \
      float,                                                               \
      const T*,                                                            \
      const T*,                                                            \
      const uint8_t*,                                                      \
      const U*,                                                            \
      const U*,                                                            \
      const U*,                                                            \
      T*,                                                                  \
      U*,                                                                  \
      U*,                                                                  \
      T*,                                                                  \
      T*,                                                                  \
      T*);                                                                 \
  template void LaunchAdd<T>(                                              \
      const GPUContext&, const DenseTensor&, const DenseTensor&, DenseTensor*)

INSTANTIATE_FUSED_FEEDFORWARD_BACKWARD(float, float);
INSTANTIATE_FUSED_FEEDFORWARD_BACKWARD(double, double);
INSTANTIATE_FUSED_FEEDFORWARD_BACKWARD(phi::float16, float);

#undef INSTANTIATE_FUSED_FEEDFORWARD_BACKWARD

}  // namespace phi::fusion
