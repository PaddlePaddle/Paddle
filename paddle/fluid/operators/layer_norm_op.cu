/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/operators/layer_norm_kernel.cu.h"
#include "paddle/fluid/operators/layer_norm_op.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

template <typename T>
void LayerNormDirectCUDAFunctor<T>::operator()(gpuStream_t stream,
                                               const T *input,
                                               std::vector<int> input_shape,
                                               const T *bias, const T *scale,
                                               T *output, T *mean, T *variance,
                                               int begin_norm_axis, float eps) {
  const auto x_dims = phi::make_ddim(input_shape);
  auto matrix_dim = phi::flatten_to_2d(x_dims, begin_norm_axis);
  int64_t batch_size = static_cast<int64_t>(matrix_dim[0]);
  int64_t feature_size = static_cast<int64_t>(matrix_dim[1]);
  switch (GetDesiredBlockDim(feature_size)) {
    FIXED_BLOCK_DIM_CASE(
        LayerNormForward<T, T, kBlockDim><<<batch_size, kBlockDim, 0, stream>>>(
            input, scale, bias, output, mean, variance, eps, feature_size));
    default:
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Product from begin_norm_axis to end in layer_norm must be larger "
          "than 1"));
      break;
  }
}

template class LayerNormDirectCUDAFunctor<float>;

template <typename T>
class LayerNormKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {}
};

template <typename T>
class LayerNormGradKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {}
};

}  // namespace operators
}  // namespace paddle

// namespace ops = paddle::operators;
// namespace plat = paddle::platform;
// #ifdef PADDLE_WITH_HIP
// MIOPEN do not support double
// REGISTER_OP_CUDA_KERNEL(
//     layer_norm,
//     ops::LayerNormKernel<paddle::platform::CUDADeviceContext, float>,
//     ops::LayerNormKernel<paddle::platform::CUDADeviceContext,
//     plat::float16>);
// REGISTER_OP_CUDA_KERNEL(
//     layer_norm_grad,
//     ops::LayerNormGradKernel<paddle::platform::CUDADeviceContext, float>,
//     ops::LayerNormGradKernel<paddle::platform::CUDADeviceContext,
//                              plat::float16>);
// #elif CUDNN_VERSION_MIN(8, 1, 0)
// REGISTER_OP_CUDA_KERNEL(
//     layer_norm,
//     ops::LayerNormKernel<paddle::platform::CUDADeviceContext, float>,
//     ops::LayerNormKernel<paddle::platform::CUDADeviceContext, double>,
//     ops::LayerNormKernel<paddle::platform::CUDADeviceContext, plat::float16>,
//     ops::LayerNormKernel<paddle::platform::CUDADeviceContext,
//     plat::bfloat16>);
// REGISTER_OP_CUDA_KERNEL(
//     layer_norm_grad,
//     ops::LayerNormGradKernel<paddle::platform::CUDADeviceContext, float>,
//     ops::LayerNormGradKernel<paddle::platform::CUDADeviceContext, double>,
//     ops::LayerNormGradKernel<paddle::platform::CUDADeviceContext,
//                              plat::float16>,
//     ops::LayerNormGradKernel<paddle::platform::CUDADeviceContext,
//                              plat::bfloat16>);
// #else
// REGISTER_OP_CUDA_KERNEL(
//     layer_norm,
//     ops::LayerNormKernel<paddle::platform::CUDADeviceContext, float>,
//     ops::LayerNormKernel<paddle::platform::CUDADeviceContext, double>,
//     ops::LayerNormKernel<paddle::platform::CUDADeviceContext,
//     plat::float16>);
// REGISTER_OP_CUDA_KERNEL(
//     layer_norm_grad,
//     ops::LayerNormGradKernel<paddle::platform::CUDADeviceContext, float>,
//     ops::LayerNormGradKernel<paddle::platform::CUDADeviceContext, double>,
//     ops::LayerNormGradKernel<paddle::platform::CUDADeviceContext,
//                              plat::float16>);
// #endif
