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
  const auto x_dims = framework::make_ddim(input_shape);
  auto matrix_dim = framework::flatten_to_2d(x_dims, begin_norm_axis);
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

template <typename T>
class LayerNormKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    using U = LayerNormParamType<T>;
    const float epsilon = ctx.Attr<float>("epsilon");
    auto *scale = ctx.Input<Tensor>("Scale");
    auto *bias = ctx.Input<Tensor>("Bias");
    auto *x = ctx.Input<Tensor>("X");

    auto *y = ctx.Output<Tensor>("Y");
    auto *mean = ctx.Output<Tensor>("Mean");
    auto *var = ctx.Output<Tensor>("Variance");
    const auto begin_norm_axis = ctx.Attr<int>("begin_norm_axis");

    const auto x_dims = x->dims();
    auto *x_data = x->data<T>();
    auto *y_data = y->mutable_data<T>(ctx.GetPlace());
    auto *mean_data = mean->mutable_data<U>(ctx.GetPlace());
    auto *var_data = var->mutable_data<U>(ctx.GetPlace());
    auto *scale_data = (scale == nullptr ? nullptr : scale->data<U>());
    auto *bias_data = (bias == nullptr ? nullptr : bias->data<U>());

    auto matrix_dim = framework::flatten_to_2d(x_dims, begin_norm_axis);
    int64_t batch_size = static_cast<int64_t>(matrix_dim[0]);
    int64_t feature_size = static_cast<int64_t>(matrix_dim[1]);

    auto stream = ctx.cuda_device_context().stream();

    switch (GetDesiredBlockDim(feature_size)) {
      FIXED_BLOCK_DIM_CASE(
          LayerNormForward<T, U,
                           kBlockDim><<<batch_size, kBlockDim, 0, stream>>>(
              x_data, scale_data, bias_data, y_data, mean_data, var_data,
              epsilon, feature_size));
      default:
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Product from begin_norm_axis to end must be larger than 1"));
        break;
    }
  }
};

template <typename T>
class LayerNormGradKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    using U = LayerNormParamType<T>;
    const float epsilon = ctx.Attr<float>("epsilon");
    // d_x, d_scale, d_bias may be nullptr
    auto *d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *d_scale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto *d_bias = ctx.Output<Tensor>(framework::GradVarName("Bias"));

    auto *x = ctx.Input<Tensor>("X");
    auto *mean = ctx.Input<Tensor>("Mean");
    auto *var = ctx.Input<Tensor>("Variance");
    auto *scale = ctx.Input<Tensor>("Scale");
    auto *d_y = ctx.Input<Tensor>(framework::GradVarName("Y"));

    auto *x_data = x->data<T>();
    auto *d_y_data = d_y->data<T>();
    auto *mean_data = mean->data<U>();
    auto *var_data = var->data<U>();

    auto *scale_data = (scale == nullptr ? nullptr : scale->data<U>());
    auto *d_scale_data =
        (d_scale == nullptr ? nullptr
                            : d_scale->mutable_data<U>(ctx.GetPlace()));
    auto *d_bias_data =
        (d_bias == nullptr ? nullptr : d_bias->mutable_data<U>(ctx.GetPlace()));
    auto *d_x_data =
        (d_x == nullptr ? nullptr : d_x->mutable_data<T>(ctx.GetPlace()));

    const auto &x_dims = x->dims();
    const auto begin_norm_axis = ctx.Attr<int>("begin_norm_axis");
    auto matrix_dim = framework::flatten_to_2d(x_dims, begin_norm_axis);
    int64_t batch_size = static_cast<int64_t>(matrix_dim[0]);
    int64_t feature_size = static_cast<int64_t>(matrix_dim[1]);

    LayerNormBackward<T, U>(x_data, d_y_data, scale_data, mean_data, var_data,
                            d_x_data, d_scale_data, d_bias_data, epsilon,
                            batch_size, feature_size,
                            ctx.cuda_device_context());
  }
};

template class LayerNormDirectCUDAFunctor<float>;
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
#ifdef PADDLE_WITH_HIP
// MIOPEN do not support double
REGISTER_OP_CUDA_KERNEL(
    layer_norm,
    ops::LayerNormKernel<paddle::platform::CUDADeviceContext, float>,
    ops::LayerNormKernel<paddle::platform::CUDADeviceContext, plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    layer_norm_grad,
    ops::LayerNormGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::LayerNormGradKernel<paddle::platform::CUDADeviceContext,
                             plat::float16>);
#else
REGISTER_OP_CUDA_KERNEL(
    layer_norm,
    ops::LayerNormKernel<paddle::platform::CUDADeviceContext, float>,
    ops::LayerNormKernel<paddle::platform::CUDADeviceContext, double>,
    ops::LayerNormKernel<paddle::platform::CUDADeviceContext, plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    layer_norm_grad,
    ops::LayerNormGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::LayerNormGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::LayerNormGradKernel<paddle::platform::CUDADeviceContext,
                             plat::float16>);
#endif
