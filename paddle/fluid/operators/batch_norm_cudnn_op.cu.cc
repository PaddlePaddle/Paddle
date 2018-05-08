/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <cfloat>

#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/operators/batch_norm_op.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/cudnn_helper.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DataLayout = framework::DataLayout;
using ScopedTensorDescriptor = platform::ScopedTensorDescriptor;
template <typename T>
using CudnnDataType = platform::CudnnDataType<T>;
template <typename T>
using BatchNormParamType = typename CudnnDataType<T>::BatchNormParamType;

void ExtractNCWHD(const framework::DDim &dims, const DataLayout &data_layout,
                  int *N, int *C, int *H, int *W, int *D) {
  *N = dims[0];
  if (dims.size() == 2) {
    *C = dims[1];
    *H = 1;
    *W = 1;
    *D = 1;
  } else {
    *C = data_layout == DataLayout::kNCHW ? dims[1] : dims[dims.size() - 1];
    *H = data_layout == DataLayout::kNCHW ? dims[2] : dims[1];
    *W = dims.size() > 3
             ? (data_layout == DataLayout::kNCHW ? dims[3] : dims[2])
             : 1;
    *D = dims.size() > 4
             ? (data_layout == DataLayout::kNCHW ? dims[4] : dims[3])
             : 1;
  }
}

void FillTensorDescriptor(const Tensor &X, const DataLayout &data_layout,
                          bool is_test, ScopedTensorDescriptor *data_desc,
                          ScopedTensorDescriptor *bn_param_desc,
                          cudnnBatchNormMode_t *mode) {
#if CUDNN_VERSION_MIN(7, 0, 0)
  *mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
#else
  *mode = CUDNN_BATCHNORM_SPATIAL;
#endif
  if (is_test) {
    // Note: PERSISTENT not implemented for inference
    *mode = CUDNN_BATCHNORM_SPATIAL;
  }
  const auto &x_dims = X.dims();
  // Get the size for each dimension.
  // NCHW [batch_size, in_channels, in_height, in_width]
  int N, C, H, W, D;
  ExtractNCWHD(x_dims, data_layout, &N, &C, &H, &W, &D);

  VLOG(10) << "Setting descriptors.";
  std::vector<int> dims;
  std::vector<int> strides;
  if (data_layout == DataLayout::kNCHW) {
    dims = {N, C, H, W, D};
    strides = {C * H * W * D, H * W * D, W * D, D, 1};
  } else {
    dims = {N, C, H, W, D};
    strides = {H * W * D * C, 1, W * D * C, D * C, C};
  }
  int nbDims = x_dims.size() > 3 ? x_dims.size() : 4;
  data_desc->descriptor<T>(data_layout, nbDims, dims, strides);
  PADDLE_ENFORCE(platform::dynload::cudnnDeriveBNTensorDescriptor(
      *bn_param_desc, *data_desc, *mode));
}

template <typename T>
class CUDNNBatchNormKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    double epsilon = static_cast<double>(ctx.Attr<float>("epsilon"));
    const float momentum = ctx.Attr<float>("momentum");
    const bool is_test = ctx.Attr<bool>("is_test");
    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
    const DataLayout data_layout =
        framework::StringToDataLayout(data_layout_str);

    const auto *x = ctx.Input<Tensor>("X");
    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *bias = ctx.Input<Tensor>("Bias");
    auto *y = ctx.Output<Tensor>("Y");
    y->mutable_data<T>(ctx.GetPlace());

    if (epsilon <= CUDNN_BN_MIN_EPSILON - FLT_EPSILON) {
      LOG(ERROR) << "Provided epsilon is smaller than "
                 << "CUDNN_BN_MIN_EPSILON. Setting it to "
                 << "CUDNN_BN_MIN_EPSILON instead.";
    }
    epsilon = std::max(epsilon, CUDNN_BN_MIN_EPSILON);

    // ------------------- cudnn descriptors ---------------------
    ScopedTensorDescriptor data_desc_;
    ScopedTensorDescriptor bn_param_desc_;
    cudnnBatchNormMode_t mode_;
    FillTensorDescriptor(*x, is_test, &data_desc_, &bn_param_desc_, &mode_);

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    // Run inference mode or training mode depends is_test or not.
    if (is_test) {
      const auto *est_mean = ctx.Input<Tensor>("Mean");
      const auto *est_var = ctx.Input<Tensor>("Variance");
      PADDLE_ENFORCE(platform::dynload::cudnnBatchNormalizationForwardInference(
          dev_ctx.cudnn_handle(), mode_, CudnnDataType<T>::kOne(),
          CudnnDataType<T>::kZero(), data_desc_, x->template data<T>(),
          data_desc_, y->template mutable_data<T>(ctx.GetPlace()),
          bn_param_desc_, scale->template data<BatchNormParamType<T>>(),
          bias->template data<BatchNormParamType<T>>(),
          est_mean->template data<BatchNormParamType<T>>(),
          est_var->template data<BatchNormParamType<T>>(), epsilon));
    } else {
      // obtain running mean and running inv var, and see if we need to
      // initialize them.
      auto *mean_out = ctx.Output<Tensor>("MeanOut");
      auto *variance_out = ctx.Output<Tensor>("VarianceOut");
      mean_out->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());
      variance_out->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());

      auto *saved_mean = ctx.Output<Tensor>("SavedMean");
      auto *saved_variance = ctx.Output<Tensor>("SavedVariance");
      saved_mean->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());
      saved_variance->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());
      math::SetConstant<platform::CUDADeviceContext, BatchNormParamType<T>>
          functor;
      functor(dev_ctx, saved_mean, static_cast<BatchNormParamType<T>>(0));
      functor(dev_ctx, saved_variance, static_cast<BatchNormParamType<T>>(0));

      double this_factor = 1. - momentum;

      PADDLE_ENFORCE(platform::dynload::cudnnBatchNormalizationForwardTraining(
          dev_ctx.cudnn_handle(), mode_, CudnnDataType<T>::kOne(),
          CudnnDataType<T>::kZero(), data_desc_, x->template data<T>(),
          data_desc_, y->template mutable_data<T>(ctx.GetPlace()),
          bn_param_desc_, scale->template data<BatchNormParamType<T>>(),
          bias->template data<BatchNormParamType<T>>(), this_factor,
          mean_out->template mutable_data<BatchNormParamType<T>>(
              ctx.GetPlace()),
          variance_out->template mutable_data<BatchNormParamType<T>>(
              ctx.GetPlace()),
          epsilon, saved_mean->template mutable_data<BatchNormParamType<T>>(
                       ctx.GetPlace()),
          saved_variance->template mutable_data<BatchNormParamType<T>>(
              ctx.GetPlace())));
    }
  }
};

template <typename T>
class CUDNNBatchNormGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    double epsilon = static_cast<double>(ctx.Attr<float>("epsilon"));
    epsilon = std::max(epsilon, CUDNN_BN_MIN_EPSILON);
    const bool is_test = ctx.Attr<bool>("is_test");
    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
    const DataLayout data_layout =
        framework::StringToDataLayout(data_layout_str);
    const auto *x = ctx.Input<Tensor>("X");
    const auto *d_y = ctx.Input<Tensor>(framework::GradVarName("Y"));
    const auto *scale = ctx.Input<Tensor>("Scale");

    // ------------------- cudnn descriptors ---------------------
    ScopedTensorDescriptor data_desc_;
    ScopedTensorDescriptor bn_param_desc_;
    cudnnBatchNormMode_t mode_;
    FillTensorDescriptor(*x, data_layout, is_test, &data_desc_, &bn_param_desc_,
                         &mode_);

    // init output
    auto *d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *d_scale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto *d_bias = ctx.Output<Tensor>(framework::GradVarName("Bias"));

    d_x->mutable_data<T>(ctx.GetPlace());
    d_scale->mutable_data<T>(ctx.GetPlace());
    d_bias->mutable_data<T>(ctx.GetPlace());

    const auto *saved_mean = ctx.Input<Tensor>("SavedMean");
    const auto *saved_var = ctx.Input<Tensor>("SavedVariance");
    const void *saved_mean_data = saved_mean->template data<T>();
    const void *saved_var_data = saved_var->template data<T>();

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    PADDLE_ENFORCE(platform::dynload::cudnnBatchNormalizationBackward(
        dev_ctx.cudnn_handle(), mode_, CudnnDataType<T>::kOne(),
        CudnnDataType<T>::kZero(), CudnnDataType<T>::kOne(),
        CudnnDataType<T>::kZero(), data_desc_, x->template data<T>(),
        data_desc_, d_y->template data<T>(), data_desc_,
        d_x->template mutable_data<T>(ctx.GetPlace()), bn_param_desc_,
        scale->template data<T>(),
        d_scale->template mutable_data<T>(ctx.GetPlace()),
        d_bias->template mutable_data<T>(ctx.GetPlace()), epsilon,
        saved_mean_data, saved_var_data));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_KERNEL(batch_norm, CUDNN, plat::CUDAPlace,
                   ops::CUDNNBatchNormKernel<float>,
                   ops::CUDNNBatchNormKernel<double>,
                   ops::CUDNNBatchNormKernel<plat::float16>);
REGISTER_OP_KERNEL(batch_norm_grad, CUDNN, plat::CUDAPlace,
                   ops::CUDNNBatchNormGradKernel<float>,
                   ops::CUDNNBatchNormGradKernel<double>);
