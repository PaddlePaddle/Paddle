/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/operators/batch_norm_op.h"

#include <cfloat>
#include "paddle/operators/math/math_function.h"
#include "paddle/platform/cudnn_helper.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T>
using CudnnDataType = platform::CudnnDataType<T>;

// BatchNormKernel for CPU, now only support NCHW data format
template <typename T>
class BatchNormKernel<platform::GPUPlace, T> : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "It must use GPUPlace.");
    double epsilon = static_cast<double>(ctx.Attr<float>("epsilon"));
    const float momentum = ctx.Attr<float>("momentum");
    const std::string tensor_format_str =
        ctx.Attr<std::string>("tensor_format");
    const bool is_test = ctx.Attr<bool>("is_test");
    const TensorFormat tensor_format = StringToTensorFormat(tensor_format_str);

    // Get the size for each dimension.
    // NCHW [batch_size, in_channels, in_height, in_width]
    const auto *x = ctx.Input<Tensor>("X");
    const auto &x_dims = x->dims();
    PADDLE_ENFORCE(x_dims.size() >= 3 && x_dims.size() <= 5,
                   "The Input dim size should be between 3 and 5");
    const int N = x_dims[0];
    const int C =
        (tensor_format == TensorFormat::NCHW ? x_dims[1]
                                             : x_dims[x_dims.size() - 1]);
    const int H = (tensor_format == TensorFormat::NCHW ? x_dims[2] : x_dims[1]);
    const int W =
        x_dims.size() > 3
            ? (tensor_format == TensorFormat::NCHW ? x_dims[3] : x_dims[2])
            : 1;
    const int D =
        x_dims.size() > 4
            ? (tensor_format == TensorFormat::NCHW ? x_dims[4] : x_dims[3])
            : 1;

    // ------------------- cudnn descriptors ---------------------
    cudnnTensorDescriptor_t data_desc_;
    cudnnTensorDescriptor_t bn_param_desc_;
    cudnnBatchNormMode_t mode_;

    PADDLE_ENFORCE(platform::dynload::cudnnCreateTensorDescriptor(&data_desc_));
    PADDLE_ENFORCE(
        platform::dynload::cudnnCreateTensorDescriptor(&bn_param_desc_));

    if (epsilon <= CUDNN_BN_MIN_EPSILON - FLT_EPSILON) {
      LOG(ERROR) << "Provided epsilon is smaller than "
                 << "CUDNN_BN_MIN_EPSILON. Setting it to "
                 << "CUDNN_BN_MIN_EPSILON instead.";
    }
    epsilon = std::max(epsilon, CUDNN_BN_MIN_EPSILON);
#if CUDNN_VERSION_MIN(7, 0, 0)
    mode_ = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
#else
    mode_ = CUDNN_BATCHNORM_SPATIAL;
#endif

    VLOG(1) << "Setting descriptors.";
    std::vector<int> dims;
    std::vector<int> strides;
    if (tensor_format == TensorFormat::NCHW) {
      dims = {N, C, H, W, D};
      strides = {C * H * W * D, H * W * D, W * D, D, 1};
    } else {
      dims = {N, C, H, W, D};
      strides = {H * W * D * C, 1, W * D * C, D * C, C};
    }
    PADDLE_ENFORCE(platform::dynload::cudnnSetTensorNdDescriptor(
        data_desc_, CudnnDataType<T>::type,
        x_dims.size() > 3 ? x_dims.size() : 4, dims.data(), strides.data()));
    PADDLE_ENFORCE(platform::dynload::cudnnDeriveBNTensorDescriptor(
        bn_param_desc_, data_desc_, mode_));

    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *bias = ctx.Input<Tensor>("Bias");

    auto *y = ctx.Output<Tensor>("Y");
    auto *mean_out = ctx.Output<Tensor>("MeanOut");
    auto *variance_out = ctx.Output<Tensor>("VarianceOut");
    auto *saved_mean = ctx.Output<Tensor>("SavedMean");
    auto *saved_variance = ctx.Output<Tensor>("SavedVariance");

    // alloc memory
    y->mutable_data<T>(ctx.GetPlace());
    mean_out->mutable_data<T>(ctx.GetPlace());
    variance_out->mutable_data<T>(ctx.GetPlace());
    saved_mean->mutable_data<T>(ctx.GetPlace());
    saved_variance->mutable_data<T>(ctx.GetPlace());

    math::SetConstant<platform::GPUPlace, T> functor;
    functor(ctx.device_context(), saved_mean, 0);
    functor(ctx.device_context(), saved_variance, 0);
    // FIXME(qiao) should not set zero self
    functor(ctx.device_context(), mean_out, 0);
    functor(ctx.device_context(), variance_out, 0);

    auto handle = ctx.cuda_device_context().cudnn_handle();

    // Now, depending on whether we are running test or not, we have two paths.
    if (is_test) {
      // only when test we use input to do computation.
      const auto *est_mean = ctx.Input<Tensor>("Mean");
      const auto *est_var = ctx.Input<Tensor>("Variance");
      // Run inference mode.
      PADDLE_ENFORCE_EQ(est_mean->dims().size(), 1UL);
      PADDLE_ENFORCE_EQ(est_var->dims().size(), 1UL);
      PADDLE_ENFORCE_EQ(est_mean->dims()[0], C);
      PADDLE_ENFORCE_EQ(est_var->dims()[0], C);

      PADDLE_ENFORCE(platform::dynload::cudnnBatchNormalizationForwardInference(
          handle,
          // Note: PERSISTENT not implemented for inference
          CUDNN_BATCHNORM_SPATIAL, CudnnDataType<T>::kOne(),
          CudnnDataType<T>::kZero(), data_desc_, x->template data<T>(),
          data_desc_, y->template mutable_data<T>(ctx.GetPlace()),
          bn_param_desc_, scale->template data<T>(), bias->template data<T>(),
          est_mean->template data<T>(), est_var->template data<T>(), epsilon));
    } else {
      // Run training mode.
      // obtain running mean and running inv var, and see if we need to
      // initialize them.
      double this_factor = 1. - momentum;

      PADDLE_ENFORCE(platform::dynload::cudnnBatchNormalizationForwardTraining(
          handle, mode_, CudnnDataType<T>::kOne(), CudnnDataType<T>::kZero(),
          data_desc_, x->template data<T>(), data_desc_,
          y->template mutable_data<T>(ctx.GetPlace()), bn_param_desc_,
          scale->template data<T>(), bias->template data<T>(), this_factor,
          mean_out->template mutable_data<T>(ctx.GetPlace()),
          variance_out->template mutable_data<T>(ctx.GetPlace()), epsilon,
          saved_mean->template mutable_data<T>(ctx.GetPlace()),
          saved_variance->template mutable_data<T>(ctx.GetPlace())));
    }

    // clean when exit.
    PADDLE_ENFORCE(platform::dynload::cudnnDestroyTensorDescriptor(data_desc_));
    PADDLE_ENFORCE(
        platform::dynload::cudnnDestroyTensorDescriptor(bn_param_desc_));
  }
};

template <typename T>
class BatchNormGradKernel<platform::GPUPlace, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "It must use GPUPlace.");
    double epsilon = static_cast<double>(ctx.Attr<float>("epsilon"));
    const auto *X = ctx.Input<Tensor>("X");
    const auto *dY = ctx.Input<Tensor>(framework::GradVarName("Y"));
    const auto *scale = ctx.Input<Tensor>("Scale");

    const auto &x_dims = X->dims();

    PADDLE_ENFORCE(x_dims.size() >= 3 && x_dims.size() <= 5,
                   "The Input dim size should be between 3 and 5");
    const int N = x_dims[0];
    const int C =
        (tensor_format == TensorFormat::NCHW ? x_dims[1]
                                             : x_dims[x_dims.size() - 1]);
    const int H = (tensor_format == TensorFormat::NCHW ? x_dims[2] : x_dims[1]);
    const int W =
        x_dims.size() > 3
            ? (tensor_format == TensorFormat::NCHW ? x_dims[3] : x_dims[2])
            : 1;
    const int D =
        x_dims.size() > 4
            ? (tensor_format == TensorFormat::NCHW ? x_dims[4] : x_dims[3])
            : 1;

    PADDLE_ENFORCE_EQ(scale->dims().size(), 1UL);
    PADDLE_ENFORCE_EQ(scale->dims()[0], C);

    // ------------------- cudnn descriptors ---------------------
    cudnnTensorDescriptor_t data_desc_;
    cudnnTensorDescriptor_t bn_param_desc_;
    cudnnBatchNormMode_t mode_;

    PADDLE_ENFORCE(platform::dynload::cudnnCreateTensorDescriptor(&data_desc_));
    PADDLE_ENFORCE(
        platform::dynload::cudnnCreateTensorDescriptor(&bn_param_desc_));
    if (epsilon <= CUDNN_BN_MIN_EPSILON - FLT_EPSILON) {
      LOG(ERROR) << "Provided epsilon is smaller than "
                 << "CUDNN_BN_MIN_EPSILON. Setting it to "
                 << "CUDNN_BN_MIN_EPSILON instead.";
    }
    epsilon = std::max(epsilon, CUDNN_BN_MIN_EPSILON);
#if CUDNN_VERSION_MIN(7, 0, 0)
    mode_ = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
#else
    mode_ = CUDNN_BATCHNORM_SPATIAL;
#endif

    vector<int> dims = {N, C, H, W, D};
    vector<int> strides = {H * W * C * D, 1, W * D * C, D * C, C};
    PADDLE_ENFORCE(cudnnSetTensorNdDescriptor(
        data_desc_, CudnnDataType<T>::type,
        x_dims.size() > 3 ? x_dims.size() : 4, dims.data(), strides.data()));
    PADDLE_ENFORCE(
        cudnnDeriveBNTensorDescriptor(bn_param_desc_, data_desc_, mode_));

    // init output
    auto *dX = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *dScale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto *dBias = ctx.Output<Tensor>(framework::GradVarName("Bias"));

    dX->mutable_data<T>(ctx.GetPlace());
    dScale->mutable_data<T>(ctx.GetPlace());
    dBias->mutable_data<T>(ctx.GetPlace());

    const auto *saved_mean = ctx.Input<Tensor>("SavedMean");
    const auto *saved_var = ctx.Input<Tensor>("SavedVariance");
    const void *saved_mean_data = saved_mean->template data<T>();
    const void *saved_var_data = saved_var->template data<T>();

    PADDLE_ENFORCE(platform::dynload::cudnnBatchNormalizationBackward(
        ctx.cuda_device_context().cudnn_handle(), mode_,
        CudnnDataType<T>::kOne(), CudnnDataType<T>::kZero(),
        CudnnDataType<T>::kOne(), CudnnDataType<T>::kZero(), data_desc_,
        X->template data<T>(), data_desc_, dY->template data<T>(), data_desc_,
        dX->template mutable_data<T>(), bn_param_desc_,
        scale->template data<T>(), dScale->template mutable_data<T>(),
        dBias->template mutable_data<T>(), epsilon, saved_mean_data,
        saved_var_data));

    // clean when exit.
    PADDLE_ENFORCE(platform::dynload::cudnnDestroyTensorDescriptor(data_desc_));
    PADDLE_ENFORCE(
        platform::dynload::cudnnDestroyTensorDescriptor(bn_param_desc_));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_GPU_KERNEL(batch_norm,
                       ops::BatchNormKernel<paddle::platform::GPUPlace, float>);
REGISTER_OP_GPU_KERNEL(
    batch_norm_grad,
    ops::BatchNormGradKernel<paddle::platform::GPUPlace, float>);
