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

#include "paddle/platform/cudnn_helper.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using CudnnDataType = platform::CudnnDataType;

// BatchNormKernel for CPU, now only support NCHW data format
template <typename T>
class BatchNormKernel<platform::GPUPlace, T> : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "It must use GPUPlace.");
    const float epsilon = ctx.Attr<float>("epsilon");
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

    const int sample_size = H * W * D;

    // ------------------- cudnn descriptors ---------------------
    cudnnTensorDescriptor_t data_desc_;
    cudnnTensorDescriptor_t bn_param_desc_;
    cudnnBatchNormMode_t mode_;

    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&data_desc_));
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&bn_param_desc_));

    if (epsilon_ <= CUDNN_BN_MIN_EPSILON - FLT_EPSILON) {
      LOG(ERROR) << "Provided epsilon is smaller than "
                 << "CUDNN_BN_MIN_EPSILON. Setting it to "
                 << "CUDNN_BN_MIN_EPSILON instead.";
    }
    epsilon_ = std::max(epsilon_, CUDNN_BN_MIN_EPSILON);
#if CUDNN_VERSION_MIN(7, 0, 0)
    mode_ = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
#else
    mode_ = CUDNN_BATCHNORM_SPATIAL;
#endif

    VLOG(1) << "Setting descriptors.";
    std::vector<int64_t> cudnn_input_dims_ = x_dims;
    std::vector<int> dims;
    std::vector<int> strides;
    if (tensor_format == TensorFormat::NCHW) {
      dims = {N, C, H, W, D};
      strides = {C * H * W * D, H * W * D, W * D, D, 1};
    } else {
      dims = {N, C, H, W, D};
      strides = {H * W * D * C, 1, W * D * C, D * C, C};
    }
    CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
        data_desc_, CudnnDataType<T>::type,
        x_dims.size() > 3 ? x_dims.size() : 4, dims.data(), strides.data()));
    CUDNN_ENFORCE(
        cudnnDeriveBNTensorDescriptor(bn_param_desc_, data_desc_, mode_));

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

    auto handle = ctx.cuda_device_context().cudnn_handle();

    // Now, depending on whether we are running test or not, we have two paths.
    if (is_test) {
      // only when test we use input to do computation.
      const auto *est_mean = ctx.Input<Tensor>("Mean");
      const auto *est_var = ctx.Input<Tensor>("Variance");
      // Run inference mode.
      const auto &est_mean = Input(EST_MEAN);
      const auto &est_var = Input(EST_VAR);
      PADDLE_ENFORCE_EQ(est_mean->dims().size(), 1UL);
      PADDLE_ENFORCE_EQ(est_var > dims().size(), 1UL);
      PADDLE_ENFORCE_EQ(est_mean->dims()[0], C);
      PADDLE_ENFORCE_EQ(est_var->dims()[0], C);

      CUDNN_ENFORCE(cudnnBatchNormalizationForwardInference(
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

      void *save_var_data = save_var->template mutable_data<T>();

      CUDNN_ENFORCE(cudnnBatchNormalizationForwardTraining(
          handle, mode_, CudnnDataType<T>::kOne(), CudnnDataType<T>::kZero(),
          data_desc_, x->template data<T>(), data_desc_,
          y->template mutable_data<T>(), bn_param_desc_,
          scale->template data<BNParamType>(),
          bias->template data<BNParamType>(), this_factor,
          mean_out->template mutable_data<T>(),
          variance_out->template mutable_data<T>(), epsilon,
          saved_mean->template mutable_data<T>(),
          saved_variance->template mutable_data<T>()));
    }

    // clean when exit.
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(data_desc_));
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(bn_param_desc_));
  }
};

template <typename T>
class BatchNormGradKernel<platform::CPUPlace, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {}

 private:
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_GPU_KERNEL(batch_norm,
                       ops::BatchNormKernel<paddle::platform::GPUPlace, float>);
REGISTER_OP_GPU_KERNEL(
    BATCH_NORM_GRAD,
    ops::BatchNormGradKernel<paddle::platform::GPUPlace, float>);
