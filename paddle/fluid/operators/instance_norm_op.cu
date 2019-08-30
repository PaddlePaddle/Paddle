/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include <cfloat>
#include <string>
#include <vector>
#include "cub/cub.cuh"
#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/operators/batch_norm_op.h"
#include "paddle/fluid/operators/instance_norm_op.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/cudnn_helper.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DataLayout = framework::DataLayout;
template <typename T>
using CudnnDataType = platform::CudnnDataType<T>;
template <typename T>
using BatchNormParamType = typename CudnnDataType<T>::BatchNormParamType;

template <typename T>
static __global__ void repeat_param(const T *input, T *output,
                                    const int repeat_num, const int C) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < repeat_num * C;
       i += blockDim.x * gridDim.x) {
    int index = i % C;
    output[i] = input[index];
  }
}

template <typename T, int BlockDim>
static __global__ void add_param(const T *input, T *output,
                                 const int repeat_num, const int C) {
  typedef cub::BlockReduce<T, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage ou_storage;
  for (int i = blockIdx.x; i < C; i += gridDim.x) {
    T ou = static_cast<T>(0);
    for (int j = threadIdx.x; j < repeat_num; j += blockDim.x) {
      const int index = j * C + i;
      ou += static_cast<T>(input[index]);
    }
    ou = BlockReduce(ou_storage).Reduce(ou, cub::Sum());
    if (threadIdx.x == 0) {
      output[i] = ou;
    }
    __syncthreads();
  }
}

template <typename T>
class InstanceNormKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "It must be CUDAPlace.");
    double epsilon = static_cast<double>(ctx.Attr<float>("epsilon"));
    const float momentum = ctx.Attr<float>("momentum");
    const bool is_test = ctx.Attr<bool>("is_test");
    const bool use_global_stats = ctx.Attr<bool>("use_global_stats");

    auto *x = ctx.Input<Tensor>("X");
    auto &x_dims = x->dims();
    PADDLE_ENFORCE(x_dims.size() >= 2 && x_dims.size() <= 5,
                   "The Input dim size should be between 2 and 5");
    int N, C, H, W, D;
    ExtractNCWHD(x_dims, DataLayout::kNCHW, &N, &C, &H, &W, &D);
    int NxC = N * C;
    Tensor x_tmp;
    x_tmp.ShareDataWith(*x).Resize({1, NxC, H, W, D});

    auto *y = ctx.Output<Tensor>("Y");
    y->mutable_data<T>(ctx.GetPlace());

    cudnnTensorDescriptor_t data_desc_;
    cudnnTensorDescriptor_t in_param_desc_;

    CUDNN_ENFORCE(platform::dynload::cudnnCreateTensorDescriptor(&data_desc_));
    CUDNN_ENFORCE(
        platform::dynload::cudnnCreateTensorDescriptor(&in_param_desc_));

    if (epsilon <= CUDNN_BN_MIN_EPSILON - FLT_EPSILON) {
      LOG(ERROR) << "Provided epsilon is smaller than "
                 << "CUDNN_BN_MIN_EPSILON. Setting it to "
                 << "CUDNN_BN_MIN_EPSILON instead.";
    }
    epsilon = std::max(epsilon, CUDNN_BN_MIN_EPSILON);

    VLOG(3) << "Setting descriptors.";
    std::vector<int> dims;
    std::vector<int> strides;
    dims = {1, NxC, H, W, D};
    strides = {NxC * H * W * D, H * W * D, W * D, D, 1};

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();

    CUDNN_ENFORCE(platform::dynload::cudnnSetTensorNdDescriptor(
        data_desc_, CudnnDataType<T>::type,
        x_dims.size() > 3 ? x_dims.size() : 4, dims.data(), strides.data()));
    CUDNN_ENFORCE(platform::dynload::cudnnDeriveBNTensorDescriptor(
        in_param_desc_, data_desc_, CUDNN_BATCHNORM_SPATIAL));

    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *bias = ctx.Input<Tensor>("Bias");

    Tensor scale_tmp =
        ctx.AllocateTmpTensor<T, platform::CUDADeviceContext>({NxC}, dev_ctx);
    scale_tmp.mutable_data<T>(ctx.GetPlace());
    Tensor bias_tmp =
        ctx.AllocateTmpTensor<T, platform::CUDADeviceContext>({NxC}, dev_ctx);
    bias_tmp.mutable_data<T>(ctx.GetPlace());

    const int n = x->numel();
    const int block = 512;
    int max_threads = dev_ctx.GetMaxPhysicalThreadCount();
    const int max_blocks = std::max(max_threads / block, 1);
    const int grid = std::min(NxC, max_blocks);

    repeat_param<T><<<grid, block, 0, dev_ctx.stream()>>>(
        scale->data<T>(), scale_tmp.data<T>(), N, C);
    repeat_param<T><<<grid, block, 0, dev_ctx.stream()>>>(
        bias->data<T>(), bias_tmp.data<T>(), N, C);

    auto handle = dev_ctx.cudnn_handle();
    math::SetConstant<platform::CUDADeviceContext, BatchNormParamType<T>>
        functor;

    if (is_test || use_global_stats) {
      const auto *est_mean = ctx.Input<Tensor>("Mean");
      const auto *est_var = ctx.Input<Tensor>("Variance");

      PADDLE_ENFORCE_EQ(est_mean->dims().size(), 1UL);
      PADDLE_ENFORCE_EQ(est_var->dims().size(), 1UL);
      PADDLE_ENFORCE_EQ(est_mean->dims()[0], NxC);
      PADDLE_ENFORCE_EQ(est_var->dims()[0], NxC);

      CUDNN_ENFORCE(platform::dynload::cudnnBatchNormalizationForwardInference(
          handle, CUDNN_BATCHNORM_SPATIAL, CudnnDataType<T>::kOne(),
          CudnnDataType<T>::kZero(), data_desc_, x_tmp.template data<T>(),
          data_desc_, y->template mutable_data<T>(ctx.GetPlace()),
          in_param_desc_, scale_tmp.template data<BatchNormParamType<T>>(),
          bias_tmp.template data<BatchNormParamType<T>>(),
          est_mean->template data<BatchNormParamType<T>>(),
          est_var->template data<BatchNormParamType<T>>(), epsilon));
    } else {
      auto *mean_out = ctx.Output<Tensor>("MeanOut");
      auto *variance_out = ctx.Output<Tensor>("VarianceOut");
      mean_out->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());
      variance_out->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());

      auto *saved_mean = ctx.Output<Tensor>("SavedMean");
      auto *saved_variance = ctx.Output<Tensor>("SavedVariance");
      saved_mean->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());
      saved_variance->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());
      functor(dev_ctx, saved_mean, static_cast<BatchNormParamType<T>>(0));
      functor(dev_ctx, saved_variance, static_cast<BatchNormParamType<T>>(0));

      double factor = 1. - momentum;
      CUDNN_ENFORCE(platform::dynload::cudnnBatchNormalizationForwardTraining(
          handle, CUDNN_BATCHNORM_SPATIAL, CudnnDataType<T>::kOne(),
          CudnnDataType<T>::kZero(), data_desc_, x_tmp.template data<T>(),
          data_desc_, y->template mutable_data<T>(ctx.GetPlace()),
          in_param_desc_, scale_tmp.template data<BatchNormParamType<T>>(),
          bias_tmp.template data<BatchNormParamType<T>>(), factor,
          mean_out->template mutable_data<BatchNormParamType<T>>(
              ctx.GetPlace()),
          variance_out->template mutable_data<BatchNormParamType<T>>(
              ctx.GetPlace()),
          epsilon, saved_mean->template mutable_data<BatchNormParamType<T>>(
                       ctx.GetPlace()),
          saved_variance->template mutable_data<BatchNormParamType<T>>(
              ctx.GetPlace())));
    }

    CUDNN_ENFORCE(platform::dynload::cudnnDestroyTensorDescriptor(data_desc_));
    CUDNN_ENFORCE(
        platform::dynload::cudnnDestroyTensorDescriptor(in_param_desc_));
  }
};

template <typename T>
static __global__ void INBwdData(const T *dy,
                                 const BatchNormParamType<T> *scale,
                                 const BatchNormParamType<T> *variance,
                                 const double epsilon, const int NxC,
                                 const int C, const int HxW, const int num,
                                 T *dx) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num;
       i += blockDim.x * gridDim.x) {
    const int nc = i / HxW % NxC;
    const int c = nc % C;
    BatchNormParamType<T> inv_var = 1.0 / sqrt(variance[nc] + epsilon);
    dx[i] = static_cast<T>(static_cast<BatchNormParamType<T>>(dy[i]) *
                           scale[c] * inv_var);
  }
}

template <typename T, int BlockDim>
static __global__ void INBwdScaleBias(const T *dy, const T *x,
                                      const BatchNormParamType<T> *mean,
                                      const BatchNormParamType<T> *variance,
                                      const double epsilon, const int N,
                                      const int C, const int HxW,
                                      BatchNormParamType<T> *d_scale,
                                      BatchNormParamType<T> *d_bias) {
  typedef cub::BlockReduce<BatchNormParamType<T>, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage ds_storage;
  __shared__ typename BlockReduce::TempStorage db_storage;

  const int inner_size = N * HxW;
  for (int i = blockIdx.x; i < C; i += gridDim.x) {
    BatchNormParamType<T> ds_sum = static_cast<BatchNormParamType<T>>(0);
    BatchNormParamType<T> db_sum = static_cast<BatchNormParamType<T>>(0);

    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int stats_index = j / HxW * C + i;
      const int index = stats_index * HxW + j % HxW;

      BatchNormParamType<T> var_i = 1.0 / sqrt(variance[stats_index] + epsilon);
      ds_sum +=
          static_cast<BatchNormParamType<T>>(dy[index]) *
          static_cast<BatchNormParamType<T>>(x[index] - mean[stats_index]) *
          static_cast<BatchNormParamType<T>>(var_i);
      db_sum += static_cast<BatchNormParamType<T>>(dy[index]);
    }
    ds_sum = BlockReduce(ds_storage).Reduce(ds_sum, cub::Sum());
    db_sum = BlockReduce(db_storage).Reduce(db_sum, cub::Sum());
    if (threadIdx.x == 0) {
      d_scale[i] = ds_sum;
      d_bias[i] = db_sum;
    }
    __syncthreads();
  }
}

template <typename T>
class InstanceNormGradKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "It must use CUDAPlace.");
    double epsilon = static_cast<double>(ctx.Attr<float>("epsilon"));
    const bool use_global_stats = ctx.Attr<bool>("use_global_stats");
    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *saved_mean = ctx.Input<Tensor>("SavedMean");
    const auto *saved_variance = ctx.Input<Tensor>("SavedVariance");
    const auto *x = ctx.Input<Tensor>("X");
    const auto *d_y = ctx.Input<Tensor>(framework::GradVarName("Y"));

    const auto &x_dims = x->dims();

    PADDLE_ENFORCE(x_dims.size() >= 2 && x_dims.size() <= 5,
                   "The Input dim size should be between 2 and 5");
    int N, C, H, W, D;
    ExtractNCWHD(x_dims, DataLayout::kNCHW, &N, &C, &H, &W, &D);
    int NxC = N * C;

    Tensor x_tmp, d_y_tmp;
    x_tmp.ShareDataWith(*x).Resize({1, NxC, H, W, D});
    d_y_tmp.ShareDataWith(*d_y).Resize({1, NxC, H, W, D});

    auto *d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *d_scale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto *d_bias = ctx.Output<Tensor>(framework::GradVarName("Bias"));

    d_x->mutable_data<T>(ctx.GetPlace());
    if (d_scale && d_bias) {
      d_scale->mutable_data<T>(ctx.GetPlace());
      d_bias->mutable_data<T>(ctx.GetPlace());
    }
    PADDLE_ENFORCE(scale->dims().size(), 1UL);
    PADDLE_ENFORCE(scale->dims()[0], C);

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();

    const int n = x->numel();
    const int block = 512;
    int max_threads = dev_ctx.GetMaxPhysicalThreadCount();
    const int max_blocks = std::max(max_threads / block, 1);
    const int grid = std::min(NxC, max_blocks);
    const int grid1 = (n + block - 1) / block;

    Tensor scale_tmp =
        ctx.AllocateTmpTensor<T, platform::CUDADeviceContext>({NxC}, dev_ctx);
    scale_tmp.mutable_data<T>(ctx.GetPlace());
    Tensor d_scale_tmp =
        ctx.AllocateTmpTensor<T, platform::CUDADeviceContext>({NxC}, dev_ctx);
    Tensor d_bias_tmp =
        ctx.AllocateTmpTensor<T, platform::CUDADeviceContext>({NxC}, dev_ctx);
    repeat_param<T><<<grid, block, 0, dev_ctx.stream()>>>(
        scale->data<T>(), scale_tmp.data<T>(), N, C);

    std::vector<int> dims;
    std::vector<int> strides;
    dims = {1, NxC, H, W, D};
    strides = {NxC * H * W * D, H * W * D, W * D, D, 1};

    if (!use_global_stats) {
      if ((H * W * D) == 1) {
        framework::TensorCopy(*d_y, ctx.GetPlace(), d_x);
        math::SetConstant<platform::CUDADeviceContext, BatchNormParamType<T>>
            functor;
        functor(dev_ctx, d_scale, static_cast<BatchNormParamType<T>>(0));
        functor(dev_ctx, d_bias, static_cast<BatchNormParamType<T>>(0));
        return;
      }

      cudnnTensorDescriptor_t data_desc_;
      cudnnTensorDescriptor_t in_param_desc_;

      CUDNN_ENFORCE(
          platform::dynload::cudnnCreateTensorDescriptor(&data_desc_));
      CUDNN_ENFORCE(
          platform::dynload::cudnnCreateTensorDescriptor(&in_param_desc_));
      if (epsilon <= CUDNN_BN_MIN_EPSILON - FLT_EPSILON) {
        LOG(ERROR) << "Provided epsilon is smaller than "
                   << "CUDNN_BN_MIN_EPSILON. Setting it to "
                   << "CUDNN_BN_MIN_EPSILON instead.";
      }
      epsilon = std::max(epsilon, CUDNN_BN_MIN_EPSILON);

      CUDNN_ENFORCE(platform::dynload::cudnnSetTensorNdDescriptor(
          data_desc_, CudnnDataType<T>::type,
          x_dims.size() > 3 ? x_dims.size() : 4, dims.data(), strides.data()));
      CUDNN_ENFORCE(platform::dynload::cudnnDeriveBNTensorDescriptor(
          in_param_desc_, data_desc_, CUDNN_BATCHNORM_SPATIAL));

      const auto *saved_mean = ctx.Input<Tensor>("SavedMean");
      const auto *saved_var = ctx.Input<Tensor>("SavedVariance");
      const void *saved_mean_data =
          saved_mean->template data<BatchNormParamType<T>>();
      const void *saved_var_data =
          saved_var->template data<BatchNormParamType<T>>();
      CUDNN_ENFORCE(platform::dynload::cudnnBatchNormalizationBackward(
          dev_ctx.cudnn_handle(), CUDNN_BATCHNORM_SPATIAL,
          CudnnDataType<T>::kOne(), CudnnDataType<T>::kZero(),
          CudnnDataType<T>::kOne(), CudnnDataType<T>::kZero(), data_desc_,
          x_tmp.template data<T>(), data_desc_, d_y_tmp.template data<T>(),
          data_desc_, d_x->template mutable_data<T>(ctx.GetPlace()),
          in_param_desc_, scale_tmp.template data<BatchNormParamType<T>>(),
          d_scale_tmp.template mutable_data<BatchNormParamType<T>>(
              ctx.GetPlace()),
          d_bias_tmp.template mutable_data<BatchNormParamType<T>>(
              ctx.GetPlace()),
          epsilon, saved_mean_data, saved_var_data));

      add_param<T, block><<<grid, block, 0, dev_ctx.stream()>>>(
          d_scale_tmp.data<T>(), d_scale->data<T>(), N, C);
      add_param<T, block><<<grid, block, 0, dev_ctx.stream()>>>(
          d_bias_tmp.data<T>(), d_bias->data<T>(), N, C);
      CUDNN_ENFORCE(
          platform::dynload::cudnnDestroyTensorDescriptor(data_desc_));
      CUDNN_ENFORCE(
          platform::dynload::cudnnDestroyTensorDescriptor(in_param_desc_));
    } else {
      const auto *running_mean = ctx.Input<Tensor>("Mean");
      const auto *running_variance = ctx.Input<Tensor>("Variance");

      const auto *running_mean_data =
          running_mean->template data<BatchNormParamType<T>>();
      const auto *running_var_data =
          running_variance->template data<BatchNormParamType<T>>();
      if (d_x) {
        INBwdData<T><<<grid1, block, 0, dev_ctx.stream()>>>(
            d_y_tmp.data<T>(), scale->data<BatchNormParamType<T>>(),
            running_var_data, epsilon, NxC, C, H * W, n, d_x->data<T>());
      }
      if (d_scale && d_bias) {
        INBwdScaleBias<T, block><<<grid, block, 0, dev_ctx.stream()>>>(
            d_y_tmp.data<T>(), x_tmp.data<T>(), running_mean_data,
            running_var_data, epsilon, N, C, H * W,
            d_scale->data<BatchNormParamType<T>>(),
            d_bias->data<BatchNormParamType<T>>());
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(
    instance_norm, ops::InstanceNormKernel<plat::CUDADeviceContext, float>,
    ops::InstanceNormKernel<plat::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    instance_norm_grad,
    ops::InstanceNormGradKernel<plat::CUDADeviceContext, float>,
    ops::InstanceNormGradKernel<plat::CUDADeviceContext, double>);
// REGISTER_OP_CUDA_KERNEL(
//    instance_norm_grad_grad,
//    ops::InstanceNormDoubleGradKernel<paddle::platform::CUDADeviceContext,
//    float>,
//    ops::InstanceNormDoubleGradKernel<paddle::platform::CUDADeviceContext,
//    double>);
