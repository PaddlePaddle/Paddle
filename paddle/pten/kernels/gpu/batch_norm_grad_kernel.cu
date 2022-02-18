// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/pten/backends/gpu/gpu_context.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/kernels/batch_norm_kernel.h"
#include "paddle/pten/kernels/funcs/eigen/common.h"

#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"

#include "paddle/fluid/operators/norm_utils.cu.h"
#include "paddle/fluid/operators/norm_utils.h"

#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/operators/layout_utils.h"
#include "paddle/fluid/platform/enforce.h"

#include "paddle/fluid/platform/flags.h"
#include "paddle/pten/kernels/gpu/batch_norm_utils.h"

#ifdef __HIPCC__
#define LAUNCH_BOUNDS(BlockDim) __launch_bounds__(BlockDim)
#else
#define LAUNCH_BOUNDS(BlockDim)
#endif

DECLARE_bool(cudnn_batchnorm_spatial_persistent);
namespace pten {

template <typename T>
using CudnnDataType = paddle::platform::CudnnDataType<T>;
template <typename T>
using BatchNormParamType = typename CudnnDataType<T>::BatchNormParamType;

template <typename T, int BlockDim, paddle::framework::DataLayout layout>
static __global__ LAUNCH_BOUNDS(BlockDim) void KeBNBackwardScaleBias(
    const T *dy,
    const T *x,
    const BatchNormParamType<T> *mean,
    const BatchNormParamType<T> *variance,
    const double epsilon,
    const int N,
    const int C,
    const int HxW,
    BatchNormParamType<T> *dscale,
    BatchNormParamType<T> *dbias) {
  const int outer_size = C;
  const int inner_size = N * HxW;
  typedef cub::BlockReduce<BatchNormParamType<T>, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage ds_storage;
  __shared__ typename BlockReduce::TempStorage db_storage;

  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    BatchNormParamType<T> ds_sum = static_cast<BatchNormParamType<T>>(0);
    BatchNormParamType<T> db_sum = static_cast<BatchNormParamType<T>>(0);

    BatchNormParamType<T> inv_var_i = 1.0 / sqrt(variance[i] + epsilon);
    BatchNormParamType<T> mean_i = mean[i];
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int index = layout == paddle::framework::DataLayout::kNCHW
                            ? (j / HxW * C + i) * HxW + j % HxW
                            : j * outer_size + i;
      ds_sum += static_cast<BatchNormParamType<T>>(dy[index]) *
                (static_cast<BatchNormParamType<T>>(x[index]) - mean_i);
      db_sum += static_cast<BatchNormParamType<T>>(dy[index]);
    }
    ds_sum = BlockReduce(ds_storage).Reduce(ds_sum, cub::Sum());
    db_sum = BlockReduce(db_storage).Reduce(db_sum, cub::Sum());
    if (threadIdx.x == 0) {
      dscale[i] = ds_sum * inv_var_i;
      dbias[i] = db_sum;
    }
    __syncthreads();
  }
}

template <typename T, paddle::framework::DataLayout layout>
static __global__ void KeBNBackwardData(const T *dy,
                                        const BatchNormParamType<T> *scale,
                                        const BatchNormParamType<T> *variance,
                                        const double epsilon,
                                        const int C,
                                        const int HxW,
                                        const int num,
                                        T *dx) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = gid; i < num; i += stride) {
    const int c =
        layout == paddle::framework::DataLayout::kNCHW ? i / HxW % C : i % C;
    BatchNormParamType<T> inv_var = 1.0 / sqrt(variance[c] + epsilon);
    dx[i] = static_cast<T>(static_cast<BatchNormParamType<T>>(dy[i]) *
                           scale[c] * inv_var);
  }
}

template <typename T>
static __global__ void KeBNRestoreData(
    const paddle::framework::DataLayout layout,
    T *x,
    const BatchNormParamType<T> *scale,
    const BatchNormParamType<T> *bias,
    const BatchNormParamType<T> *mean,
    const BatchNormParamType<T> *variance,
    double epsilon,
    int C,
    int M,
    const int num,
    const T *y) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = gid; i < num; i += stride) {
    const int c =
        layout == paddle::framework::DataLayout::kNCHW ? (i / M) % C : i % C;
    auto y_i = static_cast<BatchNormParamType<T>>(y[i]);
    auto x_i = (y_i - bias[c]) / scale[c] / variance[c] + mean[c];
    x[i] = static_cast<T>(x_i);
  }
}

template <typename T>
class InplaceHelper {
 public:
  void operator()(const paddle::framework::DataLayout layout,
                  T *x,
                  const BatchNormParamType<T> *scale,
                  const BatchNormParamType<T> *bias,
                  const BatchNormParamType<T> *mean,
                  const BatchNormParamType<T> *variance,
                  double epsilon,
                  int C,
                  int M,
                  const int num,
                  const T *y,
                  int grid2,
                  const int block,
                  const gpuStream_t &stream) {
    PADDLE_ENFORCE_EQ(x,
                      y,
                      pten::errors::InvalidArgument(
                          "X and Y should be inplaced in inplace mode"));
    KeBNRestoreData<<<grid2, block, 0, stream>>>(
        layout, x, scale, bias, mean, variance, epsilon, C, M, num, y);
  }
};

template <typename T, int BlockDim, paddle::framework::DataLayout layout>
static __global__ LAUNCH_BOUNDS(BlockDim) void BNBackward(
    const T *dy,
    const T *x,
    const BatchNormParamType<T> *scale,
    const BatchNormParamType<T> *saved_mean,
    const BatchNormParamType<T> *saved_inv_variance,
    const int C,
    const int N,
    const int HxW,
    const double epsilon,
    T *dx,
    BatchNormParamType<T> *dscale,
    BatchNormParamType<T> *dbias) {
  const int outer_size = C;
  const int inner_size = N * HxW;
  typedef cub::BlockReduce<BatchNormParamType<T>, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage ds_storage;
  __shared__ typename BlockReduce::TempStorage db_storage;
  __shared__ typename BlockReduce::TempStorage mean_storage;
  __shared__ typename BlockReduce::TempStorage variance_storeage;
  __shared__ BatchNormParamType<T> inv_var_val;
  __shared__ BatchNormParamType<T> mean_val;
  __shared__ BatchNormParamType<T> dscale_val;
  __shared__ BatchNormParamType<T> dbias_val;

  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    BatchNormParamType<T> ds_sum = static_cast<BatchNormParamType<T>>(0);
    BatchNormParamType<T> db_sum = static_cast<BatchNormParamType<T>>(0);

    if (saved_mean && saved_inv_variance) {
      if (threadIdx.x == 0) {
        inv_var_val = saved_inv_variance[i];
        mean_val = saved_mean[i];
      }
    } else {
      BatchNormParamType<T> x_sum = static_cast<BatchNormParamType<T>>(0);
      BatchNormParamType<T> x_square_sum =
          static_cast<BatchNormParamType<T>>(0);

      for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
        const int index = layout == paddle::framework::DataLayout::kNCHW
                              ? (j / HxW * C + i) * HxW + j % HxW
                              : j * outer_size + i;
        BatchNormParamType<T> x_i =
            static_cast<BatchNormParamType<T>>(x[index]);
        x_sum += x_i;
        x_square_sum += x_i * x_i;
      }
      x_sum = BlockReduce(mean_storage).Reduce(x_sum, cub::Sum());
      x_square_sum =
          BlockReduce(variance_storeage).Reduce(x_square_sum, cub::Sum());
      if (threadIdx.x == 0) {
        mean_val = x_sum / inner_size;
        inv_var_val =
            1 / sqrt(x_square_sum / inner_size - mean_val * mean_val + epsilon);
      }
    }
    __syncthreads();

    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int index = layout == paddle::framework::DataLayout::kNCHW
                            ? (j / HxW * C + i) * HxW + j % HxW
                            : j * outer_size + i;
      BatchNormParamType<T> dy_i =
          static_cast<BatchNormParamType<T>>(dy[index]);
      ds_sum +=
          dy_i * (static_cast<BatchNormParamType<T>>(x[index]) - mean_val);
      db_sum += dy_i;
    }
    ds_sum = BlockReduce(ds_storage).Reduce(ds_sum, cub::Sum());
    db_sum = BlockReduce(db_storage).Reduce(db_sum, cub::Sum());
    if (threadIdx.x == 0) {
      dscale_val = ds_sum * inv_var_val;
      dbias_val = db_sum;
      dscale[i] = dscale_val;
      dbias[i] = dbias_val;
    }
    __syncthreads();

    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int index = layout == paddle::framework::DataLayout::kNCHW
                            ? (j / HxW * C + i) * HxW + j % HxW
                            : j * outer_size + i;
      dx[index] = scale[i] * inv_var_val *
                  (static_cast<BatchNormParamType<T>>(dy[index]) -
                   dbias_val / static_cast<BatchNormParamType<T>>(inner_size) -
                   (static_cast<BatchNormParamType<T>>(x[index]) - mean_val) *
                       inv_var_val * dscale_val / inner_size);
    }
  }
}

template <typename T, int BlockDim, paddle::framework::DataLayout layout>
static __global__ LAUNCH_BOUNDS(BlockDim) void BNBackwardData(
    const T *dy,
    const BatchNormParamType<T> *scale,
    const BatchNormParamType<T> *mean,
    const T *x,
    const BatchNormParamType<T> *variance,
    const int C,
    const int N,
    const int HxW,
    T *dx) {
  const int outer_size = C;
  const int inner_size = N * HxW;
  typedef cub::BlockReduce<BatchNormParamType<T>, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage dy_storage;
  __shared__ typename BlockReduce::TempStorage dy_x_sub_mean_storage;
  __shared__ BatchNormParamType<T> dy_sum_val;
  __shared__ BatchNormParamType<T> dy_x_sub_mean_sum_val;

  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    BatchNormParamType<T> inv_var_i = variance[i];
    BatchNormParamType<T> mean_i = mean[i];
    BatchNormParamType<T> dy_sum = static_cast<BatchNormParamType<T>>(0);
    BatchNormParamType<T> dy_x_sub_mean_sum =
        static_cast<BatchNormParamType<T>>(0);
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int index = layout == paddle::framework::DataLayout::kNCHW
                            ? (j / HxW * C + i) * HxW + j % HxW
                            : j * outer_size + i;
      BatchNormParamType<T> dy_i =
          static_cast<BatchNormParamType<T>>(dy[index]);
      dy_sum += dy_i;
      dy_x_sub_mean_sum +=
          dy_i * (static_cast<BatchNormParamType<T>>(x[index]) - mean_i);
    }

    dy_sum = BlockReduce(dy_storage).Reduce(dy_sum, cub::Sum());
    dy_x_sub_mean_sum = BlockReduce(dy_x_sub_mean_storage)
                            .Reduce(dy_x_sub_mean_sum, cub::Sum());

    if (threadIdx.x == 0) {
      dy_sum_val = dy_sum;
      dy_x_sub_mean_sum_val = dy_x_sub_mean_sum;
    }
    __syncthreads();
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int index = layout == paddle::framework::DataLayout::kNCHW
                            ? (j / HxW * C + i) * HxW + j % HxW
                            : j * outer_size + i;
      dx[index] =
          (static_cast<BatchNormParamType<T>>(dy[index]) -
           dy_sum_val / static_cast<BatchNormParamType<T>>(inner_size) -
           (static_cast<BatchNormParamType<T>>(x[index]) - mean_i) *
               dy_x_sub_mean_sum_val * inv_var_i * inv_var_i / inner_size) *
          scale[i] * inv_var_i;
    }
  }
}

template <typename T, typename Context>
void BatchNormGradRawKernel(const Context &ctx,
                            const DenseTensor &y_grad,
                            const DenseTensor &x,
                            const DenseTensor &scale,
                            const DenseTensor &bias,
                            const DenseTensor &saved_mean,
                            const DenseTensor &saved_variance,
                            paddle::optional<const DenseTensor &> reserve_space,
                            paddle::optional<const DenseTensor &> mean,
                            paddle::optional<const DenseTensor &> variance,
                            float momentum,
                            float epsilon_f,
                            const std::string &data_layout_str,
                            bool is_test,
                            bool use_global_stats,
                            bool trainable_statistics,
                            bool fuse_with_relu,
                            bool is_inplace,
                            DenseTensor *x_grad,
                            DenseTensor *scale_grad,
                            DenseTensor *bias_grad) {
  double epsilon = static_cast<double>(epsilon_f);

  const DataLayout data_layout =
      paddle::framework::StringToDataLayout(data_layout_str);

  const auto *d_y = &y_grad;

  auto *d_x = x_grad;
  auto *d_scale = scale_grad;
  auto *d_bias = bias_grad;

  use_global_stats = is_test || use_global_stats;

  const auto &x_dims = x.dims();

  PADDLE_ENFORCE_EQ(
      x_dims.size() >= 2 && x_dims.size() <= 5,
      true,
      pten::errors::InvalidArgument(
          "The size of input's dimensions should be between 2 and 5."
          "But received: the size of input's dimensions is [%d],"
          "the dimensions of input is [%s]",
          x_dims.size(),
          x_dims));
  int N, C, H, W, D;
  paddle::operators::ExtractNCWHD(x_dims, data_layout, &N, &C, &H, &W, &D);

  // init output
  if (d_x) {
    d_x->mutable_data<T>(ctx.GetPlace());
  }

  if (d_scale && d_bias) {
    d_scale->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());
    d_bias->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());
  }

  PADDLE_ENFORCE_EQ(
      scale.dims().size(),
      1UL,
      pten::errors::InvalidArgument(
          "The size of scale's dimensions must equal to 1. But received: "
          "the size of scale's dimensions is [%d], the dimensions of scale "
          "is [%s].",
          scale.dims().size(),
          scale.dims()));
  PADDLE_ENFORCE_EQ(
      scale.dims()[0],
      C,
      pten::errors::InvalidArgument(
          "The first dimension of scale must equal to Channels[%d]. But "
          "received: the first dimension of scale is [%d]",
          C,
          scale.dims()[0]));

  auto dtype = paddle::platform::CudnnDataType<T>::type;
#ifdef PADDLE_WITH_HIP
  auto compute_format =
      data_layout == DataLayout::kNHWC ? DataLayout::kNHWC : DataLayout::kNCHW;

// TODO(wangran16): wait for MIOpen to improve the performance of BN
// HIP do not support compute format of NHWC
// auto compute_format = DataLayout::kNCHW;
#else
  const bool fast_nhwc_batch_norm = dtype == CUDNN_DATA_HALF &&
                                    FLAGS_cudnn_batchnorm_spatial_persistent &&
                                    (reserve_space.get_ptr() != nullptr);
  auto compute_format = fast_nhwc_batch_norm && data_layout == DataLayout::kNHWC
                            ? DataLayout::kNHWC
                            : DataLayout::kNCHW;
#endif

  DenseTensor transformed_x(x.type());
  DenseTensor transformed_d_y(d_y->type());
  DenseTensor transformed_d_x;
  if (data_layout == DataLayout::kNHWC && compute_format == DataLayout::kNCHW &&
      x_dims.size() > 2) {
    VLOG(3) << "Transform input tensor from NHWC to NCHW.";
    ResizeToChannelFirst<Context, T>(ctx, &x, &transformed_x);
    TransToChannelFirst<Context, T>(ctx, &x, &transformed_x);
    ResizeToChannelFirst<Context, T>(ctx, d_y, &transformed_d_y);
    TransToChannelFirst<Context, T>(ctx, d_y, &transformed_d_y);
    if (d_x) {
      ResizeToChannelFirst<Context, T>(ctx, d_x, &transformed_d_x);
    }
  } else {
    transformed_x.ShareDataWith(x);
    transformed_d_y.ShareDataWith(*d_y);
    if (d_x) {
      transformed_d_x.ShareDataWith(*d_x);
    }
  }

  std::vector<int> dims;
  std::vector<int> strides;
  if (compute_format == DataLayout::kNCHW) {
    dims = {N, C, H, W, D};
    strides = {C * H * W * D, H * W * D, W * D, D, 1};
  } else {
    dims = {N, C, H, W, D};
    strides = {H * W * C * D, 1, W * D * C, D * C, C};
  }

  const int num = transformed_x.numel();
#ifdef HIPCC
  const int block = 256;
#else
  const int block = 512;
#endif
  int max_threads = ctx.GetMaxPhysicalThreadCount();
  const int max_blocks = std::max(max_threads / block, 1);
  int grid1 = (num + block - 1) / block;
  int grid2 = std::min(C, max_blocks);
  auto stream = ctx.stream();
  InplaceHelper<T> inplace_functor;

  if (!use_global_stats) {
    if ((N * H * W * D) == 1) {
      if (d_x) {
        paddle::framework::TensorCopy(*d_y, ctx.GetPlace(), d_x);
      }
      pten::funcs::SetConstant<Context, BatchNormParamType<T>> functor;
      functor(ctx, d_scale, static_cast<BatchNormParamType<T>>(0));
      functor(ctx, d_bias, static_cast<BatchNormParamType<T>>(0));
      return;
    }

// ------------------- cudnn descriptors ---------------------
#ifdef PADDLE_WITH_HIP
// TODO(wangran16): wait for MIOpen to improve the performance of BN
// miopenTensorDescriptor_t data_desc_;
// miopenTensorDescriptor_t bn_param_desc_;
// miopenBatchNormMode_t mode_;

// PADDLE_ENFORCE_GPU_SUCCESS(
//     platform::dynload::miopenCreateTensorDescriptor(&data_desc_));
// PADDLE_ENFORCE_GPU_SUCCESS(
//     platform::dynload::miopenCreateTensorDescriptor(&bn_param_desc_));
#else
    cudnnTensorDescriptor_t data_desc_;
    cudnnTensorDescriptor_t bn_param_desc_;
    cudnnBatchNormMode_t mode_;

    PADDLE_ENFORCE_GPU_SUCCESS(
        paddle::platform::dynload::cudnnCreateTensorDescriptor(&data_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        paddle::platform::dynload::cudnnCreateTensorDescriptor(
            &bn_param_desc_));
#endif
    if (epsilon <= CUDNN_BN_MIN_EPSILON - FLT_EPSILON) {
      LOG(ERROR) << "Provided epsilon is smaller than "
                 << "CUDNN_BN_MIN_EPSILON. Setting it to "
                 << "CUDNN_BN_MIN_EPSILON instead.";
    }
    epsilon = std::max(epsilon, CUDNN_BN_MIN_EPSILON);
#ifdef PADDLE_WITH_HIP
// TODO(wangran16): wait for MIOpen to improve the performance of BN
// mode_ = miopenBNSpatial;
#elif CUDNN_VERSION_MIN(7, 0, 1)
    if (FLAGS_cudnn_batchnorm_spatial_persistent) {
      mode_ = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
    } else if (H == 1 && W == 1) {
      mode_ = CUDNN_BATCHNORM_PER_ACTIVATION;
    } else {
      mode_ = CUDNN_BATCHNORM_SPATIAL;
    }
#else
    if (H == 1 && W == 1) {
      mode_ = CUDNN_BATCHNORM_PER_ACTIVATION;
    } else {
      mode_ = CUDNN_BATCHNORM_SPATIAL;
    }
#endif  // CUDNN_VERSION_MIN(7, 0, 1)

#ifdef PADDLE_WITH_HIP
// TODO(wangran16): wait for MIOpen to improve the performance of BN
// PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::miopenSetTensorDescriptor(
//     data_desc_, CudnnDataType<T>::type,
//     x_dims.size() > 3 ? x_dims.size() : 4, const_cast<int *>(dims.data()),
//     const_cast<int *>(strides.data())));
// PADDLE_ENFORCE_GPU_SUCCESS(
//     platform::dynload::miopenDeriveBNTensorDescriptor(bn_param_desc_,
//                                                       data_desc_, mode_));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(
        paddle::platform::dynload::cudnnSetTensorNdDescriptor(
            data_desc_,
            CudnnDataType<T>::type,
            x_dims.size() > 3 ? x_dims.size() : 4,
            dims.data(),
            strides.data()));
    PADDLE_ENFORCE_GPU_SUCCESS(
        paddle::platform::dynload::cudnnDeriveBNTensorDescriptor(
            bn_param_desc_, data_desc_, mode_));
#endif

    const auto *saved_mean_data =
        saved_mean.template data<BatchNormParamType<T>>();
    const auto *saved_var_data =
        saved_variance.template data<BatchNormParamType<T>>();

    if (is_inplace) {
      inplace_functor(compute_format,
                      transformed_x.data<T>(),
                      scale.template data<BatchNormParamType<T>>(),
                      bias.template data<BatchNormParamType<T>>(),
                      saved_mean_data,
                      saved_var_data,
                      epsilon,
                      C,
                      H * W * D,
                      num,
                      transformed_x.data<T>(),
                      grid2,
                      block,
                      stream);
    }

    // This branch calls CUDNN APIs
    if (d_x && d_scale && d_bias) {
      bool called = false;
#if CUDNN_VERSION_MIN(7, 4, 1)
      called = true;
      size_t workspace_size = 0;
      void *workspace_ptr = nullptr;
      DenseTensor workspace_tensor;
      auto reserve_space_size = reserve_space->memory_size();
      // --------------- cudnn batchnorm workspace ---------------
      PADDLE_ENFORCE_GPU_SUCCESS(
          paddle::platform::dynload::
              cudnnGetBatchNormalizationBackwardExWorkspaceSize(
                  /*handle=*/ctx.cudnn_handle(),
                  /*mode=*/mode_,
                  /*bnIps=*/CUDNN_BATCHNORM_OPS_BN,
                  /*xDesc=*/data_desc_,
                  /*yDesc=*/data_desc_,
                  /*dyDesc=*/data_desc_,
                  /*dzDesc=*/nullptr,
                  /*dxDesc=*/data_desc_,
                  /*bnScaleBiasMeanVarDesc=*/bn_param_desc_,
                  /*activationDesc=*/nullptr,
                  /*sizeInBytes=*/&workspace_size));

      workspace_ptr = workspace_tensor.mutable_data(
          ctx.GetPlace(), transformed_x.type(), workspace_size);

      PADDLE_ENFORCE_GPU_SUCCESS(
          paddle::platform::dynload::cudnnBatchNormalizationBackwardEx(
              /*handle=*/ctx.cudnn_handle(),
              /*mode=*/mode_,
              /*bnOps=*/CUDNN_BATCHNORM_OPS_BN,
              /*alphaDataDiff=*/CudnnDataType<T>::kOne(),
              /*betaDataDiff=*/CudnnDataType<T>::kZero(),
              /*alphaParamDiff=*/CudnnDataType<T>::kOne(),
              /*betaParamDiff=*/CudnnDataType<T>::kZero(),
              /*xDesc=*/data_desc_,
              /*xData=*/transformed_x.template data<T>(),
              /*yDesc=*/nullptr,
              /*yData=*/nullptr,
              /*dyDesc=*/data_desc_,
              /*dyData=*/transformed_d_y.template data<T>(),
              /*dzDesc=*/nullptr,
              /*dzData=*/nullptr,
              /*dxDesc=*/data_desc_,
              /*dxData=*/transformed_d_x.template mutable_data<T>(
                  ctx.GetPlace()),
              /*dBnScaleBiasDesc=*/bn_param_desc_,
              /*bnScaleData=*/scale.template data<BatchNormParamType<T>>(),
              /*bnBiasData=*/nullptr,
              /*dBnScaleData=*/d_scale
                  ->template mutable_data<BatchNormParamType<T>>(
                      ctx.GetPlace()),
              /*dBnBiasData=*/d_bias
                  ->template mutable_data<BatchNormParamType<T>>(
                      ctx.GetPlace()),
              /*epsilon=*/epsilon,
              /*savedMean=*/saved_mean_data,
              /*savedInvVariance=*/saved_var_data,
              /*activationDesc=*/nullptr,
              /*workspace=*/workspace_ptr,
              /*workSpaceSizeInBytes=*/workspace_size,
              /*reserveSpace=*/const_cast<T *>(
                  reserve_space->template data<T>()),
              /*reserveSpaceSizeInBytes=*/reserve_space_size));
#endif  // CUDNN_VERSION_MIN(7, 4, 1)
      if (!called) {
#ifdef PADDLE_WITH_HIP
        if (compute_format == DataLayout::kNCHW) {
          BNBackward<T,
                     block,
                     DataLayout::kNCHW><<<grid2, block, 0, ctx.stream()>>>(
              transformed_d_y.template data<T>(),
              transformed_x.template data<T>(),
              scale.template data<BatchNormParamType<T>>(),
              saved_mean_data,
              saved_var_data,
              C,
              N,
              H * W * D,
              epsilon,
              transformed_d_x.template data<T>(),
              d_scale->template mutable_data<BatchNormParamType<T>>(
                  ctx.GetPlace()),
              d_bias->template mutable_data<BatchNormParamType<T>>(
                  ctx.GetPlace()));
        } else {
          BNBackward<T,
                     block,
                     DataLayout::kNHWC><<<grid2, block, 0, ctx.stream()>>>(
              transformed_d_y.template data<T>(),
              transformed_x.template data<T>(),
              scale.template data<BatchNormParamType<T>>(),
              saved_mean_data,
              saved_var_data,
              C,
              N,
              H * W * D,
              epsilon,
              transformed_d_x.template data<T>(),
              d_scale->template mutable_data<BatchNormParamType<T>>(
                  ctx.GetPlace()),
              d_bias->template mutable_data<BatchNormParamType<T>>(
                  ctx.GetPlace()));
        }

// TODO(wangran16): wait for MIOpen to improve the performance of BN
// PADDLE_ENFORCE_GPU_SUCCESS(
//     platform::dynload::miopenBatchNormalizationBackward(
//         dev_ctx.cudnn_handle(), mode_, CudnnDataType<T>::kOne(),
//         CudnnDataType<T>::kZero(), CudnnDataType<T>::kOne(),
//         CudnnDataType<T>::kZero(), data_desc_,
//         transformed_x.template data<T>(), data_desc_,
//         transformed_d_y.template data<T>(), data_desc_,
//         transformed_d_x.template mutable_data<T>(ctx.GetPlace()),
//         bn_param_desc_, scale->template data<BatchNormParamType<T>>(),
//         d_scale->template mutable_data<BatchNormParamType<T>>(
//             ctx.GetPlace()),
//         d_bias->template mutable_data<BatchNormParamType<T>>(
//             ctx.GetPlace()),
//         epsilon, saved_mean_data, saved_var_data));
#else
        PADDLE_ENFORCE_GPU_SUCCESS(
            paddle::platform::dynload::cudnnBatchNormalizationBackward(
                ctx.cudnn_handle(),
                mode_,
                CudnnDataType<T>::kOne(),
                CudnnDataType<T>::kZero(),
                CudnnDataType<T>::kOne(),
                CudnnDataType<T>::kZero(),
                data_desc_,
                transformed_x.template data<T>(),
                data_desc_,
                transformed_d_y.template data<T>(),
                data_desc_,
                transformed_d_x.template mutable_data<T>(ctx.GetPlace()),
                bn_param_desc_,
                scale.template data<BatchNormParamType<T>>(),
                d_scale->template mutable_data<BatchNormParamType<T>>(
                    ctx.GetPlace()),
                d_bias->template mutable_data<BatchNormParamType<T>>(
                    ctx.GetPlace()),
                epsilon,
                saved_mean_data,
                saved_var_data));
#endif
      }

      if (data_layout == DataLayout::kNHWC &&
          compute_format == DataLayout::kNCHW) {
        VLOG(3) << "Transform batchnorm output from NCHW to NHWC";
        TransToChannelLast<Context, T>(ctx, &transformed_d_x, d_x);
      }
    } else {
      // This branch call CUDA kernels
      if (compute_format == DataLayout::kNCHW) {
        if (d_x) {
          BNBackwardData<T,
                         block,
                         paddle::framework::DataLayout::
                             kNCHW><<<grid2, block, 0, ctx.stream()>>>(
              d_y->data<T>(),
              scale.data<BatchNormParamType<T>>(),
              saved_mean_data,
              x.data<T>(),
              saved_var_data,
              C,
              N,
              H * W * D,
              d_x->data<T>());
        }
        if (d_scale && d_bias) {
          KeBNBackwardScaleBias<T,
                                block,
                                paddle::framework::DataLayout::
                                    kNCHW><<<grid2, block, 0, stream>>>(
              d_y->data<T>(),
              x.data<T>(),
              saved_mean_data,
              saved_var_data,
              epsilon,
              N,
              C,
              H * W * D,
              d_scale->data<BatchNormParamType<T>>(),
              d_bias->data<BatchNormParamType<T>>());
        }
      } else {
        if (d_x) {
          BNBackwardData<T,
                         block,
                         paddle::framework::DataLayout::
                             kNHWC><<<grid2, block, 0, ctx.stream()>>>(
              d_y->data<T>(),
              scale.data<BatchNormParamType<T>>(),
              saved_mean_data,
              x.data<T>(),
              saved_var_data,
              C,
              N,
              H * W * D,
              d_x->data<T>());
        }
        if (d_scale && d_bias) {
          KeBNBackwardScaleBias<T,
                                block,
                                paddle::framework::DataLayout::
                                    kNHWC><<<grid2, block, 0, stream>>>(
              d_y->data<T>(),
              x.data<T>(),
              saved_mean_data,
              saved_var_data,
              epsilon,
              N,
              C,
              H * W * D,
              d_scale->data<BatchNormParamType<T>>(),
              d_bias->data<BatchNormParamType<T>>());
        }
      }
    }

#ifdef PADDLE_WITH_HIP
// TODO(wangran16): wait for MIOpen to improve the performance of BN
// clean when exit.
// PADDLE_ENFORCE_GPU_SUCCESS(
//     platform::dynload::miopenDestroyTensorDescriptor(data_desc_));
// PADDLE_ENFORCE_GPU_SUCCESS(
//     platform::dynload::miopenDestroyTensorDescriptor(bn_param_desc_));
#else
    // clean when exit.
    PADDLE_ENFORCE_GPU_SUCCESS(
        paddle::platform::dynload::cudnnDestroyTensorDescriptor(data_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        paddle::platform::dynload::cudnnDestroyTensorDescriptor(
            bn_param_desc_));
#endif
  } else {
    const auto *running_mean = mean.get_ptr();
    const auto *running_var = variance.get_ptr();

    const auto *running_mean_data =
        running_mean->template data<BatchNormParamType<T>>();
    const auto *running_var_data =
        running_var->template data<BatchNormParamType<T>>();

    if (is_inplace) {
      auto px = x;
      inplace_functor(data_layout,
                      px.mutable_data<T>(ctx.GetPlace()),
                      scale.template data<BatchNormParamType<T>>(),
                      bias.template data<BatchNormParamType<T>>(),
                      running_mean_data,
                      running_var_data,
                      epsilon,
                      C,
                      H * W * D,
                      num,
                      x.data<T>(),
                      grid2,
                      block,
                      stream);
    }

    if (compute_format == DataLayout::kNCHW) {
      if (d_x) {
        KeBNBackwardData<
            T,
            paddle::framework::DataLayout::kNCHW><<<grid1, block, 0, stream>>>(
            d_y->data<T>(),
            scale.data<BatchNormParamType<T>>(),
            running_var_data,
            epsilon,
            C,
            H * W,
            num,
            d_x->data<T>());
      }
      if (d_scale && d_bias) {
        KeBNBackwardScaleBias<
            T,
            block,
            paddle::framework::DataLayout::kNCHW><<<grid2, block, 0, stream>>>(
            d_y->data<T>(),
            x.data<T>(),
            running_mean_data,
            running_var_data,
            epsilon,
            N,
            C,
            H * W * D,
            d_scale->data<BatchNormParamType<T>>(),
            d_bias->data<BatchNormParamType<T>>());
      }
    } else {
      if (d_x) {
        KeBNBackwardData<
            T,
            paddle::framework::DataLayout::kNHWC><<<grid1, block, 0, stream>>>(
            d_y->data<T>(),
            scale.data<BatchNormParamType<T>>(),
            running_var_data,
            epsilon,
            C,
            H * W,
            num,
            d_x->data<T>());
      }
      if (d_scale && d_bias) {
        KeBNBackwardScaleBias<
            T,
            block,
            paddle::framework::DataLayout::kNHWC><<<grid2, block, 0, stream>>>(
            d_y->data<T>(),
            x.data<T>(),
            running_mean_data,
            running_var_data,
            epsilon,
            N,
            C,
            H * W * D,
            d_scale->data<BatchNormParamType<T>>(),
            d_bias->data<BatchNormParamType<T>>());
      }
    }
  }
}

template <typename T, typename Context>
void BatchNormGradKernel(const Context &dev_ctx,
                         const DenseTensor &y_grad,
                         const DenseTensor &x,
                         const DenseTensor &scale,
                         const DenseTensor &bias,
                         const DenseTensor &saved_mean,
                         const DenseTensor &saved_variance,
                         paddle::optional<const DenseTensor &> reserve_space,
                         paddle::optional<const DenseTensor &> mean,
                         paddle::optional<const DenseTensor &> variance,
                         float momentum,
                         float epsilon,
                         const std::string &data_layout,
                         bool is_test,
                         bool use_global_stats,
                         bool trainable_statistics,
                         bool fuse_with_relu,
                         DenseTensor *x_grad,
                         DenseTensor *scale_grad,
                         DenseTensor *bias_grad) {
  BatchNormGradRawKernel<T, Context>(dev_ctx,
                                     y_grad,
                                     x,
                                     scale,
                                     bias,
                                     saved_mean,
                                     saved_variance,
                                     reserve_space,
                                     mean,
                                     variance,
                                     momentum,
                                     epsilon,
                                     data_layout,
                                     is_test,
                                     use_global_stats,
                                     trainable_statistics,
                                     fuse_with_relu,
                                     false,
                                     x_grad,
                                     scale_grad,
                                     bias_grad);
}

}  // namespace pten

#ifdef PADDLE_WITH_HIP
PT_REGISTER_KERNEL(batch_norm_grad,
                   GPU,
                   ALL_LAYOUT,
                   pten::BatchNormGradKernel,
                   float,
                   pten::dtype::float16) {}

PT_REGISTER_KERNEL(batch_norm_grad_raw,
                   GPU,
                   ALL_LAYOUT,
                   pten::BatchNormGradRawKernel,
                   float,
                   pten::dtype::float16) {}
#else
PT_REGISTER_KERNEL(batch_norm_grad,
                   GPU,
                   ALL_LAYOUT,
                   pten::BatchNormGradKernel,
                   float,
                   double,
                   pten::dtype::float16) {
  if (kernel_key.dtype() == pten::DataType::FLOAT16) {
    kernel->OutputAt(1).SetDataType(pten::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(pten::DataType::FLOAT32);
    kernel->OutputAt(3).SetDataType(pten::DataType::FLOAT32);
    kernel->OutputAt(4).SetDataType(pten::DataType::FLOAT32);
  }
}

PT_REGISTER_KERNEL(batch_norm_grad_raw,
                   GPU,
                   ALL_LAYOUT,
                   pten::BatchNormGradRawKernel,
                   float,
                   double,
                   pten::dtype::float16) {
  if (kernel_key.dtype() == pten::DataType::FLOAT16) {
    kernel->OutputAt(1).SetDataType(pten::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(pten::DataType::FLOAT32);
    kernel->OutputAt(3).SetDataType(pten::DataType::FLOAT32);
    kernel->OutputAt(4).SetDataType(pten::DataType::FLOAT32);
  }
}

#endif
