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

#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/batch_norm_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

#include "paddle/fluid/operators/norm_utils.cu.h"
#include "paddle/fluid/operators/norm_utils.h"

#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/operators/layout_utils.h"
#include "paddle/fluid/platform/enforce.h"

#include "paddle/fluid/platform/flags.h"
#include "paddle/phi/kernels/gpu/batch_norm_utils.h"

#ifdef __HIPCC__
#define LAUNCH_BOUNDS(BlockDim) __launch_bounds__(BlockDim)
#else
#define LAUNCH_BOUNDS(BlockDim)
#endif

DECLARE_bool(cudnn_batchnorm_spatial_persistent);

namespace phi {

template <typename T>
using CudnnDataType = paddle::platform::CudnnDataType<T>;
template <typename T>
using BatchNormParamType = typename CudnnDataType<T>::BatchNormParamType;

template <typename T, phi::DataLayout layout>
static __global__ void BNForwardInference(const T *x,
                                          const BatchNormParamType<T> *mean,
                                          const BatchNormParamType<T> *variance,
                                          const BatchNormParamType<T> *scale,
                                          const BatchNormParamType<T> *bias,
                                          const int C,
                                          const int N,
                                          const int HxW,
                                          const double epsilon,
                                          T *y) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int num = N * C * HxW;
  for (int i = gid; i < num; i += stride) {
    const int c = layout == phi::DataLayout::kNCHW ? i / HxW % C : i % C;
    BatchNormParamType<T> x_sub_mean =
        static_cast<BatchNormParamType<T>>(x[i]) - mean[c];
    BatchNormParamType<T> inv_var = 1 / sqrt(variance[c] + epsilon);
    y[i] = static_cast<T>(scale[c] * x_sub_mean * inv_var + bias[c]);
  }
}

template <typename T, int BlockDim, phi::DataLayout layout>
static __global__ LAUNCH_BOUNDS(BlockDim) void BNForwardTraining(
    const T *x,
    const BatchNormParamType<T> *scale,
    const BatchNormParamType<T> *bias,
    const int C,
    const int N,
    const int HxW,
    const double epsilon,
    double exponentialAverageFactor,
    T *y,
    BatchNormParamType<T> *mean,
    BatchNormParamType<T> *variance,
    BatchNormParamType<T> *save_mean,
    BatchNormParamType<T> *save_inv_variance) {
  int outer_size = C;
  int inner_size = N * HxW;
  typedef cub::BlockReduce<BatchNormParamType<T>, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage mean_storage;
  __shared__ typename BlockReduce::TempStorage variance_storeage;
  __shared__ BatchNormParamType<T> mean_val;
  __shared__ BatchNormParamType<T> variance_val;
  __shared__ BatchNormParamType<T> inv_var_val;

  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    BatchNormParamType<T> x_sum = static_cast<BatchNormParamType<T>>(0);
    BatchNormParamType<T> x_square_sum = static_cast<BatchNormParamType<T>>(0);

    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int index = layout == phi::DataLayout::kNCHW
                            ? (j / HxW * C + i) * HxW + j % HxW
                            : j * outer_size + i;
      BatchNormParamType<T> x_i = static_cast<BatchNormParamType<T>>(x[index]);
      x_sum += x_i;
      x_square_sum += x_i * x_i;
    }
    x_sum = BlockReduce(mean_storage).Reduce(x_sum, cub::Sum());
    x_square_sum =
        BlockReduce(variance_storeage).Reduce(x_square_sum, cub::Sum());
    if (threadIdx.x == 0) {
      mean_val = x_sum / inner_size;
      variance_val = x_square_sum / inner_size - mean_val * mean_val;
      inv_var_val = 1 / sqrt(variance_val + epsilon);

      if (save_mean && save_inv_variance) {
        save_mean[i] = mean_val;
        save_inv_variance[i] = inv_var_val;
      }
      mean[i] = (1 - exponentialAverageFactor) * mean_val +
                exponentialAverageFactor * mean[i];
      variance[i] = (1 - exponentialAverageFactor) * variance_val +
                    exponentialAverageFactor * variance[i];
    }
    __syncthreads();

    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int index = layout == phi::DataLayout::kNCHW
                            ? (j / HxW * C + i) * HxW + j % HxW
                            : j * outer_size + i;
      BatchNormParamType<T> x_sub_mean =
          static_cast<BatchNormParamType<T>>(x[index]) - mean_val;
      y[index] = scale[i] * x_sub_mean * inv_var_val + bias[i];
    }
  }
}

template <typename T, typename Context>
void BatchNormKernel(const Context &ctx,
                     const DenseTensor &x,
                     const DenseTensor &scale,
                     const DenseTensor &bias,
                     const DenseTensor &mean,
                     const DenseTensor &variance,
                     float momentum,
                     float epsilon_f,
                     const std::string &data_layout_str,
                     bool is_test,
                     bool use_global_stats,
                     bool trainable_statistics,
                     bool fuse_with_relu,
                     DenseTensor *y,
                     DenseTensor *mean_out,
                     DenseTensor *variance_out,
                     DenseTensor *saved_mean,
                     DenseTensor *saved_variance,
                     DenseTensor *reserve_space) {
  double epsilon = epsilon_f;
  const bool trainable_stats = trainable_statistics;
  const DataLayout data_layout =
      paddle::framework::StringToDataLayout(data_layout_str);
  bool test_mode = is_test && (!trainable_stats);

  // Get the size for each dimension.
  // NCHW [batch_size, in_channels, in_height, in_width]
  const auto &x_dims = x.dims();
  PADDLE_ENFORCE_EQ(
      x_dims.size() >= 2 && x_dims.size() <= 5,
      true,
      phi::errors::InvalidArgument(
          "The size of input's dimensions should be between 2 and 5"
          "But received: the size of input's dimensions is [%d]",
          x_dims.size()));

  ctx.template Alloc<T>(y);
  int N, C, H, W, D;
  paddle::operators::ExtractNCWHD(x_dims, data_layout, &N, &C, &H, &W, &D);

  auto dtype = paddle::platform::CudnnDataType<T>::type;

#ifdef PADDLE_WITH_HIP
  auto compute_format =
      data_layout == DataLayout::kNHWC ? DataLayout::kNHWC : DataLayout::kNCHW;

// TODO(wangran16): wait for MIOpen to improve the performance of BN
// HIP do not support compute format of NHWC
// auto compute_format = DataLayout::kNCHW;
#else
  const bool fast_nhwc_batch_norm =
      test_mode ||
      (dtype == CUDNN_DATA_HALF && FLAGS_cudnn_batchnorm_spatial_persistent);

  auto compute_format = fast_nhwc_batch_norm && data_layout == DataLayout::kNHWC
                            ? DataLayout::kNHWC
                            : DataLayout::kNCHW;
#endif

  DenseTensor transformed_x(x.type());
  DenseTensor transformed_y(y->type());

  if (data_layout == DataLayout::kNHWC && compute_format == DataLayout::kNCHW &&
      x_dims.size() > 2) {
    VLOG(3) << "Transform input tensor from NHWC to NCHW.";
    ResizeToChannelFirst<Context, T>(ctx, &x, &transformed_x);
    TransToChannelFirst<Context, T>(ctx, &x, &transformed_x);
    ResizeToChannelFirst<Context, T>(ctx, y, &transformed_y);
  } else {
    transformed_x.ShareDataWith(x);
    transformed_y.ShareDataWith(*y);
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
      paddle::platform::dynload::cudnnCreateTensorDescriptor(&bn_param_desc_));
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

  VLOG(3) << "Setting descriptors.";
  std::vector<int> dims;
  std::vector<int> strides;
  if (compute_format == DataLayout::kNCHW) {
    dims = {N, C, H, W, D};
    strides = {C * H * W * D, H * W * D, W * D, D, 1};
  } else {
    dims = {N, C, H, W, D};
    strides = {H * W * D * C, 1, W * D * C, D * C, C};
  }

#ifdef PADDLE_WITH_HIP
// TODO(wangran16): wait for MIOpen to improve the performance of BN
// PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::miopenSetTensorDescriptor(
//     data_desc_, CudnnDataType<T>::type,
//     x_dims.size() > 3 ? x_dims.size() : 4, const_cast<int *>(dims.data()),
//     const_cast<int *>(strides.data())));
// Note: PERSISTENT not implemented for inference
// PADDLE_ENFORCE_GPU_SUCCESS(
//     platform::dynload::miopenDeriveBNTensorDescriptor(
//         bn_param_desc_, data_desc_, test_mode ? miopenBNSpatial : mode_));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(
      paddle::platform::dynload::cudnnSetTensorNdDescriptor(
          data_desc_,
          CudnnDataType<T>::type,
          x_dims.size() > 3 ? x_dims.size() : 4,
          dims.data(),
          strides.data()));
  // Note: PERSISTENT not implemented for inference
  PADDLE_ENFORCE_GPU_SUCCESS(
      paddle::platform::dynload::cudnnDeriveBNTensorDescriptor(
          bn_param_desc_,
          data_desc_,
          test_mode ? CUDNN_BATCHNORM_SPATIAL : mode_));
#endif

  auto handle = ctx.cudnn_handle();

  // Now, depending on whether we are running test or not, we have two paths.
  // It is training mode when it's not reference AND not using pre-trained
  // model.
  bool training = !test_mode && !use_global_stats;
  if (!training) {
    // only when test we use input to do computation.
    const auto *est_mean = &mean;
    const auto *est_var = &variance;
    // Run inference mode.
    PADDLE_ENFORCE_EQ(
        est_mean->dims().size(),
        1UL,
        phi::errors::InvalidArgument(
            "The size of mean's dimensions must equal to 1."
            "But received: the size of mean's dimensions mean is [%d],"
            "the dimensions of mean is [%s].",
            est_mean->dims().size(),
            est_mean->dims()));
    PADDLE_ENFORCE_EQ(
        est_var->dims().size(),
        1UL,
        phi::errors::InvalidArgument(
            "The size of variance's dimensions must equal to 1."
            "But received: the size of variance's dimensions is [%d],"
            "the dimensions of variance is [%s].",
            est_var->dims().size(),
            est_var->dims()));
    PADDLE_ENFORCE_EQ(
        est_mean->dims()[0],
        C,
        phi::errors::InvalidArgument(
            "The first dimension of mean must equal to the number of "
            "Channels, which is [%d]. But received: the first dimension"
            "of mean is [%d], the dimensions of mean is [%s].",
            C,
            est_mean->dims()[0],
            est_mean->dims()));
    PADDLE_ENFORCE_EQ(
        est_var->dims()[0],
        C,
        phi::errors::InvalidArgument(
            "The first dimension of variance must equal to the number"
            "of Channels, which is [%d]. But received: the first dimension of"
            "variance is [%d], the dimensions of variance is [%s].",
            C,
            est_var->dims()[0],
            est_var->dims()));

#ifdef PADDLE_WITH_HIP
    const int block_size = 256;
    const int grid_size = (N * C * H * W * D + block_size - 1) / block_size;
    if (compute_format == DataLayout::kNCHW) {
      BNForwardInference<
          T,
          DataLayout::kNCHW><<<grid_size, block_size, 0, ctx.stream()>>>(
          transformed_x.template data<T>(),
          est_mean->template data<BatchNormParamType<T>>(),
          est_var->template data<BatchNormParamType<T>>(),
          scale.template data<BatchNormParamType<T>>(),
          bias.template data<BatchNormParamType<T>>(),
          C,
          N,
          H * W * D,
          epsilon,
          transformed_y.template data<T>());
    } else {
      BNForwardInference<
          T,
          DataLayout::kNHWC><<<grid_size, block_size, 0, ctx.stream()>>>(
          transformed_x.template data<T>(),
          est_mean->template data<BatchNormParamType<T>>(),
          est_var->template data<BatchNormParamType<T>>(),
          scale.template data<BatchNormParamType<T>>(),
          bias.template data<BatchNormParamType<T>>(),
          C,
          N,
          H * W * D,
          epsilon,
          transformed_y.template data<T>());
    }
// TODO(wangran16): wait for MIOpen to improve the performance of BN
// PADDLE_ENFORCE_GPU_SUCCESS(
//     platform::dynload::miopenBatchNormalizationForwardInference(
//         handle, miopenBNSpatial,
//         const_cast<void *>(
//             static_cast<const void *>(CudnnDataType<T>::kOne())),
//         const_cast<void *>(
//             static_cast<const void *>(CudnnDataType<T>::kZero())),
//         data_desc_,
//         static_cast<const void *>(transformed_x.template data<T>()),
//         data_desc_,
//         static_cast<void *>(
//             transformed_y.template mutable_data<T>(ctx.GetPlace())),
//         bn_param_desc_,
//         const_cast<void *>(static_cast<const void *>(
//             scale->template data<BatchNormParamType<T>>())),
//         const_cast<void *>(static_cast<const void *>(
//             bias->template data<BatchNormParamType<T>>())),
//         const_cast<void *>(static_cast<const void *>(
//             est_mean->template data<BatchNormParamType<T>>())),
//         const_cast<void *>(static_cast<const void *>(
//             est_var->template data<BatchNormParamType<T>>())),
//         epsilon));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(
        paddle::platform::dynload::cudnnBatchNormalizationForwardInference(
            handle,
            // Note: PERSISTENT not implemented for inference
            CUDNN_BATCHNORM_SPATIAL,
            CudnnDataType<T>::kOne(),
            CudnnDataType<T>::kZero(),
            data_desc_,
            transformed_x.template data<T>(),
            data_desc_,
            ctx.template Alloc<T>(&transformed_y),
            bn_param_desc_,
            scale.template data<BatchNormParamType<T>>(),
            bias.template data<BatchNormParamType<T>>(),
            est_mean->template data<BatchNormParamType<T>>(),
            est_var->template data<BatchNormParamType<T>>(),
            epsilon));
#endif
  } else {
    // if MomentumTensor is set, use MomentumTensor value, momentum
    // is only used in this training branch

    // need to solve here
    // if (ctx.HasInput("MomentumTensor")) {
    //   const auto *mom_tensor = MomentumTensor;
    //   DenseTensor mom_cpu;
    //   paddle::framework::TensorCopySync(*mom_tensor, platform::CPUPlace(),
    //                                     &mom_cpu);
    //   momentum = mom_cpu.data<float>()[0];
    // }

    // Run training mode.
    // obtain running mean and running inv var, and there is no need
    // to initialize them.
    ctx.template Alloc<BatchNormParamType<T>>(mean_out);
    ctx.template Alloc<BatchNormParamType<T>>(variance_out);

    ctx.template Alloc<BatchNormParamType<T>>(saved_mean);
    ctx.template Alloc<BatchNormParamType<T>>(saved_variance);

    if ((N * H * W * D) == 1) {
      // Only 1 element in normalization dimension,
      // skip the batch norm calculation, let y = x.
      paddle::framework::TensorCopy(x, ctx.GetPlace(), y);
    } else {
      double this_factor = 1. - momentum;

      bool called = false;
#if CUDNN_VERSION_MIN(7, 4, 1)
      called = true;
      size_t workspace_size = 0;
      size_t reserve_space_size = 0;
      void *reserve_space_ptr = nullptr;
      void *workspace_ptr = nullptr;
      DenseTensor workspace_tensor;
      DenseTensor reserve_space_tensor;
      // Create reserve space and workspace for batch norm.
      // Create tensor for each batchnorm op, it will be used in the
      // backward. Thus this tensor shouldn't be temp.
      // auto *reserve_space = ctx.Output<Tensor>("ReserveSpace");
      if (reserve_space == nullptr) {
        reserve_space = &reserve_space_tensor;
      }
      PADDLE_ENFORCE_NOT_NULL(
          reserve_space,
          phi::errors::NotFound(
              "The argument ReserveSpace of batch_norm op is not found."));
      // --------------- cudnn batchnorm workspace ---------------
      PADDLE_ENFORCE_GPU_SUCCESS(
          paddle::platform::dynload::
              cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
                  /*handle=*/handle,
                  /*mode=*/mode_,
                  /*bnIps=*/CUDNN_BATCHNORM_OPS_BN,
                  /*xDesc=*/data_desc_,
                  /*zDesc=*/nullptr,
                  /*yDesc=*/data_desc_,
                  /*bnScaleBiasMeanVarDesc=*/bn_param_desc_,
                  /*activationDesc=*/nullptr,
                  /*sizeInBytes=*/&workspace_size));

      // -------------- cudnn batchnorm reserve space --------------
      PADDLE_ENFORCE_GPU_SUCCESS(
          paddle::platform::dynload::
              cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
                  /*handle=*/handle,
                  /*mode=*/mode_,
                  /*bnOps=*/CUDNN_BATCHNORM_OPS_BN,
                  /*activationDesc=*/nullptr,
                  /*xDesc=*/data_desc_,
                  /*sizeInBytes=*/&reserve_space_size));

      reserve_space->Resize({static_cast<int64_t>(reserve_space_size)});
      reserve_space_ptr =
          static_cast<void *>(ctx.template Alloc<uint8_t>(reserve_space));
      workspace_tensor.Resize({static_cast<int64_t>(workspace_size)});
      workspace_ptr =
          static_cast<void *>(ctx.template Alloc<uint8_t>(&workspace_tensor));
      PADDLE_ENFORCE_GPU_SUCCESS(
          paddle::platform::dynload::cudnnBatchNormalizationForwardTrainingEx(
              handle,
              mode_,
              CUDNN_BATCHNORM_OPS_BN,
              CudnnDataType<T>::kOne(),
              CudnnDataType<T>::kZero(),
              data_desc_,
              transformed_x.template data<T>(),
              nullptr,
              nullptr,
              data_desc_,
              transformed_y.template data<T>(),
              bn_param_desc_,
              scale.template data<BatchNormParamType<T>>(),
              bias.template data<BatchNormParamType<T>>(),
              this_factor,
              ctx.template Alloc<BatchNormParamType<T>>(mean_out),
              ctx.template Alloc<BatchNormParamType<T>>(variance_out),
              epsilon,
              ctx.template Alloc<BatchNormParamType<T>>(saved_mean),
              ctx.template Alloc<BatchNormParamType<T>>(saved_variance),
              nullptr,
              workspace_ptr,
              workspace_size,
              reserve_space_ptr,
              reserve_space_size));
#endif  // CUDNN_VERSION_MIN(7, 4, 1)
      if (!called) {
#ifdef PADDLE_WITH_HIP
        const int num = transformed_x.numel();
        const int block = 256;
        const int max_threads = ctx.GetMaxPhysicalThreadCount();
        const int max_blocks = std::max(max_threads / block, 1);
        const int grid = std::min(C, max_blocks);
        if (compute_format == DataLayout::kNCHW) {
          BNForwardTraining<
              T,
              block,
              DataLayout::kNCHW><<<grid, block, 0, ctx.stream()>>>(
              transformed_x.template data<T>(),
              scale.template data<BatchNormParamType<T>>(),
              bias.template data<BatchNormParamType<T>>(),
              C,
              N,
              H * W * D,
              epsilon,
              this_factor,
              transformed_y.template data<T>(),
              mean_out->template data<BatchNormParamType<T>>(),
              variance_out->template data<BatchNormParamType<T>>(),
              saved_mean->template data<BatchNormParamType<T>>(),
              saved_variance->template data<BatchNormParamType<T>>());
        } else {
          BNForwardTraining<
              T,
              block,
              DataLayout::kNHWC><<<grid, block, 0, ctx.stream()>>>(
              transformed_x.template data<T>(),
              scale.template data<BatchNormParamType<T>>(),
              bias.template data<BatchNormParamType<T>>(),
              C,
              N,
              H * W * D,
              epsilon,
              this_factor,
              transformed_y.template data<T>(),
              mean_out->template data<BatchNormParamType<T>>(),
              variance_out->template data<BatchNormParamType<T>>(),
              saved_mean->template data<BatchNormParamType<T>>(),
              saved_variance->template data<BatchNormParamType<T>>());
        }
// TODO(wangran16): wait for MIOpen to improve the performance of BN
// PADDLE_ENFORCE_GPU_SUCCESS(
//     platform::dynload::miopenBatchNormalizationForwardTraining(
//         handle, mode_, const_cast<void *>(static_cast<const void *>(
//                            CudnnDataType<T>::kOne())),
//         const_cast<void *>(
//             static_cast<const void *>(CudnnDataType<T>::kZero())),
//         data_desc_,
//         static_cast<const void *>(transformed_x.template data<T>()),
//         data_desc_,
//         static_cast<void *>(
//             transformed_y.template mutable_data<T>(ctx.GetPlace())),
//         bn_param_desc_,
//         const_cast<void *>(static_cast<const void *>(
//             scale->template data<BatchNormParamType<T>>())),
//         const_cast<void *>(static_cast<const void *>(
//             bias->template data<BatchNormParamType<T>>())),
//         this_factor,
//         static_cast<void *>(
//             mean_out->template mutable_data<BatchNormParamType<T>>(
//                 ctx.GetPlace())),
//         static_cast<void *>(variance_out->template mutable_data<
//                             BatchNormParamType<T>>(ctx.GetPlace())),
//         epsilon,
//         static_cast<void *>(
//             saved_mean->template mutable_data<BatchNormParamType<T>>(
//                 ctx.GetPlace())),
//         static_cast<void *>(saved_variance->template mutable_data<
//                             BatchNormParamType<T>>(ctx.GetPlace()))));
#else
        PADDLE_ENFORCE_GPU_SUCCESS(
            paddle::platform::dynload::cudnnBatchNormalizationForwardTraining(
                handle,
                mode_,
                CudnnDataType<T>::kOne(),
                CudnnDataType<T>::kZero(),
                data_desc_,
                transformed_x.template data<T>(),
                data_desc_,
                ctx.template Alloc<T>(&transformed_y),
                bn_param_desc_,
                scale.template data<BatchNormParamType<T>>(),
                bias.template data<BatchNormParamType<T>>(),
                this_factor,
                ctx.template Alloc<BatchNormParamType<T>>(mean_out),
                ctx.template Alloc<BatchNormParamType<T>>(variance_out),
                epsilon,
                ctx.template Alloc<BatchNormParamType<T>>(saved_mean),
                ctx.template Alloc<BatchNormParamType<T>>(saved_variance)));
#endif
      }
    }
  }

  if (data_layout == DataLayout::kNHWC && compute_format == DataLayout::kNCHW &&
      x_dims.size() > 2) {
    VLOG(3) << "Transform batchnorm output from NCHW to NHWC";
    TransToChannelLast<Context, T>(ctx, &transformed_y, y);
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
      paddle::platform::dynload::cudnnDestroyTensorDescriptor(bn_param_desc_));
#endif
}

}  // namespace phi

#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(batch_norm,
                   GPU,
                   ALL_LAYOUT,
                   phi::BatchNormKernel,
                   float,
                   phi::dtype::float16) {}
#else
PD_REGISTER_KERNEL(batch_norm,
                   GPU,
                   ALL_LAYOUT,
                   phi::BatchNormKernel,
                   float,
                   double,
                   phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(4).SetDataType(phi::DataType::FLOAT32);
  }
}

#endif
