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

#include "paddle/phi/kernels/batch_norm_kernel.h"
#include "glog/logging.h"

#include "paddle/phi/backends/gpu/gpu_dnn.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/batch_norm_utils.h"
#include "paddle/phi/kernels/funcs/norm_utils.h"

namespace phi {

template <typename T>
using CudnnDataType = phi::backends::gpu::CudnnDataType<T>;
template <typename T>
using BatchNormParamType = typename CudnnDataType<T>::BatchNormParamType;

template <typename T, typename Context>
void BatchNormInferKernel(const Context &dev_ctx,
                          const DenseTensor &x,
                          const DenseTensor &mean,
                          const DenseTensor &variance,
                          const DenseTensor &scale,
                          const DenseTensor &bias,
                          float momentum,
                          float epsilon,
                          const std::string &data_layout,
                          DenseTensor *y,
                          DenseTensor *mean_out,
                          DenseTensor *variance_out) {
  // Since saved_mean and saved_variance are used regardless of whether
  // they are in test mode, temporary variables need to be created here
  // to be compatible
  auto saved_mean = phi::EmptyLike<T, Context>(dev_ctx, *mean_out);
  auto saved_variance = phi::EmptyLike<T, Context>(dev_ctx, *variance_out);
  BatchNormKernel<T, Context>(dev_ctx,
                              x,
                              mean,
                              variance,
                              scale,
                              bias,
                              /*is_test=*/true,
                              momentum,
                              epsilon,
                              data_layout,
                              /*use_global_stats=*/false,
                              /*trainable_statistics=*/false,
                              y,
                              mean_out,
                              variance_out,
                              &saved_mean,
                              &saved_variance,
                              /*reserve_space=*/nullptr);
}

template <typename T, typename Context>
void BatchNormMUSAKernel(const Context &ctx,
                         const DenseTensor &x,
                         const DenseTensor &mean,
                         const DenseTensor &variance,
                         const DenseTensor &scale,
                         const DenseTensor &bias,
                         float momentum,
                         float epsilon_f,
                         const std::string &data_layout_str,
                         DenseTensor *y,
                         DenseTensor *mean_out,
                         DenseTensor *variance_out) {
  double epsilon = epsilon_f;
  const DataLayout data_layout = phi::StringToDataLayout(data_layout_str);

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
  phi::funcs::ExtractNCWHD(x_dims, data_layout, &N, &C, &H, &W, &D);

#if defined(PADDLE_WITH_HIP) || defined(PADDLE_WITH_MUSA)
  auto compute_format =
      data_layout == DataLayout::kNHWC ? DataLayout::kNHWC : DataLayout::kNCHW;

// TODO(wangran16): wait for MIOpen to improve the performance of BN
// HIP do not support compute format of NHWC
// auto compute_format = DataLayout::kNCHW;
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
#elif defined(PADDLE_WITH_MUSA)
  backends::gpu::TensorDescriptor data_desc_;
  backends::gpu::TensorDescriptor output_desc_;
  backends::gpu::TensorDescriptor mean_desc_;
  backends::gpu::TensorDescriptor variance_desc_;
  backends::gpu::TensorDescriptor scale_desc_;
  backends::gpu::TensorDescriptor bias_desc_;
  dynload::BatchNorm batch_norm_desc_;
  dynload::BatchNorm::Mode mode_;
#else
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
#elif defined(PADDLE_WITH_MUSA)
  if (H == 1 && W == 1) {
    mode_ = dynload::BatchNorm::Mode::PER_ACTIVATION;
  } else {
    mode_ = dynload::BatchNorm::Mode::PER_CHANNEL;
  }
#elif CUDNN_VERSION_MIN(7, 0, 1)
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

#if defined(PADDLE_WITH_HIP)
#elif defined(PADDLE_WITH_MUSA)
  auto layout_format = data_layout == DataLayout::kNHWC
                           ? dynload::Tensor::Format::NHWC
                           : dynload::Tensor::Format::NCHW;
  data_desc_.set(transformed_x, layout_format);
  output_desc_.set(transformed_y, layout_format);
#else
#endif

  auto handle = ctx.cudnn_handle();

  // Now, depending on whether we are running test or not, we have two paths.
  // It is training mode when it's not reference AND not using pre-trained
  // model.
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

  scale_desc_.set<BatchNormParamType<T>>(
      scale, scale.template data<BatchNormParamType<T>>());
  bias_desc_.set<BatchNormParamType<T>>(
      bias, bias.template data<BatchNormParamType<T>>());
  mean_desc_.set<BatchNormParamType<T>>(
      *est_mean, mean.template data<BatchNormParamType<T>>());
  variance_desc_.set<BatchNormParamType<T>>(
      *est_var, est_var->template data<BatchNormParamType<T>>());
  batch_norm_desc_.SetEpsilon(epsilon);
  batch_norm_desc_.SetMode(mode_);
  batch_norm_desc_.RunPure(*handle,
                           *output_desc_.desc(),
                           *data_desc_.desc(),
                           *mean_desc_.desc(),
                           *variance_desc_.desc(),
                           *scale_desc_.desc(),
                           *bias_desc_.desc());

  if (data_layout == DataLayout::kNHWC && compute_format == DataLayout::kNCHW &&
      x_dims.size() > 2) {
    VLOG(3) << "Transform batchnorm output from NCHW to NHWC";
    TransToChannelLast<Context, T>(ctx, &transformed_y, y);
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(batch_norm_infer,
                   CPU,
                   ALL_LAYOUT,
                   phi::BatchNormInferKernel,
                   float,
                   double) {}
#ifdef PADDLE_WITH_CUDA
#if CUDNN_VERSION_MIN(8, 1, 0)
PD_REGISTER_KERNEL(batch_norm_infer,
                   GPU,
                   ALL_LAYOUT,
                   phi::BatchNormInferKernel,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16 ||
      kernel_key.dtype() == phi::DataType::BFLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
  }
}
#else
PD_REGISTER_KERNEL(batch_norm_infer,
                   GPU,
                   ALL_LAYOUT,
                   phi::BatchNormInferKernel,
                   float,
                   double,
                   phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
  }
}
#endif
#endif
#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(batch_norm_infer,
                   GPU,
                   ALL_LAYOUT,
                   phi::BatchNormInferKernel,
                   float,
                   phi::dtype::float16) {}
#endif
#ifdef PADDLE_WITH_MUSA
PD_REGISTER_KERNEL(batch_norm_infer,
                   GPU,
                   ALL_LAYOUT,
                   phi::BatchNormMUSAKernel,
                   float,
                   phi::dtype::float16) {}
#endif
#ifdef PADDLE_WITH_XPU
PD_REGISTER_KERNEL(batch_norm_infer,
                   XPU,
                   ALL_LAYOUT,
                   phi::BatchNormInferKernel,
                   float,
                   phi::dtype::float16) {}
#endif
