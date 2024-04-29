/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/pool_kernel.h"

#include "paddle/phi/backends/gpu/gpu_dnn.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/pooling.h"
#include "paddle/phi/kernels/gpudnn/pool_gpudnn.h"

namespace phi {

template <typename T, typename Context>
void PoolRawGPUDNNKernel(const Context& ctx,
                         const DenseTensor& x,
                         const std::vector<int>& kernel_size,
                         const std::vector<int>& strides,
                         const std::vector<int>& paddings,
                         bool exclusive,
                         const std::string& data_format,
                         const std::string& pooling_type,
                         bool global_pooling,
                         bool adaptive,
                         const std::string& padding_algorithm,
                         DenseTensor* out) {
  PADDLE_ENFORCE_EQ(
      ctx.GetPlace().GetType() == phi::AllocationType::GPU,
      true,
      errors::InvalidArgument("Pool operator CUDA kernel must use CUDAPlace "
                              "rather than CPUPlace."));

  const DenseTensor* input = &x;
  DenseTensor* output = out;
  std::vector<int> paddings_ = paddings;
  std::vector<int> kernel_size_ = kernel_size;

  ctx.template Alloc<T>(output);

  const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");

  // update paddings_
  auto x_dims = input->dims();
  DDim data_dims;
  if (channel_last) {
    data_dims = slice_ddim(x_dims, 1, x_dims.size() - 1);
  } else {
    data_dims = slice_ddim(x_dims, 2, x_dims.size());
  }
  funcs::UpdatePadding(&paddings_,
                       global_pooling,
                       adaptive,
                       padding_algorithm,
                       data_dims,
                       strides,
                       kernel_size_);
  if (data_dims.size() * 2 == static_cast<int>(paddings_.size())) {
    for (int i = 0; i < data_dims.size(); ++i) {
      paddings_.erase(paddings_.begin() + i + 1);
    }
  }

  if (global_pooling) {
    funcs::UpdateKernelSize(&kernel_size_, data_dims);
  }

  const std::string str_NCHW = "NCHW", str_NHWC = "NHWC";
  const std::string str_NCDHW = "NCDHW", str_NDHWC = "NDHWC";

  // -----------------transformed tensor ------------------------

  DenseTensor transformed_input(input->type());
  DenseTensor transformed_output(output->type());
  GPUDNNDataLayout layout;

  if (data_format == str_NDHWC) {
    layout = GPUDNNDataLayout::kNCDHW;
    std::vector<int> axis{0, 4, 1, 2, 3};

    // input
    transformed_input.Resize(input->dims());

    auto in_dims_vec = common::vectorize(input->dims());
    in_dims_vec[1] = input->dims()[4];
    in_dims_vec[2] = input->dims()[1];
    in_dims_vec[3] = input->dims()[2];
    in_dims_vec[4] = input->dims()[3];
    transformed_input.Resize(common::make_ddim(in_dims_vec));
    ctx.Alloc(&transformed_input, input->type());

    funcs::Transpose<Context, T, 5> trans5;
    trans5(ctx, *input, &transformed_input, axis);

    // output
    transformed_output.Resize(output->dims());

    auto out_dims_vec = common::vectorize(output->dims());
    out_dims_vec[1] = output->dims()[4];
    out_dims_vec[2] = output->dims()[1];
    out_dims_vec[3] = output->dims()[2];
    out_dims_vec[4] = output->dims()[3];
    transformed_output.Resize(common::make_ddim(out_dims_vec));
#ifdef PADDLE_WITH_HIP
    // MIOPEN not support NHWC data layout
  } else if (data_format == str_NHWC) {
    layout = GPUDNNDataLayout::kNCHW;

    std::vector<int> axis{0, 3, 1, 2};

    transformed_input.Resize(input->dims());
    auto in_dims_vec = common::vectorize(input->dims());
    in_dims_vec[1] = input->dims()[3];
    in_dims_vec[2] = input->dims()[1];
    in_dims_vec[3] = input->dims()[2];
    transformed_input.Resize(common::make_ddim(in_dims_vec));
    ctx.Alloc(&transformed_input, input->type());

    funcs::Transpose<Context, T, 4> trans;
    trans(ctx, *input, &transformed_input, axis);

    transformed_output.Resize(output->dims());
    auto out_dims_vec = common::vectorize(output->dims());
    out_dims_vec[1] = output->dims()[3];
    out_dims_vec[2] = output->dims()[1];
    out_dims_vec[3] = output->dims()[2];
    transformed_output.Resize(common::make_ddim(out_dims_vec));
#endif
  } else {
    layout = GetLayoutFromStr(data_format);
    transformed_input = *input;
    transformed_output = *output;
  }

  const T* transformed_input_data = transformed_input.data<T>();
  T* transformed_output_data = ctx.template Alloc<T>(&transformed_output);

  // ------------------- cudnn descriptors ---------------------
  ScopedTensorDescriptor input_desc;
  ScopedTensorDescriptor output_desc;
  ScopedPoolingDescriptor pool_desc;

#ifdef PADDLE_WITH_HIP
  miopenTensorDescriptor_t cudnn_input_desc = input_desc.descriptor<T>(
      layout, common::vectorize<int>(transformed_input.dims()));
  miopenTensorDescriptor_t cudnn_output_desc = output_desc.descriptor<T>(
      layout, common::vectorize<int>(transformed_output.dims()));
#else
  cudnnTensorDescriptor_t cudnn_input_desc = input_desc.descriptor<T>(
      layout, common::vectorize<int>(transformed_input.dims()));
  cudnnTensorDescriptor_t cudnn_output_desc = output_desc.descriptor<T>(
      layout, common::vectorize<int>(transformed_output.dims()));
#endif
  PoolingMode pooling_mode;
  if (pooling_type == "max") {
    pooling_mode = PoolingMode::kMaximum;
  } else {
    pooling_mode = exclusive ? PoolingMode::kAverageExclusive
                             : PoolingMode::kAverageInclusive;
  }

#ifdef PADDLE_WITH_HIP
  miopenPoolingDescriptor_t cudnn_pool_desc =
      pool_desc.descriptor(pooling_mode, kernel_size_, paddings_, strides);
#else
  cudnnPoolingDescriptor_t cudnn_pool_desc =
      pool_desc.descriptor(pooling_mode, kernel_size_, paddings_, strides);
#endif

  // ------------------- cudnn pool algorithm ---------------------
  auto handle = ctx.cudnn_handle();
  ScalingParamType<T> alpha = 1.0f, beta = 0.0f;

#ifdef PADDLE_WITH_HIP
  char* pool_workspace;
  size_t pool_workernel_size_ = 0;
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::miopenPoolingGetWorkSpaceSizeV2(
      cudnn_pool_desc, cudnn_output_desc, &pool_workernel_size_));
  PADDLE_ENFORCE_GPU_SUCCESS(hipMalloc(&pool_workspace, pool_workernel_size_));
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::miopenPoolingForward(handle,
                                    cudnn_pool_desc,
                                    &alpha,
                                    cudnn_input_desc,
                                    transformed_input_data,
                                    &beta,
                                    cudnn_output_desc,
                                    transformed_output_data,
                                    false,
                                    pool_workspace,
                                    pool_workernel_size_));
  PADDLE_ENFORCE_GPU_SUCCESS(hipFree(pool_workspace));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cudnnPoolingForward(handle,
                                   cudnn_pool_desc,
                                   &alpha,
                                   cudnn_input_desc,
                                   transformed_input_data,
                                   &beta,
                                   cudnn_output_desc,
                                   transformed_output_data));
#endif
  // add
  if (data_format == str_NDHWC) {
    std::vector<int> axis{0, 2, 3, 4, 1};
    funcs::Transpose<Context, T, 5> trans5_v2;
    trans5_v2(ctx, transformed_output, output, axis);
  }
#ifdef PADDLE_WITH_HIP
  // MIOPEN not support NHWC data layout
  if (data_format == str_NHWC) {
    std::vector<int> axis{0, 2, 3, 1};
    funcs::Transpose<Context, T, 4> trans;
    trans(ctx, transformed_output, output, axis);
  }
#endif
}

template <typename T, typename Context>
void Pool2dGPUDNNKernel(const Context& ctx,
                        const DenseTensor& x,
                        const IntArray& kernel_size,
                        const std::vector<int>& strides,
                        const std::vector<int>& paddings,
                        bool ceil_mode,
                        bool exclusive,
                        const std::string& data_format,
                        const std::string& pooling_type,
                        bool global_pooling,
                        bool adaptive,
                        const std::string& padding_algorithm,
                        DenseTensor* out) {
  std::vector<int> kernel_size_val(kernel_size.GetData().begin(),
                                   kernel_size.GetData().end());
  PoolRawGPUDNNKernel<T, Context>(ctx,
                                  x,
                                  kernel_size_val,
                                  strides,
                                  paddings,
                                  exclusive,
                                  data_format,
                                  pooling_type,
                                  global_pooling,
                                  adaptive,
                                  padding_algorithm,
                                  out);
}

template <typename T, typename Context>
void Pool3dGPUDNNKernel(const Context& ctx,
                        const DenseTensor& x,
                        const std::vector<int>& kernel_size,
                        const std::vector<int>& strides,
                        const std::vector<int>& paddings,
                        bool ceil_mode,
                        bool exclusive,
                        const std::string& data_format,
                        const std::string& pooling_type,
                        bool global_pooling,
                        bool adaptive,
                        const std::string& padding_algorithm,
                        DenseTensor* out) {
  PoolRawGPUDNNKernel<T, Context>(ctx,
                                  x,
                                  kernel_size,
                                  strides,
                                  paddings,
                                  exclusive,
                                  data_format,
                                  pooling_type,
                                  global_pooling,
                                  adaptive,
                                  padding_algorithm,
                                  out);
}

}  // namespace phi

using phi::dtype::float16;

#ifdef PADDLE_WITH_HIP
// MIOPEN do not support double
PD_REGISTER_KERNEL(
    pool2d, GPUDNN, ALL_LAYOUT, phi::Pool2dGPUDNNKernel, float, float16) {}
PD_REGISTER_KERNEL(
    pool3d, GPUDNN, ALL_LAYOUT, phi::Pool3dGPUDNNKernel, float, float16) {}
#else
PD_REGISTER_KERNEL(pool2d,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::Pool2dGPUDNNKernel,
                   float,
                   double,
                   float16) {}
PD_REGISTER_KERNEL(pool3d,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::Pool3dGPUDNNKernel,
                   float,
                   double,
                   float16) {}
#endif
