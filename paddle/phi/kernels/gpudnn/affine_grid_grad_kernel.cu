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

#ifndef PADDLE_WITH_HIP

#include "paddle/phi/kernels/affine_grid_grad_kernel.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_device_function.h"
#include "paddle/phi/backends/gpu/gpu_dnn.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

using ScopedSpatialTransformerDescriptor =
    phi::backends::gpu::ScopedSpatialTransformerDescriptor;

template <typename T, typename Context>
void AffineGridGradCudnnKernel(const Context& dev_ctx,
                               const DenseTensor& output_grad,
                               const IntArray& outputShape,
                               bool align_corners,
                               DenseTensor* input_grad) {
  PADDLE_ENFORCE_EQ(
      dev_ctx.GetPlace().GetType() == phi::AllocationType::GPU,
      true,
      common::errors::InvalidArgument(
          "Only support for CUDAPlace.Please switch your context from "
          "CPUPlace to CUDAPlace or update your cudnn."));
  auto handle = dev_ctx.cudnn_handle();
  auto& theta_grad = input_grad;

  int n = output_grad.dims()[0];
  auto& size_attr = outputShape.GetData();
  int h_size_data[4] = {0};
  h_size_data[0] = n;
  h_size_data[1] = size_attr[1];
  h_size_data[2] = size_attr[2];
  h_size_data[3] = size_attr[3];

  ScopedSpatialTransformerDescriptor st_desc;
  cudnnSpatialTransformerDescriptor_t cudnn_st_desc =
      st_desc.descriptor<T>(4, h_size_data);

  const T* output_grad_data = output_grad.data<T>();
  T* theta_grad_data = dev_ctx.template Alloc<T>(theta_grad);

  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSpatialTfGridGeneratorBackward(
      handle, cudnn_st_desc, output_grad_data, theta_grad_data));
}

}  // namespace phi

PD_REGISTER_KERNEL(affine_grid_grad,  // cuda_only
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::AffineGridGradCudnnKernel,
                   float,
                   double){};
#endif
