/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

// Original OneFlow copyright notice:

/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <assert.h>
#include "paddle/phi/backends/gpu/cuda/cudnn_helper.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#ifndef PADDLE_WITH_HIP
#include <cuda.h>          // NOLINT
#include <cuda_runtime.h>  // NOLINT
#include <cub/cub.cuh>
#include "paddle/phi/kernels/gpu/fused_rms_norm_funcs.h"
#endif

namespace phi {

namespace {
#ifndef PADDLE_WITH_HIP

template <typename Context, typename T, typename U, typename V>
void HostApplyRMSNorm(const Context& dev_ctx,
                      T* output,
                      U* invvar,
                      const T* input,
                      int n1,
                      int n2,
                      double epsilon,
                      const V* gamma) {
  const dim3 threads(32, 4, 1);
  const uint64_t maxGridY = dev_ctx.GetCUDAMaxGridDimSize()[1];
  const dim3 blocks(1, std::min((uint64_t)n1, maxGridY), 1);
  int nshared =
      threads.y > 1 ? threads.y * sizeof(U) + (threads.y / 2) * sizeof(U) : 0;
  cudaStream_t stream = dev_ctx.stream();
  cuApplyRMSNorm<<<blocks, threads, nshared, stream>>>(
      output, invvar, input, n1, n2, U(epsilon), gamma);
}

template <typename T, typename Context>
void cuda_rms_norm(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& scale,
                   const float epsilon,
                   DenseTensor* out,
                   DenseTensor* invvar) {
  const auto x_dims = x.dims();
  auto matrix_dim = phi::flatten_to_2d(x_dims, x_dims.size() - 1);
  int rows = static_cast<int>(matrix_dim[0]);
  int cols = static_cast<int>(matrix_dim[1]);

  dev_ctx.template Alloc<T>(out);
  dev_ctx.template Alloc<float>(invvar);

  DISPATCH_SCALE_TYPE(
      T,
      scale.dtype(),
      "cuda_rms_norm_kernel",
      HostApplyRMSNorm(dev_ctx,
                       out->data<T>(),
                       invvar->data<float>(),
                       const_cast<T*>(x.data<T>()),
                       rows,
                       cols,
                       epsilon,
                       const_cast<SCALE_TYPE*>(scale.data<SCALE_TYPE>())));
}

#endif
}  // namespace

template <typename T, typename Context>
void FusedRmsNormKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& scale,
                        const float epsilon,
                        DenseTensor* out,
                        DenseTensor* invvar) {
#if defined(PADDLE_WITH_HIP)
  PADDLE_THROW(phi::errors::Unimplemented(
      "Please compile with CUDA, ROCM platform isn't support it."));
#else
  cuda_rms_norm<T, Context>(dev_ctx, x, scale, epsilon, out, invvar);
#endif
}
}  // namespace phi

#ifdef PADDLE_WITH_HIP
// MIOPEN do not support double
PD_REGISTER_KERNEL(fused_rms_norm,
                   GPU,
                   ALL_LAYOUT,
                   phi::FusedRmsNormKernel,
                   float,
                   phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
  }
}

#elif CUDNN_VERSION_MIN(8, 1, 0)
PD_REGISTER_KERNEL(fused_rms_norm,
                   GPU,
                   ALL_LAYOUT,
                   phi::FusedRmsNormKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
  }
}

#else
PD_REGISTER_KERNEL(fused_rms_norm,
                   GPU,
                   ALL_LAYOUT,
                   phi::FusedRmsNormKernel,
                   float,
                   phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
  }
}

#endif
