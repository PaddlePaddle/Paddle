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

static cudaDeviceProp GetDevicePropImpl() {
  int device = -1;
  PD_CHECK(cudaGetDevice(&device) == cudaSuccess);
  cudaDeviceProp prop;
  PD_CHECK(cudaGetDeviceProperties(&prop, device) == cudaSuccess);
  return prop;
}

static cudaDeviceProp* GetDeviceProp() {
  static auto prop = GetDevicePropImpl();
  return &prop;
}

template <typename T, typename U, typename V, typename Context>
void HostRMSNormGradient(const Context& dev_ctx,
                         const T* dout,
                         const U* invvar,
                         const DenseTensor& input,
                         int n1,
                         int n2,
                         const V* gamma,
                         double epsilon,
                         T* grad_input,
                         V* grad_gamma,
                         cudaStream_t stream) {
  if (gamma != NULL) {
    const int part_size = 16;
    const dim3 threads2(32, 4, 1);
    const dim3 blocks2((n2 + threads2.x - 1) / threads2.x, part_size, 1);
    const int nshared2_a =
        2 * sizeof(U) * threads2.y * threads2.y * (threads2.x + 1);
    const int nshared2_b = threads2.x * threads2.y * sizeof(U);
    const int nshared2 = nshared2_a > nshared2_b ? nshared2_a : nshared2_b;
    std::vector<int64_t> shape = {part_size, n2};
    DenseTensor part_grad_gamma(
        std::shared_ptr<phi::Allocation>(nullptr),
        phi::DenseTensorMeta(phi::DataType::FLOAT32,
                             common::make_ddim({shape})));
    dev_ctx.template Alloc<float>(&part_grad_gamma);

    cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, stream>>>(
        dout,
        input.data<T>(),
        n1,
        n2,
        invvar,  // unused
        invvar,
        U(epsilon),
        part_grad_gamma.data<U>(),
        part_grad_gamma.data<U>(), /* unused */
        true);

    const dim3 threads3(32, 8, 1);
    const dim3 blocks3((n2 + threads2.x - 1) / threads2.x, 1, 1);
    const int nshared3 = threads3.x * threads3.y * sizeof(U);
    cuComputeGradGammaBeta<<<blocks3, threads3, nshared3, stream>>>(
        part_grad_gamma.data<U>(),
        part_grad_gamma.data<U>(), /* unused */
        part_size,
        n1,
        n2,
        grad_gamma,
        grad_gamma, /* unused */
        true);
  }

  // compute grad_input
  const uint64_t maxGridY = GetDeviceProp()->maxGridSize[1];
  const dim3 blocks1(1, std::min((uint64_t)n1, maxGridY), 1);
  const dim3 threads1(32, 4, 1);
  int nshared = threads1.y > 1 ? threads1.y * threads1.x * sizeof(U) : 0;

  const V* gamma_tmp = gamma;
  cuComputeGradInput<<<blocks1, threads1, nshared, stream>>>(
      dout,
      input.data<T>(),
      n1,
      n2,
      invvar, /* unused */
      invvar,
      U(epsilon),
      gamma_tmp,
      grad_input,
      true);
}

template <typename T, typename Context>
void cuda_rms_norm_gradient(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& scale,
                            const DenseTensor& invvar,
                            const DenseTensor& dy,
                            const float epsilon,
                            DenseTensor* grad_x,
                            DenseTensor* grad_scale) {
  const auto x_dims = x.dims();
  auto matrix_dim = phi::flatten_to_2d(x_dims, x_dims.size() - 1);
  int rows = static_cast<int>(matrix_dim[0]);
  int cols = static_cast<int>(matrix_dim[1]);
  dev_ctx.template Alloc<T>(grad_x);

  DISPATCH_SCALE_TYPE(T,
                      scale.type(),
                      "scale grad allocate",
                      dev_ctx.template Alloc<SCALE_TYPE>(grad_scale));

  DISPATCH_SCALE_TYPE(T,
                      scale.type(),
                      "cuda_rms_norm_gradient_kernel",
                      HostRMSNormGradient<T, float, SCALE_TYPE, Context>(
                          dev_ctx,
                          dy.data<T>(),
                          invvar.data<float>(),
                          x,
                          rows,
                          cols,
                          scale.data<SCALE_TYPE>(),
                          epsilon,
                          grad_x->data<T>(),
                          grad_scale->data<SCALE_TYPE>(),
                          dev_ctx.stream()));
}
#endif
}  // namespace

template <typename T, typename Context>
void FusedRmsNormGradKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& scale,
                            const DenseTensor& invvar,
                            const DenseTensor& dy,
                            const float epsilon,
                            DenseTensor* grad_x,
                            DenseTensor* grad_scale) {
#if defined(PADDLE_WITH_HIP)
  LOG(ERROR) << "Please compile with CUDA, ROCM platform isn't support it";
#else
  cuda_rms_norm_gradient<T, Context>(
      dev_ctx, x, scale, invvar, dy, epsilon, grad_x, grad_scale);
#endif
}
}  // namespace phi

#ifdef PADDLE_WITH_HIP
// MIOPEN do not support double

PD_REGISTER_KERNEL(fused_rms_norm_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::FusedRmsNormGradKernel,
                   float,
                   phi::dtype::float16) {}

#elif CUDNN_VERSION_MIN(8, 1, 0)

PD_REGISTER_KERNEL(fused_rms_norm_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::FusedRmsNormGradKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

#else

PD_REGISTER_KERNEL(fused_rms_norm_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::FusedRmsNormGradKernel,
                   float,
                   phi::dtype::float16) {}
#endif
