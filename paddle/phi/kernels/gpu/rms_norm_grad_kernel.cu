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

/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved. */
/*This code is copied from NVIDIA apex:
 *     https://github.com/NVIDIA/apex
 *     with minor changes. */

#include <assert.h>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#ifdef PADDLE_WITH_HIP
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include "paddle/phi/backends/gpu/rocm/miopen_helper.h"
namespace cub = hipcub;
#define GPU(str) hip##str
#else
#include <cuda.h>          // NOLINT
#include <cuda_runtime.h>  // NOLINT
#include <cub/cub.cuh>
#include "paddle/phi/backends/gpu/cuda/cudnn_helper.h"
#define GPU(str) cuda##str
#endif
#include "paddle/phi/kernels/gpu/rms_norm_funcs.h"

namespace phi {

namespace {

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
                         V* grad_gamma) {
  GPU(Stream_t) stream = dev_ctx.stream();
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
  const uint64_t maxGridY = dev_ctx.GetCUDAMaxGridDimSize()[1];
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
                            DenseTensor* grad_scale,
                            const int begin_norm_axis) {
  const auto x_dims = x.dims();
  auto matrix_dim = phi::flatten_to_2d(x_dims, begin_norm_axis);
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
                          grad_scale->data<SCALE_TYPE>()));
}

}  // namespace

template <typename T, typename Context>
void RmsNormGradKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const paddle::optional<DenseTensor>& bias,
                       const paddle::optional<DenseTensor>& residual,
                       const DenseTensor& norm_weight,
                       const paddle::optional<DenseTensor>& norm_bias,
                       const DenseTensor& inv_var,
                       const DenseTensor& out_grad,
                       const float epsilon,
                       const int begin_norm_axis,
                       const float quant_scale,
                       DenseTensor* x_grad,
                       DenseTensor* norm_weight_grad,
                       DenseTensor* norm_bias_grad) {
  if (bias || residual || norm_bias) {
    PADDLE_THROW(common::errors::Unimplemented(
        "bias or residual or norm_bias is not supported yet"));
  }
  if (quant_scale > 0.0f) {
    PADDLE_THROW(common::errors::Unimplemented("quant is not supported yet"));
  }
  cuda_rms_norm_gradient<T, Context>(dev_ctx,
                                     x,
                                     norm_weight,
                                     inv_var,
                                     out_grad,
                                     epsilon,
                                     x_grad,
                                     norm_weight_grad,
                                     begin_norm_axis);
}
}  // namespace phi

#ifdef PADDLE_WITH_HIP
// MIOPEN do not support double

PD_REGISTER_KERNEL(rms_norm_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::RmsNormGradKernel,
                   float,
                   phi::dtype::float16) {}

#elif CUDNN_VERSION_MIN(8, 1, 0)

PD_REGISTER_KERNEL(rms_norm_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::RmsNormGradKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

#else

PD_REGISTER_KERNEL(rms_norm_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::RmsNormGradKernel,
                   float,
                   phi::dtype::float16) {}
#endif
