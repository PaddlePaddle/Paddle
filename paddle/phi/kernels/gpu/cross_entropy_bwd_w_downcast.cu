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

#include "paddle/phi/kernels/cross_entropy_grad_kernel.h"

#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include "paddle/phi/backends/gpu/gpu_device_function.h"
#include "paddle/phi/backends/gpu/gpu_dnn.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/softmax.h"
#include "paddle/phi/kernels/gpudnn/softmax_gpudnn.h"

namespace phi {

/*
  Wrapper of softmax with cross entropy grad hard label.
*/
template <typename T, typename LabelT, typename LogitT>
__global__ void SoftmaxWithCrossEntropyGradHardLabel(LogitT* logits_grad,
                                                     const T* loss_grad,
                                                     const T* softmax,
                                                     const LabelT* labels,
                                                     const int64_t n,
                                                     const int64_t dim,
                                                     const int64_t d,
                                                     const int ignore_index) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t idx_n = idx / (d * dim);
  int64_t idx_dim = (idx / d) % dim;
  int64_t idx_d = idx % d;
  int64_t ids = idx_n * d + idx_d;

  if (idx < n * dim * d) {
    auto lbl = static_cast<int64_t>(labels[ids]);
    if (lbl == ignore_index) {
      logits_grad[idx] = static_cast<LogitT>(0.0);
    } else if (lbl == idx_dim) {
      logits_grad[idx] = static_cast<LogitT>(
          (static_cast<float>(softmax[idx]) - static_cast<float>(1.0)) *
          static_cast<float>(loss_grad[ids]));
    } else {
      logits_grad[idx] =
          static_cast<LogitT>(static_cast<float>(softmax[idx]) *
                              static_cast<float>(loss_grad[ids]));
    }
  }
}

template <typename T, typename LabelT>
void CrossEntropyWithSoftmaxBwdWithDowncastGPUKernel(
    const GPUContext& dev_ctx,
    const DenseTensor& label,
    const DenseTensor& softmax,
    const DenseTensor& loss_grad,
    int axis,
    DenseTensor* logits_grad) {
  PADDLE_ENFORCE_EQ(
      dev_ctx.GetPlace().GetType(),
      phi::AllocationType::GPU,
      common::errors::Unavailable("softmax_with_cross_entropy operator's "
                                  "CUDA kernel only runs on GPU device."));
  using LogitT = phi::bfloat16;
  const T* loss_grad_data = loss_grad.data<T>();
  DenseTensor* logit_grad = logits_grad;

  LogitT* logit_grad_data = nullptr;
  // using no-copy branch in
  // Paddle/paddle/phi/kernels/gpu/cross_entropy_grad_kernel.cu
  logit_grad_data = dev_ctx.template Alloc<LogitT>(logit_grad);

  const int rank = logit_grad->dims().size();
  const int axis_v = phi::funcs::CanonicalAxis(axis, rank);
  int axis_dim = logit_grad->dims()[axis_v];

  const int64_t n = phi::funcs::SizeToAxis(axis_v, logit_grad->dims());
  const int64_t d = phi::funcs::SizeFromAxis(axis_v, logit_grad->dims());
  const int64_t remain = d / axis_dim;

  int block = 512;
  auto stream = dev_ctx.stream();

  // using hard_label branch
  const T* softmax_data = softmax.data<T>();
  const auto* label_data = label.data<LabelT>();
  int64_t grid = (n * d + block - 1) / block;
  SoftmaxWithCrossEntropyGradHardLabel<T, LabelT, LogitT>
      <<<grid, block, 0, stream>>>(logit_grad_data,
                                   loss_grad_data,
                                   softmax_data,
                                   label_data,
                                   n,
                                   d / remain,
                                   remain,
                                   -100);
}

template <typename T, typename Context>
void CrossEntropyWithSoftmaxBwdWithDowncastKernel(const Context& dev_ctx,
                                                  const DenseTensor& label,
                                                  const DenseTensor& softmax,
                                                  const DenseTensor& loss_grad,
                                                  DenseTensor* logits_grad) {
  constexpr int axis = -1;
  if (logits_grad->numel() == 0) {
    dev_ctx.template Alloc<phi::bfloat16>(logits_grad);
    return;
  }
  auto dtype = label.dtype();
  PD_VISIT_INTEGRAL_TYPES(
      dtype, "CrossEntropyWithSoftmaxBwdWithDowncastGPUKernel", ([&] {
        CrossEntropyWithSoftmaxBwdWithDowncastGPUKernel<T, data_t>(
            dev_ctx, label, softmax, loss_grad, axis, logits_grad);
      }));
}

}  // namespace phi

PD_REGISTER_KERNEL(cross_entropy_with_softmax_bwd_w_downcast,
                   GPU,
                   ALL_LAYOUT,
                   phi::CrossEntropyWithSoftmaxBwdWithDowncastKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
