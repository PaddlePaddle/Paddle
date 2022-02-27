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

#include "paddle/phi/kernels/gpu/sigmoid_cross_entropy_with_logits.h"
#include "paddle/phi/kernels/sigmoid_cross_entropy_with_logits_grad_kernel.h"

#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/operators/math.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/hostdevice.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T>
__global__ void GPUSigmoidBackward(const T *x_data,
                                   const T *label_data,
                                   const int ignore_index,
                                   const T *dout_data,
                                   const int limit,
                                   T *dx_data,
                                   T *counts) {
  CUDA_KERNEL_LOOP(i, limit) {
    T x = x_data[i];
    T label = label_data[i];
    T dout = dout_data[i];
    T eps = static_cast<T>(1e-5);
    T diff = label - static_cast<T>(ignore_index);
    if ((diff > -eps) && (diff < eps)) {
      dx_data[i] = static_cast<T>(0.);
      counts[i] = 0;
    } else {
      T simoid_x = static_cast<T>(1) /
                   (static_cast<T>(1) + paddle::operators::real_exp(-x));
      T diff = simoid_x - label;
      dx_data[i] = dout * diff;
      counts[i] = 1;
    }
  }
}

// dx = sigmoid(x) - label
template <typename T, typename Context>
void SigmoidCrossEntropyWithLogitsGradKernel(const Context &dev_ctx,
                                             const DenseTensor &x,
                                             const DenseTensor &label,
                                             const DenseTensor &out_grad,
                                             bool normalize,
                                             int ignore_index,
                                             DenseTensor *in_grad) {
  auto dx_data = dev_ctx.template Alloc<T>(in_grad);

  // Temporary memory
  auto cnt_ptr = paddle::memory::Alloc(dev_ctx, x.numel() * sizeof(T));
  T *counts = reinterpret_cast<T *>(cnt_ptr->ptr());

  int limit = x.numel();
  int blocks = NumBlocks(limit);
  int threads = kNumCUDAThreads;
  GPUSigmoidBackward<T><<<blocks, threads, 0, dev_ctx.stream()>>>(
      x.data<T>(),
      label.data<T>(),
      ignore_index,
      out_grad.data<T>(),
      limit,
      dx_data,
      counts);
  if (normalize) {
    auto norm_ptr = paddle::memory::Alloc(dev_ctx, sizeof(T));
    T *norm = reinterpret_cast<T *>(norm_ptr->ptr());
    Sum<T, kNumCUDAThreads><<<1, kNumCUDAThreads, 0, dev_ctx.stream()>>>(
        counts, limit, static_cast<T>(1e-5), norm);
    Div<T><<<blocks, threads, 0, dev_ctx.stream()>>>(dx_data, limit, norm);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(sigmoid_cross_entropy_with_logits_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::SigmoidCrossEntropyWithLogitsGradKernel,
                   float,
                   double) {}
