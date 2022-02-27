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
#include "paddle/phi/kernels/sigmoid_cross_entropy_with_logits_kernel.h"

#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/operators/math.h"
#include "paddle/phi/core/hostdevice.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T>
__global__ void GPUSigmoidForward(const T *x_data,
                                  const T *label_data,
                                  const int ignore_index,
                                  const int limit,
                                  T *out_data,
                                  T *counts) {
  CUDA_KERNEL_LOOP(i, limit) {
    T x = x_data[i];
    T label = label_data[i];
    T eps = static_cast<T>(1e-5);
    T diff = label - static_cast<T>(ignore_index);
    if ((diff > -eps) && (diff < eps)) {
      out_data[i] = static_cast<T>(0.);
      counts[i] = 0;
    } else {
      T term1 = (x > 0) ? x : 0;
      T term2 = x * label;
      T term3 = paddle::operators::real_log(
          static_cast<T>(1) +
          paddle::operators::real_exp(static_cast<T>(-abs(x))));
      out_data[i] = term1 - term2 + term3;
      counts[i] = 1;
    }
  }
}

// Out = max(x, 0) - x * label + log(1 + exp(-abs(x)))
template <typename T, typename Context>
void SigmoidCrossEntropyWithLogitsKernel(const Context &dev_ctx,
                                         const DenseTensor &x,
                                         const DenseTensor &label,
                                         bool normalize,
                                         int ignore_index,
                                         DenseTensor *out) {
  auto out_data = dev_ctx.template Alloc<T>(out);

  // Temporary memory
  auto cnt_ptr = paddle::memory::Alloc(dev_ctx, label.numel() * sizeof(T));
  T *counts = reinterpret_cast<T *>(cnt_ptr->ptr());

  int limit = out->numel();
  int blocks = NumBlocks(limit);
  int threads = kNumCUDAThreads;
  GPUSigmoidForward<T><<<blocks, threads, 0, dev_ctx.stream()>>>(
      x.data<T>(), label.data<T>(), ignore_index, limit, out_data, counts);
  if (normalize) {
    auto norm_ptr = paddle::memory::Alloc(dev_ctx, sizeof(T));
    T *norm = reinterpret_cast<T *>(norm_ptr->ptr());
    Sum<T, kNumCUDAThreads><<<1, kNumCUDAThreads, 0, dev_ctx.stream()>>>(
        counts, limit, static_cast<T>(1e-5), norm);
    Div<T><<<blocks, threads, 0, dev_ctx.stream()>>>(out_data, limit, norm);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(sigmoid_cross_entropy_with_logits,
                   GPU,
                   ALL_LAYOUT,
                   phi::SigmoidCrossEntropyWithLogitsKernel,
                   float,
                   double) {}
