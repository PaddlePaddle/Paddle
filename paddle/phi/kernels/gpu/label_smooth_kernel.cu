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

#include "paddle/phi/kernels/label_smooth_kernel.h"

#include <vector>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"

namespace phi {

template <typename T>
struct LabelSmoothFunctor {
  T epsilon;
  T label_dim;

  __forceinline__ LabelSmoothFunctor(float epsilon_data, int label_dim_data) {
    epsilon = static_cast<T>(epsilon_data);
    label_dim = static_cast<T>(label_dim_data);
  }

  __device__ __forceinline__ T operator()(const T x) const {
    return (static_cast<T>(1 - epsilon) * x +
            static_cast<T>(epsilon / label_dim));
  }
};

template <typename T>
__global__ void LabelSmoothRunDistKernel(const int N,
                                         const float epsilon,
                                         const int dist_numel,
                                         const T* src,
                                         const T* dist_data,
                                         T* dst) {
  CUDA_KERNEL_LOOP(idx, N) {
    int dist_idx = idx % dist_numel;
    dst[idx] = static_cast<T>(1 - epsilon) * src[idx] +
               static_cast<T>(epsilon) * dist_data[dist_idx];
  }
}

template <typename T, typename Context>
void LabelSmoothKernel(const Context& ctx,
                       const DenseTensor& label,
                       const paddle::optional<DenseTensor>& prior_dist,
                       float epsilon,
                       DenseTensor* out) {
  auto label_dim = label.dims()[label.dims().size() - 1];
  auto size_prob = label.numel();
  const T* in_data = label.data<T>();
  T* out_data = ctx.template Alloc<T>(out);

  if (prior_dist.get_ptr()) {
    int threads = 512;
    int grid = (size_prob + threads - 1) / threads;
    auto stream = ctx.stream();
    const auto* dist_t = prior_dist.get_ptr();
    auto dist_numel = dist_t->numel();
    const T* dist_data = dist_t->data<T>();
    LabelSmoothRunDistKernel<T><<<grid, threads, 0, stream>>>(
        size_prob, epsilon, dist_numel, in_data, dist_data, out_data);

  } else {
    std::vector<const DenseTensor*> ins = {&label};
    std::vector<DenseTensor*> outs = {out};
    auto functor = LabelSmoothFunctor<T>(epsilon, label_dim);
    phi::funcs::ElementwiseKernel<T>(ctx, ins, &outs, functor);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    label_smooth, GPU, ALL_LAYOUT, phi::LabelSmoothKernel, float, double) {}
