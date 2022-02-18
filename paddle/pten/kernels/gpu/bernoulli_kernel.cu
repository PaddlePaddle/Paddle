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

#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <algorithm>
#include <vector>
#include "paddle/pten/backends/gpu/gpu_context.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/kernels/bernoulli_kernel.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/transform.h"

namespace pten {

template <typename T>
struct BernoulliCudaFunctor {
  unsigned int seed_;
  unsigned int offset_;
  __host__ __device__ BernoulliCudaFunctor(unsigned int seed,
                                           unsigned int offset)
      : seed_(seed), offset_(offset) {}

  __host__ __device__ T operator()(const unsigned int n, const T p) const {
    // NOTE(zhiqiu): currently, PADDLE_ENFORCE in cuda kernel may print several
    // lines of error messages if, and it should be refined.
    PADDLE_ENFORCE(p >= 0.0 && p <= 1.0,
                   "The probability should be >=0 and <= 1, but got %f",
                   p);
    thrust::minstd_rand rng;
    rng.seed(seed_);
    thrust::uniform_real_distribution<T> dist(0.0, 1.0);
    rng.discard(n + offset_);
    return static_cast<T>(dist(rng) < p);
  }
};

template <typename T, typename Context>
void BernoulliKernel(const Context& ctx,
                     const DenseTensor& x,
                     DenseTensor* out) {
  auto numel = x.numel();
  auto* x_data = x.data<T>();
  T* out_data = ctx.template Alloc<T>(out);

  auto gen_cuda = ctx.GetGenerator();
  auto seed_offset = gen_cuda->IncrementOffset(1);
  int64_t gen_offset = numel * seed_offset.second;
  paddle::platform::Transform<pten::GPUContext> trans;
  thrust::counting_iterator<int64_t> index_sequence_begin(0);
  trans(ctx,
        index_sequence_begin,
        index_sequence_begin + numel,
        x_data,
        out_data,
        BernoulliCudaFunctor<T>(static_cast<int64_t>(seed_offset.first),
                                static_cast<int64_t>(gen_offset)));
}

}  // namespace pten

PT_REGISTER_KERNEL(
    bernoulli, GPU, ALL_LAYOUT, pten::BernoulliKernel, float, double) {}
