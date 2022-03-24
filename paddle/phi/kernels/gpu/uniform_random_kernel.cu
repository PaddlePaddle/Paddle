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

#include "paddle/phi/kernels/uniform_random_kernel.h"

#include "gflags/gflags.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/distribution_helper.h"
#include "paddle/phi/kernels/funcs/index_impl.cu.h"

DECLARE_bool(use_curand);

namespace phi {

template <typename T>
struct UniformGenerator {
  T min_, max_;
  unsigned int seed_;
  T diag_val_;
  unsigned int diag_num_;
  unsigned int diag_step_;
  __host__ __device__ UniformGenerator(
      T min, T max, int seed, int diag_num, int diag_step, T diag_val)
      : min_(min),
        max_(max),
        seed_(seed),
        diag_num_(diag_num),
        diag_step_(diag_step),
        diag_val_(diag_val) {}

  __host__ __device__ T operator()(const unsigned int n) const {
    thrust::minstd_rand rng;
    rng.seed(seed_);
    thrust::uniform_real_distribution<T> dist(min_, max_);
    rng.discard(n);
    T out = dist(rng);
    unsigned int remainder = n % (diag_step_ + 1);
    if (remainder == 0 && diag_num_ > n / (diag_step_ + 1)) {
      out = diag_val_;
    }
    return out;
  }
};

template <typename T>
struct UniformGeneratorOffset {
  T min_, max_;
  unsigned int seed_;
  T diag_val_;
  unsigned int diag_num_;
  unsigned int diag_step_;
  int offset_;
  __host__ __device__ UniformGeneratorOffset(T min,
                                             T max,
                                             int seed,
                                             int diag_num,
                                             int diag_step,
                                             T diag_val,
                                             int offset)
      : min_(min),
        max_(max),
        seed_(seed),
        diag_num_(diag_num),
        diag_step_(diag_step),
        diag_val_(diag_val),
        offset_(offset) {}

  __host__ __device__ T operator()(const unsigned int n) const {
    thrust::minstd_rand rng;
    rng.seed(seed_);
    thrust::uniform_real_distribution<T> dist(min_, max_);
    rng.discard(n + offset_);
    T out = dist(rng);
    unsigned int remainder = n % (diag_step_ + 1);
    if (remainder == 0 && diag_num_ > n / (diag_step_ + 1)) {
      out = diag_val_;
    }
    return out;
  }
};

template <typename T, typename Context>
void UniformRandomRawKernel(const Context& dev_ctx,
                            const ScalarArray& shape,
                            DataType dtype,
                            float min,
                            float max,
                            int seed,
                            int diag_num,
                            int diag_step,
                            float diag_val,
                            DenseTensor* out) {
  out->Resize(phi::make_ddim(shape.GetData()));
  T* data = dev_ctx.template Alloc<T>(out);
  auto size = out->numel();
  bool seed_flag = false;
  if (seed == 0) {
    std::random_device rd;
    seed = rd();
    seed_flag = true;
  }

  auto generator = dev_ctx.GetGenerator();
  if (generator->GetIsInitPy() && seed_flag) {
    if (FLAGS_use_curand) {
      using MT = typename kps::details::MPTypeTrait<T>::Type;
      funcs::uniform_distribution<MT> dist;
      funcs::uniform_real_transform<MT> trans(min, max);
      funcs::distribution_and_transform<T>(dev_ctx, out, dist, trans);
    } else {
      auto seed_offset = generator->IncrementOffset(1);
      int64_t gen_offset = size * seed_offset.second;
      auto func = UniformGeneratorOffset<T>(min,
                                            max,
                                            seed_offset.first,
                                            diag_num,
                                            diag_step,
                                            diag_val,
                                            gen_offset);
      IndexKernel<T, UniformGeneratorOffset<T>>(dev_ctx, out, func);
    }
  } else {
    auto func =
        UniformGenerator<T>(min, max, seed, diag_num, diag_step, diag_val);
    IndexKernel<T, UniformGenerator<T>>(dev_ctx, out, func);
  }
}

template <typename T, typename Context>
void UniformRandomKernel(const Context& dev_ctx,
                         const ScalarArray& shape,
                         DataType dtype,
                         float min,
                         float max,
                         int seed,
                         DenseTensor* out) {
  UniformRandomRawKernel<T>(
      dev_ctx, shape, dtype, min, max, seed, 0, 0, 0.0f, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(uniform_random_raw,
                   GPU,
                   ALL_LAYOUT,
                   phi::UniformRandomRawKernel,
                   float,
                   double) {}

PD_REGISTER_KERNEL(
    uniform_random, GPU, ALL_LAYOUT, phi::UniformRandomKernel, float, double) {}
