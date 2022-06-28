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

#include "paddle/phi/kernels/rrelu_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/distribution_helper.h"
#include "paddle/phi/kernels/funcs/for_range.h"

namespace phi {

template <typename T>
struct RReluTrainCudaFunctor {
 public:
  RReluTrainCudaFunctor(const T* in, T* out, T* noise)
      : in_(in), out_(out), noise_(noise) {
    zero_ = static_cast<T>(0);
  }

  __device__ void operator()(int64_t idx) {
    T x = in_[idx];
    if (x < zero_) {
      out_[idx] = noise_[idx] * x;
    } else {
      out_[idx] = x;
      noise_[idx] = 1.0;
    }
  }

 private:
  const T* in_;
  T* out_;
  T* noise_;
  T zero_;
};

template <typename T>
struct RReluTestCudaFunctor {
 public:
  RReluTestCudaFunctor(const T* in, T* out, T* noise, T mid_val)
      : in_(in), out_(out), noise_(noise), mid_val_(mid_val) {
    zero_ = static_cast<T>(0);
  }

  __device__ void operator()(int64_t idx) {
    T x = in_[idx];
    if (x < zero_) {
      out_[idx] = mid_val_ * x;
      noise_[idx] = mid_val_;
    } else {
      out_[idx] = x;
      noise_[idx] = 1.0;
    }
  }

 private:
  const T* in_;
  T* out_;
  T* noise_;
  T zero_;
  T mid_val_;
};

template <typename T, typename Context>
void RReluKernel(const Context& ctx,
                 const DenseTensor& x,
                 const float lower,
                 const float upper,
                 bool is_test,
                 DenseTensor* out,
                 DenseTensor* noise) {
  const T* x_data = x.data<T>();
  T* out_data = ctx.template Alloc<T>(out);
  T* noise_data = ctx.template Alloc<T>(noise);
  auto size = x.numel();
  if (size <= 0) return;

  phi::funcs::ForRange<Context> for_range(ctx, size);
  if (is_test) {
    T mid_val = static_cast<T>((lower + upper) / 2.0);
    RReluTestCudaFunctor<T> functor(x_data, out_data, noise_data, mid_val);
    for_range(functor);
  } else {
    using MT = typename kps::details::MPTypeTrait<T>::Type;
    funcs::uniform_distribution<MT> dist;
    funcs::uniform_real_transform<MT> trans(lower, upper);
    funcs::distribution_and_transform<T>(ctx, noise, dist, trans);
    RReluTrainCudaFunctor<T> functor(x_data, out_data, noise_data);
    for_range(functor);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(rrelu,
                   GPU,
                   ALL_LAYOUT,
                   phi::RReluKernel,
                   float,
                   phi::dtype::float16,
                   double) {}
