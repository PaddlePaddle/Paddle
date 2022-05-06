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


#include "paddle/phi/kernels/rrelu_kernel.h"
// #include "paddle/phi/kernels/gpu/rrelu_impl.cu.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/funcs/distribution_helper.h"


namespace phi {

template <typename T>
struct RReluTrainCudaFunctor {
 public:
  RReluTrainCudaFunctor(const T* in,
                        T* out,
                        T* mask)
      : in_(in), out_(out), mask_(mask) {
         zero_ = static_cast<T>(0);
         one_ = static_cast<T>(1);
      }

  __device__ void operator()(int64_t idx) {
   if (in_[idx] < zero_) {
      out_[idx] = in_[idx] * mask_[idx];
   } else {
      mask_[idx] = one_;
      out_[idx] = in_[idx];
   }
  }

 private:
  const T* in_;
  T* out_;
  T* mask_;
  T zero_;
  T one_;
};


template <typename T>
struct RReluTestCudaFunctor {
 public:
  RReluTestCudaFunctor(const T* in,
                        T* out,
                        T* mask,
                        T mid_value)
      : in_(in), out_(out), mask_(mask), mid_value_(mid_value) {
         zero_ = static_cast<T>(0);
         one_ = static_cast<T>(1);
      }

  __device__ void operator()(int64_t idx) {
   if (in_[idx] < zero_) {
      mask_[idx] = mid_value_;
      out_[idx] = in_[idx] * mid_value_;
   } else {
      mask_[idx] = one_;
      out_[idx] = in_[idx];
   }
  }

 private:
  const T* in_;
  T* out_;
  T* mask_;
  T mid_value_;
  T zero_;
  T one_;
};



template <typename T, typename Context>
void RReluKernel(const Context& ctx, 
                 const DenseTensor& x, 
                 const float lower,
                 const float upper,
                 bool is_test,
                 DenseTensor* out,
                 DenseTensor* mask) {
  const T* x_data = x.data<T>();
  T* out_data = ctx.template Alloc<T>(out);
  T* mask_data = ctx.template Alloc<T>(mask);
  auto size = x.numel();
  if (size <= 0)
      return;
  phi::funcs::ForRange<Context> for_range(ctx, size);

      if (!is_test) {
            using MT = typename kps::details::MPTypeTrait<T>::Type;
            funcs::uniform_distribution<MT> dist;
            funcs::uniform_real_transform<MT> trans(lower, upper);
            funcs::distribution_and_transform<T>(ctx, mask, dist, trans);

            RReluTrainCudaFunctor<T> functor(x_data, out_data, mask_data);
            for_range(functor);
      } else {
            T mid_value = static_cast<T>((lower + upper) / 2.0f);
            RReluTestCudaFunctor<T> functor(x_data, out_data, mask_data, mid_value);
            for_range(functor);
      }
}


}  // namespace phi

PD_REGISTER_KERNEL(rrelu,
                   GPU,
                   ALL_LAYOUT,
                   phi::RReluKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
                  //  phi::dtype::bfloat16) {}
