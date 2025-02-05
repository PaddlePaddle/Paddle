// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/gpu/gpu_device_function.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/activation_functor.h"
#include "paddle/phi/kernels/funcs/elementwise/elementwise_op_impl.cu.h"

namespace phi {

template <typename T>
struct CudaSoftReluFunctor {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);
  float threshold;

  void SetAttrs(float threshold_) { threshold = threshold_; }

  // soft_relu(x) = log(1 + exp(max(min(x, threshold), -threshold)))
  // threshold should not be negative
  __device__ __forceinline__ T operator()(const T arg_x) {
    MPType x = static_cast<MPType>(arg_x);
    MPType t = static_cast<MPType>(threshold);
    MPType temp_min = x < t ? x : t;
    MPType temp_max = temp_min > -t ? temp_min : -t;
    return static_cast<T>(log(one + exp(temp_max)));
  }
};

template <typename T, typename Context>
void SoftReluCudaKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        float threshold,
                        DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  std::vector<const phi::DenseTensor*> ins = {&x};
  std::vector<phi::DenseTensor*> outs = {out};
  CudaSoftReluFunctor<T> functor;
  functor.SetAttrs(threshold);
  phi::funcs::LaunchSameDimsElementwiseCudaKernel<T>(
      dev_ctx, ins, &outs, functor);
}
}  // namespace phi

PD_REGISTER_KERNEL(soft_relu,
                   GPU,
                   ALL_LAYOUT,
                   phi::SoftReluCudaKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
