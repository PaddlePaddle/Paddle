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
struct CudaSoftReluGradFunctor {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);
  float threshold;

  void SetAttrs(float threshold_) { threshold = threshold_; }

  // dx = (out > -threshold && out < threshold) ? dout * (1 - exp(-out)) : 0
  // threshold should not be negative
  __device__ __forceinline__ T operator()(const T arg_dout, const T arg_out) {
    MPType dout = static_cast<MPType>(arg_dout);
    MPType out = static_cast<MPType>(arg_out);
    MPType t = static_cast<MPType>(threshold);
    return (out > -t && out < t) ? static_cast<T>(dout * (one - exp(-out)))
                                 : static_cast<T>(0.0f);
  }
};

template <typename T, typename Context>
void SoftReluGradCudaKernel(const Context& dev_ctx,
                            const DenseTensor& x_in UNUSED,
                            const DenseTensor& out_in,
                            const DenseTensor& out_grad,
                            float threshold,
                            DenseTensor* x_grad) {
  dev_ctx.template Alloc<T>(x_grad);
  CudaSoftReluGradFunctor<T> functor;
  functor.SetAttrs(threshold);

  std::vector<const phi::DenseTensor*> ins = {&out_grad};
  std::vector<phi::DenseTensor*> outs = {x_grad};

  // Only need forward output Out
  ins.push_back(&out_in);
  phi::funcs::LaunchSameDimsElementwiseCudaKernel<T>(
      dev_ctx, ins, &outs, functor);
}
}  // namespace phi

PD_REGISTER_KERNEL(soft_relu_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::SoftReluGradCudaKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
