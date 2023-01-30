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

#include "paddle/phi/kernels/label_smooth_grad_kernel.h"

<<<<<<< HEAD
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
=======
#include "paddle/fluid/operators/elementwise/elementwise_op_impl.cu.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

namespace phi {
template <typename T>
struct LabelSmoothGradFunctor {
  T epsilon;

  __forceinline__ LabelSmoothGradFunctor(float epsilon_data) {
    epsilon = static_cast<T>(epsilon_data);
  }

  __device__ __forceinline__ T operator()(const T x) const {
    return static_cast<T>(1 - epsilon) * x;
  }
};

template <typename T, typename Context>
void LabelSmoothGradKernel(const Context& ctx,
                           const DenseTensor& out_grad,
                           float epsilon,
                           DenseTensor* label_grad) {
  ctx.template Alloc<T>(label_grad);

  std::vector<const DenseTensor*> ins = {&out_grad};
  std::vector<DenseTensor*> outs = {label_grad};
  auto functor = LabelSmoothGradFunctor<T>(epsilon);
<<<<<<< HEAD
  phi::funcs::ElementwiseKernel<T>(ctx, ins, &outs, functor);
=======
  paddle::operators::LaunchSameDimsElementwiseCudaKernel<T>(
      ctx, ins, &outs, functor);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
}

}  // namespace phi

PD_REGISTER_KERNEL(label_smooth_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::LabelSmoothGradKernel,
                   float,
                   double) {}
