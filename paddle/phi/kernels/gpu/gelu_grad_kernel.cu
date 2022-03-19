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

#include "paddle/phi/kernels/gelu_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/gpu/gelu_funcs.h"

DECLARE_bool(use_fast_math);

namespace phi {

template <typename T>
struct GeluWithApproximateGradFunctor {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  inline HOSTDEVICE T operator()(T arg_x, T arg_dout) {
    MPType x = static_cast<MPType>(arg_x);
    MPType dout = static_cast<MPType>(arg_dout);
    MPType one = static_cast<MPType>(1);
    MPType half = static_cast<MPType>(0.5);
    MPType kAlpha = static_cast<MPType>(M_2_SQRTPI * M_SQRT1_2);
    MPType kBeta =
        kAlpha * static_cast<MPType>(GELU_CONSTANT) * static_cast<MPType>(3);
    auto cube_x = x * x * x;
    auto tanh_out =
        tanh(kAlpha * ((static_cast<MPType>(GELU_CONSTANT) * cube_x) + x));
    auto ans =
        half * (one + tanh_out +
                (one - tanh_out * tanh_out) * (x * kAlpha + kBeta * cube_x));
    return static_cast<T>(ans * dout);
  }
};

template <typename T>
struct GeluWithoutApproximateGradFunctor {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  inline HOSTDEVICE T operator()(T arg_x, T arg_dout) {
    MPType x = static_cast<MPType>(arg_x);
    MPType dout = static_cast<MPType>(arg_dout);
    constexpr MPType kBeta = M_2_SQRTPI * M_SQRT1_2 * static_cast<MPType>(0.5);
    const MPType cdf = normcdf(x);
    const MPType pdf = exp(static_cast<MPType>(-0.5) * x * x) * kBeta;
    return static_cast<T>(dout * (cdf + x * pdf));
  }
};

template <typename T, typename Context>
void GeluGradKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& out_grad,
                    bool approximate,
                    DenseTensor* x_grad) {
  dev_ctx.template Alloc<T>(x_grad);
  std::vector<const DenseTensor*> ins = {&x, &out_grad};
  std::vector<DenseTensor*> outs = {x_grad};
  if (approximate) {
#ifdef __NVCC__
    if (std::is_same<T, dtype::float16>::value) {
      size_t n = x.numel();
      const auto* x_ptr = reinterpret_cast<const __half*>(x.data<T>());
      const auto* y_g_ptr = reinterpret_cast<const __half*>(out_grad.data<T>());
      auto* x_g_ptr = reinterpret_cast<__half*>(x_grad->data<T>());
      if (TryLaunchFP16FastGeluBwdVectorizeCUDAKernel(
              dev_ctx, x_ptr, y_g_ptr, x_g_ptr, n)) {
        return;
      }
    }
#endif
    phi::funcs::BroadcastKernel<ElementwiseType::kBinary, T, T>(
        dev_ctx, ins, &outs, 0, GeluWithApproximateGradFunctor<T>());
  } else {
    phi::funcs::BroadcastKernel<ElementwiseType::kBinary, T, T>(
        dev_ctx, ins, &outs, 0, GeluWithoutApproximateGradFunctor<T>());
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(gelu_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::GeluGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
