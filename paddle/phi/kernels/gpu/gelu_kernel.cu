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
struct GeluWithApproximateFunctor {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  inline HOSTDEVICE T operator()(T arg_x) {
    // this function is tanh approximation of gelu
    MPType x = static_cast<MPType>(arg_x);
    MPType one = static_cast<MPType>(1);
    MPType half = static_cast<MPType>(0.5);
    MPType kAlpha = static_cast<MPType>(M_2_SQRTPI * M_SQRT1_2);
    auto tanh_out =
        tanh(kAlpha * x * (one + static_cast<MPType>(GELU_CONSTANT) * x * x));
    MPType out = x * half * (one + tanh_out);
    return static_cast<T>(out);
  }
};

template <typename T>
struct GeluWithoutApproximateFunctor {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  inline HOSTDEVICE T operator()(T arg_x) {
    // actual gelu with approximation = false
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(x * normcdf(x));
  }
};

template <typename T, typename Context>
void GeluKernel(const Context& dev_ctx,
                const DenseTensor& x,
                bool approximate,
                DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  std::vector<const DenseTensor*> ins = {&x};
  std::vector<DenseTensor*> outs = {out};
  if (approximate) {
#ifdef __NVCC__
    if (std::is_same<T, dtype::float16>::value) {
      size_t n = x.numel();
      const auto* in_ptr = reinterpret_cast<const __half*>(x.data<T>());
      auto* out_ptr = reinterpret_cast<__half*>(out->data<T>());
      if (TryLaunchFP16FastGeluFwdVectorizeCUDAKernel(
              dev_ctx, in_ptr, out_ptr, n)) {
        return;
      }
    }
#endif
    phi::funcs::BroadcastKernel<ElementwiseType::kBinary, T, T>(
        dev_ctx, ins, &outs, 0, GeluWithApproximateFunctor<T>());
  } else {
    phi::funcs::BroadcastKernel<ElementwiseType::kBinary, T, T>(
        dev_ctx, ins, &outs, 0, GeluWithoutApproximateFunctor<T>());
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(gelu,
                   GPU,
                   ALL_LAYOUT,
                   phi::GeluKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
