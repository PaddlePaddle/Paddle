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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/activation_kernel.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/impl/activation_impl.h"

#include "paddle/fluid/platform/device/gpu/gpu_device_function.h"

namespace phi {

template <typename T, typename Context, typename Functor>
void ActivationGPUImpl(const Context& dev_ctx,
                       const DenseTensor& x,
                       DenseTensor* out,
                       const Functor& functor) {
  PADDLE_ENFORCE_NOT_NULL(out,
                          errors::NotFound("Output Out should not be nullptr"));
  dev_ctx.template Alloc<T>(out);
  std::vector<const DenseTensor*> ins = {&x};
  std::vector<DenseTensor*> outs = {out};
  funcs::ElementwiseKernel<T>(dev_ctx, ins, &outs, functor);
}

#define DEFINE_GPU_ACTIVATION_KERNEL(name, functor_class)                   \
  template <typename T, typename Context>                                   \
  void name##Kernel(                                                        \
      const Context& dev_ctx, const DenseTensor& x, DenseTensor* out) {     \
    functor_class functor;                                                  \
    ActivationGPUImpl<T, Context, functor_class>(dev_ctx, x, out, functor); \
  }

DEFINE_GPU_ACTIVATION_KERNEL(Cos, funcs::CudaCosFunctor<T>)
DEFINE_GPU_ACTIVATION_KERNEL(Tan, funcs::CudaTanFunctor<T>)
DEFINE_GPU_ACTIVATION_KERNEL(Acos, funcs::CudaAcosFunctor<T>)
DEFINE_GPU_ACTIVATION_KERNEL(Sin, funcs::CudaSinFunctor<T>)
DEFINE_GPU_ACTIVATION_KERNEL(Asin, funcs::CudaAsinFunctor<T>)
DEFINE_GPU_ACTIVATION_KERNEL(Atan, funcs::CudaAtanFunctor<T>)
DEFINE_GPU_ACTIVATION_KERNEL(Sinh, funcs::CudaSinhFunctor<T>)
DEFINE_GPU_ACTIVATION_KERNEL(Cosh, funcs::CudaCoshFunctor<T>)
DEFINE_GPU_ACTIVATION_KERNEL(Asinh, funcs::CudaAsinhFunctor<T>)
DEFINE_GPU_ACTIVATION_KERNEL(Acosh, funcs::CudaAcoshFunctor<T>)
DEFINE_GPU_ACTIVATION_KERNEL(Atanh, funcs::CudaAtanhFunctor<T>)
DEFINE_GPU_ACTIVATION_KERNEL(Relu, funcs::CudaReluFunctor<T>)

}  // namespace phi

#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(relu,
                   GPU,
                   ALL_LAYOUT,
                   phi::ReluKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
#else
PD_REGISTER_KERNEL(relu,
                   GPU,
                   ALL_LAYOUT,
                   phi::ReluKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#endif
PD_REGISTER_KERNEL(
    sin, GPU, ALL_LAYOUT, phi::SinKernel, float, double, phi::dtype::float16) {}
PD_REGISTER_KERNEL(
    cos, GPU, ALL_LAYOUT, phi::CosKernel, float, double, phi::dtype::float16) {}
PD_REGISTER_KERNEL(
    tan, GPU, ALL_LAYOUT, phi::TanKernel, float, double, phi::dtype::float16) {}
PD_REGISTER_KERNEL(acos,
                   GPU,
                   ALL_LAYOUT,
                   phi::AcosKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
PD_REGISTER_KERNEL(asin,
                   GPU,
                   ALL_LAYOUT,
                   phi::AsinKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
PD_REGISTER_KERNEL(atan,
                   GPU,
                   ALL_LAYOUT,
                   phi::AtanKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
PD_REGISTER_KERNEL(sinh,
                   GPU,
                   ALL_LAYOUT,
                   phi::SinhKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
PD_REGISTER_KERNEL(cosh,
                   GPU,
                   ALL_LAYOUT,
                   phi::CoshKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
PD_REGISTER_KERNEL(asinh,
                   GPU,
                   ALL_LAYOUT,
                   phi::AsinhKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
PD_REGISTER_KERNEL(acosh,
                   GPU,
                   ALL_LAYOUT,
                   phi::AcoshKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
PD_REGISTER_KERNEL(atanh,
                   GPU,
                   ALL_LAYOUT,
                   phi::AtanhKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
