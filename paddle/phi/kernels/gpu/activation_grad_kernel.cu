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

#include "paddle/phi/kernels/activation_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/impl/activation_grad_impl.h"

#include "paddle/fluid/platform/device/gpu/gpu_device_function.h"

namespace phi {

template <typename T, typename Context, typename Functor>
void ActivationGradGPUImpl(const Context& dev_ctx,
                           const DenseTensor* x,
                           const DenseTensor* out,
                           const DenseTensor* d_out,
                           DenseTensor* d_x,
                           const Functor& functor) {
  if (static_cast<int>(Functor::FwdDeps()) &
      static_cast<int>(funcs::ActBwdOpFwdDeps::kDepOut)) {
    PADDLE_ENFORCE_NOT_NULL(
        out, errors::NotFound("The input DenseTensor Out can not be nullptr"));
  }
  PADDLE_ENFORCE_NOT_NULL(
      d_out, errors::NotFound("The input DenseTensor dOut can not be nullptr"));
  PADDLE_ENFORCE_NOT_NULL(
      d_x, errors::NotFound("The output DenseTensor dX can not be nullptr"));
  if (!out) {
    out = d_out;  // fake out
  }
  if (static_cast<int>(Functor::FwdDeps()) &
      static_cast<int>(funcs::ActBwdOpFwdDeps::kDepX)) {
    PADDLE_ENFORCE_NOT_NULL(
        x, errors::NotFound("The input DenseTensor X can not be nullptr"));
  } else {
    VLOG(10) << "Inplace activation of Op Functor: " << typeid(Functor).name();
    x = d_x;
  }

  dev_ctx.template Alloc<T>(d_x);

  std::vector<const DenseTensor*> ins = {d_out};
  std::vector<DenseTensor*> outs = {d_x};

  if (static_cast<int>(Functor::FwdDeps()) ==
      static_cast<int>(funcs::ActBwdOpFwdDeps::kDepOut)) {
    // Only need forward output Out
    ins.push_back(out);
    funcs::ElementwiseKernel<T>(dev_ctx, ins, &outs, functor);
  } else if (static_cast<int>(Functor::FwdDeps()) ==
             static_cast<int>(funcs::ActBwdOpFwdDeps::kDepX)) {
    // Only need forward input X
    ins.push_back(x);
    funcs::ElementwiseKernel<T>(dev_ctx, ins, &outs, functor);
  } else {
    funcs::ElementwiseKernel<T>(dev_ctx, ins, &outs, functor);
  }
}

#define DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DepX(name, functor_class) \
  template <typename T, typename Context>                           \
  void name##GradKernel(const Context& dev_ctx,                     \
                        const DenseTensor& x,                       \
                        const DenseTensor& dout,                    \
                        DenseTensor* dx) {                          \
    functor_class functor;                                          \
    ActivationGradGPUImpl<T, Context, functor_class>(               \
        dev_ctx, &x, nullptr, &dout, dx, functor);                  \
  }

#define DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DepOut(name, functor_class) \
  template <typename T, typename Context>                             \
  void name##GradKernel(const Context& dev_ctx,                       \
                        const DenseTensor& out,                       \
                        const DenseTensor& dout,                      \
                        DenseTensor* dx) {                            \
    functor_class functor;                                            \
    ActivationGradGPUImpl<T, Context, functor_class>(                 \
        dev_ctx, nullptr, &out, &dout, dx, functor);                  \
  }

DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DepOut(Relu, funcs::CudaReluGradFunctor<T>);
DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DepX(Cos, funcs::CudaCosGradFunctor<T>);
DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DepX(Tan, funcs::CudaTanGradFunctor<T>);
DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DepX(Acos, funcs::CudaAcosGradFunctor<T>);
DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DepX(Sin, funcs::CudaSinGradFunctor<T>);
DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DepX(Asin, funcs::CudaAsinGradFunctor<T>);
DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DepX(Atan, funcs::CudaAtanGradFunctor<T>);
DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DepX(Sinh, funcs::CudaSinhGradFunctor<T>);
DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DepX(Cosh, funcs::CudaCoshGradFunctor<T>);
DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DepX(Asinh, funcs::CudaAsinhGradFunctor<T>);
DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DepX(Acosh, funcs::CudaAcoshGradFunctor<T>);
DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DepX(Atanh, funcs::CudaAtanhGradFunctor<T>);

}  // namespace phi
PD_REGISTER_KERNEL(cos_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::CosGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
PD_REGISTER_KERNEL(tan_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::TanGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
PD_REGISTER_KERNEL(acos_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::AcosGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
PD_REGISTER_KERNEL(sin_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::SinGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
PD_REGISTER_KERNEL(asin_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::AsinGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
PD_REGISTER_KERNEL(atan_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::AtanGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
PD_REGISTER_KERNEL(sinh_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::SinhGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
PD_REGISTER_KERNEL(cosh_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::CoshGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
PD_REGISTER_KERNEL(asinh_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::AsinhGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
PD_REGISTER_KERNEL(acosh_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::AcoshGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
PD_REGISTER_KERNEL(atanh_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::AtanhGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(relu_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::ReluGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
PD_REGISTER_KERNEL(relu_double_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::ReluDoubleGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
#else
PD_REGISTER_KERNEL(relu_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::ReluGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(relu_double_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::ReluDoubleGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#endif
