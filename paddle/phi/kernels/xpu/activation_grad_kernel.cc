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

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/activation_functor.h"

namespace phi {

template <typename T, typename Context, typename Functor>
void ActivationGradXPUImpl(const Context& dev_ctx,
                           const DenseTensor* x,
                           const DenseTensor* out,
                           const DenseTensor* d_out,
                           DenseTensor* d_x,
                           const Functor& functor) {
  PADDLE_ENFORCE_NOT_NULL(
      d_out, errors::NotFound("The input DenseTensor dOut can not be nullptr"));
  PADDLE_ENFORCE_NOT_NULL(
      d_x, errors::NotFound("The output DenseTensor dX can not be nullptr"));
  if (!out) {
    out = d_out;  // fake out
  }
  dev_ctx.template Alloc<T>(d_x);
  functor(dev_ctx, x, out, d_out, d_x);
}

#define DEFINE_XPU_ACTIVATION_GRAD_KERNEL_DEPX(name, functor_class) \
  template <typename T, typename Context>                           \
  void name##GradKernel(const Context& dev_ctx,                     \
                        const DenseTensor& x,                       \
                        const DenseTensor& dout,                    \
                        DenseTensor* dx) {                          \
    functor_class<T> functor;                                       \
    ActivationGradXPUImpl<T, Context, functor_class<T>>(            \
        dev_ctx, &x, nullptr, &dout, dx, functor);                  \
  }

#define DEFINE_XPU_ACT_GRAD_KERNEL_WITH_ONE_ATTRS_DEPX(  \
    name, functor_class, attr)                           \
  template <typename T, typename Context>                \
  void name##GradKernel(const Context& dev_ctx,          \
                        const DenseTensor& x,            \
                        const DenseTensor& dout,         \
                        float attr,                      \
                        DenseTensor* dx) {               \
    functor_class<T> functor;                            \
    auto attrs = functor.GetAttrs();                     \
    *(attrs[0].second) = attr;                           \
    ActivationGradXPUImpl<T, Context, functor_class<T>>( \
        dev_ctx, &x, nullptr, &dout, dx, functor);       \
  }

#define DEFINE_XPU_ACT_GRAD_KERNEL_WITH_TWO_ATTRS_DEPX(  \
    name, functor_class, attr1, attr2)                   \
  template <typename T, typename Context>                \
  void name##GradKernel(const Context& dev_ctx,          \
                        const DenseTensor& x,            \
                        const DenseTensor& dout,         \
                        float attr1,                     \
                        float attr2,                     \
                        DenseTensor* dx) {               \
    functor_class<T> functor;                            \
    auto attrs = functor.GetAttrs();                     \
    *(attrs[0].second) = attr1;                          \
    *(attrs[1].second) = attr2;                          \
    ActivationGradXPUImpl<T, Context, functor_class<T>>( \
        dev_ctx, &x, nullptr, &dout, dx, functor);       \
  }

#define DEFINE_XPU_ACTIVATION_GRAD_KERNEL_DEPOUT(name, functor_class) \
  template <typename T, typename Context>                             \
  void name##GradKernel(const Context& dev_ctx,                       \
                        const DenseTensor& out,                       \
                        const DenseTensor& dout,                      \
                        DenseTensor* dx) {                            \
    functor_class<T> functor;                                         \
    ActivationGradXPUImpl<T, Context, functor_class<T>>(              \
        dev_ctx, nullptr, &out, &dout, dx, functor);                  \
  }

#define DEFINE_XPU_ACT_GRAD_KERNEL_WITH_ONE_ATTRS_DEPOUT( \
    name, functor_class, attr)                            \
  template <typename T, typename Context>                 \
  void name##GradKernel(const Context& dev_ctx,           \
                        const DenseTensor& out,           \
                        const DenseTensor& dout,          \
                        float attr,                       \
                        DenseTensor* dx) {                \
    functor_class<T> functor;                             \
    auto attrs = functor.GetAttrs();                      \
    *(attrs[0].second) = attr;                            \
    ActivationGradXPUImpl<T, Context, functor_class<T>>(  \
        dev_ctx, nullptr, &out, &dout, dx, functor);      \
  }

#define DEFINE_XPU_ACT_GRAD_KERNEL_WITH_TWO_ATTRS_DEPOUT( \
    name, functor_class, attr1, attr2)                    \
  template <typename T, typename Context>                 \
  void name##GradKernel(const Context& dev_ctx,           \
                        const DenseTensor& out,           \
                        const DenseTensor& dout,          \
                        float attr1,                      \
                        float attr2,                      \
                        DenseTensor* dx) {                \
    functor_class<T> functor;                             \
    auto attrs = functor.GetAttrs();                      \
    *(attrs[0].second) = attr1;                           \
    *(attrs[1].second) = attr2;                           \
    ActivationGradXPUImpl<T, Context, functor_class<T>>(  \
        dev_ctx, nullptr, &out, &dout, dx, functor);      \
  }

#define DEFINE_XPU_ACTIVATION_GRAD_KERNEL_NODEP(name, functor_class)      \
  template <typename T, typename Context>                                 \
  void name##GradKernel(                                                  \
      const Context& dev_ctx, const DenseTensor& dout, DenseTensor* dx) { \
    functor_class<T> functor;                                             \
    ActivationGradXPUImpl<T, Context, functor_class<T>>(                  \
        dev_ctx, nullptr, nullptr, &dout, dx, functor);                   \
  }

template <typename Context, typename T, typename XPUType>
int xpu_activation_backward(const Context& dev_ctx,
                            const DenseTensor* x,
                            const DenseTensor* out,
                            const DenseTensor* dout,
                            DenseTensor* dx,
                            std::function<int(xpu::Context*,
                                              const XPUType*,
                                              const XPUType*,
                                              const XPUType*,
                                              XPUType*,
                                              int)> func) {
  /* TODO: relu tanh sigmoid are inplace */
  const XPUType* x_data = nullptr;
  const XPUType* y_data = nullptr;
  const XPUType* y_grad = nullptr;
  if (x != nullptr) x_data = reinterpret_cast<const XPUType*>(x->data<T>());
  if (out != nullptr) y_data = reinterpret_cast<const XPUType*>(out->data<T>());
  if (dout != nullptr)
    y_grad = reinterpret_cast<const XPUType*>(dout->data<T>());
  XPUType* x_grad = reinterpret_cast<XPUType*>(dx->data<T>());

  int r =
      func(dev_ctx.x_context(), x_data, y_data, y_grad, x_grad, dx->numel());
  return r;
}

template <typename T>
struct XPULogGradFunctor : public funcs::BaseActivationFunctor<T> {
  template <typename Context>
  void operator()(const Context& dev_ctx,
                  const DenseTensor* x,
                  const DenseTensor* out,
                  const DenseTensor* dOut,
                  DenseTensor* dX) const {
    const T* x_data = nullptr;
    const T* dout_data = nullptr;
    if (x != nullptr) x_data = x->data<T>();
    if (dOut != nullptr) dout_data = dOut->data<T>();

    T* dx_data = dev_ctx.template Alloc<T>(dX);
    xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
    float* tmp = RAII_GUARD.alloc_l3_or_gm<T>(x->numel());
    int r = xpu::constant<T>(
        dev_ctx.x_context(), tmp, x->numel(), static_cast<T>(1.0));
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");

    auto x_dims = common::vectorize<int>(x->dims());

    // use [1] to replace [], because xpu not support []
    if (x_dims.size() == 0) {
      x_dims = std::vector<int>({1});
    }
    // dx.device(d) = dout * (static_cast<T>(1) / x);
    r = xpu::broadcast_div(dev_ctx.x_context(),
                           reinterpret_cast<const float*>(tmp),
                           reinterpret_cast<const float*>(x_data),
                           reinterpret_cast<float*>(tmp),
                           x_dims,
                           x_dims);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_div");
    r = xpu::broadcast_mul(dev_ctx.x_context(),
                           reinterpret_cast<const float*>(tmp),
                           reinterpret_cast<const float*>(dout_data),
                           reinterpret_cast<float*>(dx_data),
                           x_dims,
                           x_dims);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_mul");
  }
};

template <typename T>
struct XPULeakyReluGradFunctor : public funcs::BaseActivationFunctor<T> {
  float alpha;
  typename funcs::BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"alpha", &alpha}};
  }

  template <typename Context>
  void operator()(const Context& dev_ctx,
                  const DenseTensor* x,
                  const DenseTensor* out,
                  const DenseTensor* dout,
                  DenseTensor* dx) const {
    const T* x_data = nullptr;
    const T* y_grad = nullptr;
    if (x != nullptr) x_data = x->data<T>();
    if (dout != nullptr) y_grad = dout->data<T>();
    T* x_grad = dx->data<T>();
    auto xpu_context = dev_ctx.x_context();

    // The signs of x and y are the same,
    // y == nullptr here,
    // so we give 2 x to the api
    int r = xpu::leaky_relu_grad(xpu_context,
                                 reinterpret_cast<const float*>(x_data),
                                 reinterpret_cast<const float*>(x_data),
                                 reinterpret_cast<const float*>(y_grad),
                                 reinterpret_cast<float*>(x_grad),
                                 dx->numel(),
                                 alpha);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "leaky_relu_grad");
  }
};

template <typename T>
struct XPUHardSigmoidGradFunctor : public funcs::BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  float slope;
  float offset;
  typename funcs::BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"slope", &slope}, {"offset", &offset}};
  }

  template <typename Context>
  void operator()(const Context& dev_ctx,
                  const DenseTensor* x,
                  const DenseTensor* out,
                  const DenseTensor* dout,
                  DenseTensor* dx) const {
    const T* y_data = out->data<T>();
    const T* y_grad = dout->data<T>();
    T* x_grad = dx->data<T>();

    auto xpu_context = dev_ctx.x_context();
    int r = xpu::hard_sigmoid_grad(
        xpu_context,
        reinterpret_cast<const XPUType*>(
            y_data),  // hardsigmoid_grad do not need x_data
        reinterpret_cast<const XPUType*>(y_data),
        reinterpret_cast<const XPUType*>(y_grad),
        reinterpret_cast<XPUType*>(x_grad),
        dx->numel(),
        slope);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "hardsigmoid_grad");
  }
};

template <typename T>
struct XPUHardSwishGradFunctor : public funcs::BaseActivationFunctor<T> {
  float threshold;
  float scale;
  float offset;

  typename funcs::BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}, {"scale", &scale}, {"offset", &offset}};
  }
  template <typename Context>
  void operator()(const Context& dev_ctx,
                  const DenseTensor* x,
                  const DenseTensor* out,
                  const DenseTensor* dout,
                  DenseTensor* dx) const {
    using XPUType = typename XPUTypeTrait<T>::Type;
    PADDLE_ENFORCE_EQ(
        threshold,
        6.0f,
        errors::External("Not support threshold [%f] in XPU", threshold));
    PADDLE_ENFORCE_EQ(
        scale, 6.0f, errors::External("Not support scale [%f] in XPU", scale));
    PADDLE_ENFORCE_EQ(
        offset,
        3.0f,
        errors::External("Not support offset [%f] in XPU", offset));
    int r = xpu_activation_backward<Context, T, XPUType>(
        dev_ctx, x, out, dout, dx, xpu::hard_swish_grad<XPUType>);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "hard_swish_grad");
  }
};

template <typename T>
struct XPUReciprocalGradFunctor : public funcs::BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  template <typename Context>
  void operator()(const Context& dev_ctx,
                  const DenseTensor* x,
                  const DenseTensor* out,
                  const DenseTensor* dout,
                  DenseTensor* dx) const {
    int r = xpu_activation_backward<Context, T, XPUType>(
        dev_ctx, x, out, dout, dx, xpu::reciprocal_grad<XPUType>);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "reciprocal_grad");
  }
};

template <typename T>
struct XPUReluGradFunctor : public funcs::BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  template <typename Context>
  void operator()(const Context& dev_ctx,
                  const DenseTensor* x,
                  const DenseTensor* out,
                  const DenseTensor* dout,
                  DenseTensor* dx) const {
    int r = xpu_activation_backward<Context, T, XPUType>(
        dev_ctx, x, out, dout, dx, xpu::relu_grad<XPUType>);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "relu_grad");
  }
};

template <typename T>
struct XPURelu6GradFunctor : public funcs::BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  typename funcs::BaseActivationFunctor<T>::AttrPair GetAttrs() { return {{}}; }
  template <typename Context>
  void operator()(const Context& dev_ctx,
                  const DenseTensor* x,
                  const DenseTensor* out,
                  const DenseTensor* dout,
                  DenseTensor* dx) const {
    int r = xpu_activation_backward<Context, T, XPUType>(
        dev_ctx, x, out, dout, dx, xpu::relu6_grad<XPUType>);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "relu6_grad");
  }
};

template <typename T>
struct XPUSiluGradFunctor : public funcs::BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  template <typename Context>
  void operator()(const Context& dev_ctx,
                  const DenseTensor* x,
                  const DenseTensor* out,
                  const DenseTensor* dout,
                  DenseTensor* dx) const {
    dev_ctx.template Alloc<T>(dx);
    const XPUType* x_data = reinterpret_cast<const XPUType*>(x->data<T>());
    const XPUType* y_grad = reinterpret_cast<const XPUType*>(dout->data<T>());
    XPUType* x_grad = reinterpret_cast<XPUType*>(dx->data<T>());

    if (std::getenv("XPU_PADDLE_ACT_LUT") != nullptr) {
      if (!std::is_same<T, ::phi::dtype::bfloat16>::value) {
        // use fast_swish_grad if NOT bf16
        int r = xpu::fast_swish_grad(
            dev_ctx.x_context(), x_data, y_grad, x_grad, dx->numel());
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "fast_swish_grad");
      } else {
        // use plain swish_grad
        int r = xpu::swish_grad(
            dev_ctx.x_context(), x_data, y_grad, x_grad, dx->numel());
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "swish_grad");
      }
    } else {
      // use plain swish_grad
      int r = xpu::swish_grad(
          dev_ctx.x_context(), x_data, y_grad, x_grad, dx->numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "swish_grad");
    }
  }
};

template <typename T>
struct XPUSigmoidGradFunctor : public funcs::BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  template <typename Context>
  void operator()(const Context& dev_ctx,
                  const DenseTensor* x,
                  const DenseTensor* out,
                  const DenseTensor* dout,
                  DenseTensor* dx) const {
    dev_ctx.template Alloc<T>(dx);
    const XPUType* y_data = reinterpret_cast<const XPUType*>(out->data<T>());
    const XPUType* y_grad = reinterpret_cast<const XPUType*>(dout->data<T>());
    XPUType* x_grad = reinterpret_cast<XPUType*>(dx->data<T>());
    int r = xpu::sigmoid_grad(
        dev_ctx.x_context(), y_data, y_grad, x_grad, dx->numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "sigmoid_grad");
  }
};

template <typename T>
struct XPUSquareGradFunctor : public funcs::BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  template <typename Context>
  void operator()(const Context& dev_ctx,
                  const DenseTensor* x,
                  const DenseTensor* out,
                  const DenseTensor* dout,
                  DenseTensor* dx) const {
    int r = xpu_activation_backward<Context, T, XPUType>(
        dev_ctx, x, out, dout, dx, xpu::square_grad<XPUType>);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "square_grad");
  }
};

template <typename T>
struct XPUSqrtGradFunctor : public funcs::BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  template <typename Context>
  void operator()(const Context& dev_ctx,
                  const DenseTensor* x,
                  const DenseTensor* out,
                  const DenseTensor* dout,
                  DenseTensor* dx) const {
    int r = xpu_activation_backward<Context, T, XPUType>(
        dev_ctx, x, out, dout, dx, xpu::sqrt_grad<XPUType>);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "sqrt_grad");
  }
};

template <typename T, typename Context>
void PowGradKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& dout,
                   const Scalar& factor,
                   DenseTensor* dx) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  dev_ctx.template Alloc<T>(dx);
  const T* x_data = x.data<T>();
  const T* y_grad = dout.data<T>();
  T* x_grad = dx->data<T>();

  // check dims: all dims should equal
  auto x_dims = common::vectorize<int>(x.dims());
  auto dy_dims = common::vectorize<int>(dout.dims());
  auto dx_dims = common::vectorize<int>(dx->dims());
  PADDLE_ENFORCE_EQ(x_dims,
                    dy_dims,
                    errors::PreconditionNotMet("x_dims should match dy_dims."));
  PADDLE_ENFORCE_EQ(x_dims,
                    dx_dims,
                    errors::PreconditionNotMet("x_dims should match dx_dims."));
  float pow_factor = factor.to<float>();

  auto xpu_context = dev_ctx.x_context();
  int r = xpu::pow_grad(xpu_context,
                        reinterpret_cast<const XPUType*>(x_data),
                        reinterpret_cast<const XPUType*>(y_grad),
                        reinterpret_cast<XPUType*>(x_grad),
                        x.numel(),
                        pow_factor);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "pow_grad");
}

template <typename T>
struct XPUSwishGradFunctor : public funcs::BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  typename funcs::BaseActivationFunctor<T>::AttrPair GetAttrs() { return {{}}; }

  template <typename Context>
  void operator()(const Context& dev_ctx,
                  const DenseTensor* x,
                  const DenseTensor* out,
                  const DenseTensor* dout,
                  DenseTensor* dx) const {
    const XPUType* x_data = reinterpret_cast<const XPUType*>(x->data<T>());
    const XPUType* y_grad = reinterpret_cast<const XPUType*>(dout->data<T>());
    XPUType* x_grad = reinterpret_cast<XPUType*>(dx->data<T>());

    auto xpu_context = dev_ctx.x_context();
    int r = xpu::swish_grad(xpu_context, x_data, y_grad, x_grad, dx->numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "swish_grad");
  }
};

template <typename T>
struct XPUMishGradFunctor : public funcs::BaseActivationFunctor<T> {
  float threshold;

  typename funcs::BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  template <typename Context>
  void operator()(const Context& dev_ctx,
                  const DenseTensor* x,
                  const DenseTensor* out,
                  const DenseTensor* dout,
                  DenseTensor* dx) const {
    const T* x_data = x->data<T>();
    const T* y_grad = dout->data<T>();
    T* x_grad = dx->data<T>();

    auto xpu_context = dev_ctx.x_context();
    int r = xpu::mish_grad(
        xpu_context,
        reinterpret_cast<const float*>(x_data),
        reinterpret_cast<const float*>(x_data),  // mish_grad do not need y_data
        reinterpret_cast<const float*>(y_grad),
        reinterpret_cast<float*>(x_grad),
        dx->numel(),
        threshold);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "mish_grad");
  }
};

template <typename T>
struct XPUSoftPlusGradFunctor : public funcs::BaseActivationFunctor<T> {
  float beta;
  float threshold;
  typename funcs::BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"beta", &beta}, {"threshold", &threshold}};
  }

  template <typename Context>
  void operator()(const Context& dev_ctx,
                  const DenseTensor* x,
                  const DenseTensor* out,
                  const DenseTensor* dOut,
                  DenseTensor* dX) const {
    const T* x_data = x->data<T>();
    const T* y_grad = dOut->data<T>();
    T* x_grad = dX->data<T>();

    auto xpu_context = dev_ctx.x_context();
    int r = xpu::softplus_grad(xpu_context,
                               reinterpret_cast<const float*>(x_data),
                               reinterpret_cast<const float*>(
                                   x_data),  // softplus_grad do not need y_data
                               reinterpret_cast<const float*>(y_grad),
                               reinterpret_cast<float*>(x_grad),
                               dX->numel(),
                               beta,
                               threshold);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "softplus_grad");
  }
};

template <typename T>
struct XPUSinGradFunctor : public funcs::BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  template <typename Context>
  void operator()(const Context& dev_ctx,
                  const DenseTensor* x,
                  const DenseTensor* out,
                  const DenseTensor* dout,
                  DenseTensor* dx) const {
    int64_t len = dx->numel();
    auto dx_data = dev_ctx.template Alloc<T>(dx);
    auto dout_data = dout->data<T>();
    auto x_data = x->data<T>();

    int r = xpu::sin_grad<T>(dev_ctx.x_context(),
                             reinterpret_cast<const XPUType*>(x_data),
                             reinterpret_cast<const XPUType*>(dout_data),
                             reinterpret_cast<XPUType*>(dx_data),
                             len);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "sin_grad");
  }
};

template <typename T>
struct XPUCosGradFunctor : public funcs::BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  template <typename Context>
  void operator()(const Context& dev_ctx,
                  const DenseTensor* x,
                  const DenseTensor* out,
                  const DenseTensor* dout,
                  DenseTensor* dx) const {
    int64_t len = dx->numel();
    auto dx_data = dev_ctx.template Alloc<T>(dx);
    auto dout_data = dout->data<T>();
    auto x_data = x->data<T>();

    int r = xpu::cos_grad<T>(dev_ctx.x_context(),
                             reinterpret_cast<const XPUType*>(x_data),
                             reinterpret_cast<const XPUType*>(dout_data),
                             reinterpret_cast<XPUType*>(dx_data),
                             len);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cos_grad");
  }
};

template <typename T>
struct XPURsqrtGradFunctor : public funcs::BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  template <typename Context>
  void operator()(const Context& dev_ctx,
                  const DenseTensor* x,
                  const DenseTensor* out,
                  const DenseTensor* dout,
                  DenseTensor* dx) const {
    int r = xpu_activation_backward<Context, T, XPUType>(
        dev_ctx, x, out, dout, dx, xpu::rsqrt_grad<XPUType>);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "rsqrt_grad");
  }
};

DEFINE_XPU_ACTIVATION_GRAD_KERNEL_DEPOUT(Reciprocal, XPUReciprocalGradFunctor);
DEFINE_XPU_ACTIVATION_GRAD_KERNEL_DEPOUT(Sigmoid, XPUSigmoidGradFunctor);
DEFINE_XPU_ACTIVATION_GRAD_KERNEL_DEPOUT(Sqrt, XPUSqrtGradFunctor);
DEFINE_XPU_ACTIVATION_GRAD_KERNEL_DEPOUT(Relu, XPUReluGradFunctor);
DEFINE_XPU_ACTIVATION_GRAD_KERNEL_DEPOUT(Relu6, XPURelu6GradFunctor);
DEFINE_XPU_ACTIVATION_GRAD_KERNEL_DEPOUT(Rsqrt, XPURsqrtGradFunctor);

DEFINE_XPU_ACTIVATION_GRAD_KERNEL_DEPX(Log, XPULogGradFunctor);
DEFINE_XPU_ACTIVATION_GRAD_KERNEL_DEPX(Square, XPUSquareGradFunctor);
DEFINE_XPU_ACTIVATION_GRAD_KERNEL_DEPX(Swish, XPUSwishGradFunctor);
DEFINE_XPU_ACTIVATION_GRAD_KERNEL_DEPX(Sin, XPUSinGradFunctor);
DEFINE_XPU_ACTIVATION_GRAD_KERNEL_DEPX(Cos, XPUCosGradFunctor);

DEFINE_XPU_ACT_GRAD_KERNEL_WITH_ONE_ATTRS_DEPX(Mish,
                                               XPUMishGradFunctor,
                                               threshold);
DEFINE_XPU_ACT_GRAD_KERNEL_WITH_ONE_ATTRS_DEPX(LeakyRelu,
                                               XPULeakyReluGradFunctor,
                                               alpha);

DEFINE_XPU_ACT_GRAD_KERNEL_WITH_TWO_ATTRS_DEPX(Softplus,
                                               XPUSoftPlusGradFunctor,
                                               beta,
                                               threshold)
DEFINE_XPU_ACT_GRAD_KERNEL_WITH_TWO_ATTRS_DEPOUT(HardSigmoid,
                                                 XPUHardSigmoidGradFunctor,
                                                 slope,
                                                 offset)

template <typename T, typename Context>
void HardSwishGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& dout,
                         DenseTensor* dx) {
  XPUHardSwishGradFunctor<T> functor;
  float threshold = 6;
  float scale = 6;
  float offset = 3;
  auto attrs = functor.GetAttrs();
  *(attrs[0].second) = threshold;
  *(attrs[1].second) = scale;
  *(attrs[2].second) = offset;
  ActivationGradXPUImpl<T, Context, XPUHardSwishGradFunctor<T>>(
      dev_ctx, &x, nullptr, &dout, dx, functor);
}

template <typename T, typename Context>
void SiluGradKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& out,
                    const DenseTensor& dout,
                    DenseTensor* dx) {
  XPUSiluGradFunctor<T> functor;
  ActivationGradXPUImpl<T, Context, XPUSiluGradFunctor<T>>(
      dev_ctx, &x, &out, &dout, dx, functor);
}

template <typename T, typename Context>
void ExpGradKernel(const Context& dev_ctx,
                   const DenseTensor& out,
                   const DenseTensor& dout,
                   DenseTensor* dx) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  dev_ctx.template Alloc<T>(dx);
  const XPUType* y_data = reinterpret_cast<const XPUType*>(out.data<T>());
  const XPUType* y_grad = reinterpret_cast<const XPUType*>(dout.data<T>());
  XPUType* x_grad = reinterpret_cast<XPUType*>(dx->data<T>());

  int r =
      xpu::exp_grad(dev_ctx.x_context(), y_data, y_grad, x_grad, dx->numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "exp_grad");
}

template <typename T, typename Context>
void TanhGradKernel(const Context& dev_ctx,
                    const DenseTensor& out,
                    const DenseTensor& dout,
                    DenseTensor* dx) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  dev_ctx.template Alloc<T>(dx);
  const XPUType* y_data = reinterpret_cast<const XPUType*>(out.data<T>());
  const XPUType* y_grad = reinterpret_cast<const XPUType*>(dout.data<T>());
  XPUType* x_grad = reinterpret_cast<XPUType*>(dx->data<T>());

  int r =
      xpu::tanh_grad(dev_ctx.x_context(), y_data, y_grad, x_grad, dx->numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "tanh_grad");
}
}  // namespace phi

PD_REGISTER_KERNEL(relu_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::ReluGradKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(silu_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::SiluGradKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

#define PD_REGISTER_ACTIVATION_GRAD_KERNEL(name, func) \
  PD_REGISTER_KERNEL(name, XPU, ALL_LAYOUT, phi::func, float) {}

PD_REGISTER_KERNEL(tanh_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::TanhGradKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(exp_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::ExpGradKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(square_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::SquareGradKernel,
                   float,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(swish_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::SwishGradKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(hardswish_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::HardSwishGradKernel,
                   float,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(sigmoid_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::SigmoidGradKernel,
                   float,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(pow_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::PowGradKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(rsqrt_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::RsqrtGradKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_ACTIVATION_GRAD_KERNEL(log_grad, LogGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(leaky_relu_grad, LeakyReluGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(hardsigmoid_grad, HardSigmoidGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(reciprocal_grad, ReciprocalGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(relu6_grad, Relu6GradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(sqrt_grad, SqrtGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(mish_grad, MishGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(softplus_grad, SoftplusGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(sin_grad, SinGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(cos_grad, CosGradKernel)
