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

#include "paddle/phi/kernels/activation_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/activation_functor.h"

namespace phi {

template <typename T, typename Context, typename Functor>
void ActivationXPUImpl(const Context& dev_ctx,
                       const DenseTensor& x,
                       DenseTensor* out,
                       const Functor& functor) {
  PADDLE_ENFORCE_NOT_NULL(out,
                          errors::NotFound("Output Out should not be nullptr"));
  dev_ctx.template Alloc<T>(out);
  functor(dev_ctx, x, out);
}

#define DEFINE_XPU_ACTIVATION_KERNEL(name, functor_class)                      \
  template <typename T, typename Context>                                      \
  void name##Kernel(                                                           \
      const Context& dev_ctx, const DenseTensor& x, DenseTensor* out) {        \
    functor_class<T> functor;                                                  \
    ActivationXPUImpl<T, Context, functor_class<T>>(dev_ctx, x, out, functor); \
  }

#define DEFINE_XPU_ACTIVATION_KERNEL_WITH_ONE_ATTRS(name, functor_class, attr) \
  template <typename T, typename Context>                                      \
  void name##Kernel(const Context& dev_ctx,                                    \
                    const DenseTensor& x,                                      \
                    float attr,                                                \
                    DenseTensor* out) {                                        \
    functor_class<T> functor;                                                  \
    auto attrs = functor.GetAttrs();                                           \
    *(attrs[0].second) = attr;                                                 \
    ActivationXPUImpl<T, Context, functor_class<T>>(dev_ctx, x, out, functor); \
  }

#define DEFINE_XPU_ACTIVATION_KERNEL_WITH_TWO_ATTRS(                           \
    name, functor_class, attr1, attr2)                                         \
  template <typename T, typename Context>                                      \
  void name##Kernel(const Context& dev_ctx,                                    \
                    const DenseTensor& x,                                      \
                    float attr1,                                               \
                    float attr2,                                               \
                    DenseTensor* out) {                                        \
    functor_class<T> functor;                                                  \
    auto attrs = functor.GetAttrs();                                           \
    *(attrs[0].second) = attr1;                                                \
    *(attrs[1].second) = attr2;                                                \
    ActivationXPUImpl<T, Context, functor_class<T>>(dev_ctx, x, out, functor); \
  }

template <typename Context, typename T, typename XPUType>
int xpu_activation_func(
    const Context& dev_ctx,
    const DenseTensor& x,
    DenseTensor* out,
    std::function<int(xpu::Context*, const XPUType*, XPUType*, int)> func) {
  int r = func(dev_ctx.x_context(),
               reinterpret_cast<const XPUType*>(x.data<T>()),
               reinterpret_cast<XPUType*>(out->data<T>()),
               x.numel());
  return r;
}

template <typename Context, typename T, typename XPUType>
int xpu_activation_func_with_max_x_y(
    const Context& dev_ctx,
    const DenseTensor& x,
    DenseTensor* out,
    std::function<
        int(xpu::Context*, const XPUType*, XPUType*, int, const float*, float*)>
        func) {
  // does not support "const float* max_x, float* max_y" now
  int r = func(dev_ctx.x_context(),
               reinterpret_cast<const XPUType*>(x.data<T>()),
               reinterpret_cast<XPUType*>(out->data<T>()),
               x.numel(),
               nullptr,
               nullptr);
  return r;
}

template <typename Context, typename T, typename XPUType>
int xpu_activation_1attr_func(const Context& dev_ctx,
                              const DenseTensor& x,
                              DenseTensor* out,
                              float attr,
                              std::function<int(xpu::Context*,
                                                const XPUType*,
                                                XPUType*,
                                                int,
                                                float,
                                                const float*,
                                                float*)> func) {
  // does not support "const float* max_x, float* max_y" now
  int r = func(dev_ctx.x_context(),
               reinterpret_cast<const XPUType*>(x.data<T>()),
               reinterpret_cast<XPUType*>(out->data<T>()),
               x.numel(),
               attr,
               nullptr,
               nullptr);
  return r;
}

template <typename Context, typename T, typename XPUType>
int xpu_activation_2attr_func(const Context& dev_ctx,
                              const DenseTensor& x,
                              DenseTensor* out,
                              float attr1,
                              float attr2,
                              std::function<int(xpu::Context*,
                                                const XPUType*,
                                                XPUType*,
                                                int,
                                                float,
                                                float,
                                                const float*,
                                                float*)> func) {
  // does not support "const float* max_x, float* max_y" now
  int r = func(dev_ctx.x_context(),
               reinterpret_cast<const XPUType*>(x.data<T>()),
               reinterpret_cast<XPUType*>(out->data<T>()),
               x.numel(),
               attr1,
               attr2,
               nullptr,
               nullptr);
  return r;
}

template <typename T>
struct XPUExpFunctor : public funcs::BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  template <typename Context>
  void operator()(const Context& dev_ctx,
                  const DenseTensor& x,
                  DenseTensor* out) const {
    int r = xpu_activation_func<Context, T, XPUType>(
        dev_ctx, x, out, xpu::exp<XPUType>);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "exp");
  }
};

template <typename T>
struct XPULogFunctor : public funcs::BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  template <typename Context>
  void operator()(const Context& dev_ctx,
                  const DenseTensor& x,
                  DenseTensor* out) const {
    int r = xpu_activation_func<Context, T, XPUType>(
        dev_ctx, x, out, xpu::log<XPUType>);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "log");
  }
};

template <typename T>
struct XPULeakyReluFunctor : public funcs::BaseActivationFunctor<T> {
  float alpha;
  typename funcs::BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"alpha", &alpha}};
  }
  template <typename Context>
  void operator()(const Context& dev_ctx,
                  const DenseTensor& x,
                  DenseTensor* out) const {
    using XPUType = typename XPUTypeTrait<T>::Type;
    int r = xpu_activation_1attr_func<Context, T, XPUType>(
        dev_ctx, x, out, alpha, xpu::leaky_relu<XPUType>);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "leaky_relu");
  }
};

template <typename T>
struct XPURoundFunctor : public funcs::BaseActivationFunctor<T> {
  int decimals;
  std::vector<std::pair<const char*, int*>> GetAttrs() {
    return {{"decimals", &decimals}};
  }

  template <typename Context>
  void operator()(const Context& dev_ctx,
                  const DenseTensor& x,
                  DenseTensor* out) const {
    using XPUType = typename XPUTypeTrait<T>::Type;
    int r = xpu::round<XPUType>(dev_ctx.x_context(),
                                reinterpret_cast<const XPUType*>(x.data<T>()),
                                reinterpret_cast<XPUType*>(out->data<T>()),
                                x.numel(),
                                decimals);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "round");
  }
};

template <typename T, typename Context>
void PowKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const Scalar& factor,
               DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  dev_ctx.template Alloc<T>(out);

  const XPUType* x_data = reinterpret_cast<const XPUType*>(x.data<T>());
  XPUType* y_data = reinterpret_cast<XPUType*>(out->data<T>());
  XPUType pow_factor = static_cast<XPUType>(factor.to<T>());

  auto xpu_context = dev_ctx.x_context();

  int r = xpu::pow_tensor_scalar(
      xpu_context, x_data, pow_factor, y_data, x.numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "pow_tensor_scalar");
}

template <typename T>
struct XPUHardSigmoidFunctor : public funcs::BaseActivationFunctor<T> {
  float slope;
  float offset;
  typename funcs::BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"slope", &slope}, {"offset", &offset}};
  }

  template <typename Context>
  void operator()(const Context& dev_ctx,
                  const DenseTensor& x,
                  DenseTensor* out) const {
    using XPUType = typename XPUTypeTrait<T>::Type;
    int r = xpu_activation_1attr_func<Context, T, XPUType>(
        dev_ctx, x, out, slope, xpu::hard_sigmoid<XPUType>);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "hardsigmoid");
  }
};

template <typename T>
struct XPUHardSwishFunctor : public funcs::BaseActivationFunctor<T> {
  float threshold;
  float scale;
  float offset;

  typename funcs::BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}, {"scale", &scale}, {"offset", &offset}};
  }

  template <typename Context>
  void operator()(const Context& dev_ctx,
                  const DenseTensor& x,
                  DenseTensor* out) const {
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
    int r = xpu_activation_func_with_max_x_y<Context, T, XPUType>(
        dev_ctx, x, out, xpu::hard_swish<XPUType>);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "hard_swish");
  }
};

template <typename T>
struct XPUReciprocalFunctor : public funcs::BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  template <typename Context>
  void operator()(const Context& dev_ctx,
                  const DenseTensor& x,
                  DenseTensor* out) const {
    int r = xpu_activation_func<Context, T, XPUType>(
        dev_ctx, x, out, xpu::reciprocal<XPUType>);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "reciprocal");
  }
};

template <typename T>
struct XPUReluFunctor : public funcs::BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  template <typename Context>
  void operator()(const Context& dev_ctx,
                  const DenseTensor& x,
                  DenseTensor* out) const {
    const XPUType* x_data = reinterpret_cast<const XPUType*>(x.data<T>());
    XPUType* y_data = reinterpret_cast<XPUType*>(out->data<T>());

    auto xpu_context = dev_ctx.x_context();
    int r = xpu::relu(xpu_context, x_data, y_data, x.numel(), nullptr, nullptr);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "relu");
  }
};

template <typename T>
struct XPURelu6Functor : public funcs::BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  float threshold;
  typename funcs::BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }
  template <typename Context>
  void operator()(const Context& dev_ctx,
                  const DenseTensor& x,
                  DenseTensor* out) const {
    int r = xpu_activation_func_with_max_x_y<Context, T, XPUType>(
        dev_ctx, x, out, xpu::relu6<XPUType>);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "relu6");
  }
};

template <typename T>
struct XPUSiluFunctor : public funcs::BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  template <typename Context>
  void operator()(const Context& dev_ctx,
                  const DenseTensor& x,
                  DenseTensor* out) const {
    dev_ctx.template Alloc<T>(out);
    const XPUType* x_data = reinterpret_cast<const XPUType*>(x.data<T>());
    XPUType* y_data = reinterpret_cast<XPUType*>(out->data<T>());

    auto xpu_context = dev_ctx.x_context();
    if (std::getenv("XPU_PADDLE_ACT_LUT") != nullptr) {
      if (!std::is_same<T, ::phi::dtype::bfloat16>::value) {
        // use fast_swish if NOT bf16
        int r = xpu::fast_swish(
            xpu_context, x_data, y_data, x.numel(), nullptr, nullptr);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "fast_swish");
      } else {
        // use plain swish
        int r = xpu::swish(
            xpu_context, x_data, y_data, x.numel(), nullptr, nullptr);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "swish");
      }
    } else {
      // use plain swish
      int r =
          xpu::swish(xpu_context, x_data, y_data, x.numel(), nullptr, nullptr);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "swish");
    }
  }
};

template <typename T>
struct XPUSigmoidFunctor : public funcs::BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  template <typename Context>
  void operator()(const Context& dev_ctx,
                  const DenseTensor& x,
                  DenseTensor* out) const {
    int r = xpu_activation_func_with_max_x_y<Context, T, XPUType>(
        dev_ctx, x, out, xpu::sigmoid<XPUType>);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "sigmoid");
  }
};

template <typename T>
struct XPUSquareFunctor : public funcs::BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  template <typename Context>
  void operator()(const Context& dev_ctx,
                  const DenseTensor& x,
                  DenseTensor* out) const {
    int r = xpu_activation_func<Context, T, XPUType>(
        dev_ctx, x, out, xpu::square<XPUType>);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "square");
  }
};

template <typename T>
struct XPUSqrtFunctor : public funcs::BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  template <typename Context>
  void operator()(const Context& dev_ctx,
                  const DenseTensor& x,
                  DenseTensor* out) const {
    int r = xpu_activation_func<Context, T, XPUType>(
        dev_ctx, x, out, xpu::sqrt<XPUType>);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "sqrt");
  }
};

template <typename T>
struct XPUMishFunctor : public funcs::BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  float threshold;
  typename funcs::BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  template <typename Context>
  void operator()(const Context& dev_ctx,
                  const DenseTensor& x,
                  DenseTensor* out) const {
    int r = xpu_activation_1attr_func<Context, T, XPUType>(
        dev_ctx, x, out, threshold, xpu::mish<XPUType>);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "mish");
  }
};

template <typename T, typename Context>
void SwishKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  dev_ctx.template Alloc<T>(out);
  int r = xpu::swish(dev_ctx.x_context(),
                     reinterpret_cast<const XPUType*>(x.data<T>()),
                     reinterpret_cast<XPUType*>(out->data<T>()),
                     x.numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "swish");
}

template <typename T, typename Context>
void EluKernel(const Context& dev_ctx,
               const DenseTensor& x,
               float alpha,
               DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  dev_ctx.template Alloc<T>(out);
  // template<typename T> int elu(Context* ctx, const T* x, T* y, int64_t len,
  // float alpha = 1.0f, const float* max_x = nullptr, float* max_y = nullptr)
  int r = xpu::elu(dev_ctx.x_context(),
                   reinterpret_cast<const XPUType*>(x.data<T>()),
                   reinterpret_cast<XPUType*>(out->data<T>()),
                   x.numel(),
                   alpha);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "elu");
}

template <typename T, typename Context>
void Relu6Kernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 DenseTensor* out) {
  XPURelu6Functor<T> functor;
  auto attrs = functor.GetAttrs();
  *(attrs[0].second) = 6.0;
  ActivationXPUImpl<T, Context, XPURelu6Functor<T>>(dev_ctx, x, out, functor);
}

template <typename T>
struct XPUSoftplusFunctor : public funcs::BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  float beta;
  float threshold;

  typename funcs::BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"beta", &beta}, {"threshold", &threshold}};
  }

  template <typename Context>
  void operator()(const Context& dev_ctx,
                  const DenseTensor& x,
                  DenseTensor* out) const {
    int r = xpu_activation_2attr_func<Context, T, XPUType>(
        dev_ctx, x, out, beta, threshold, xpu::softplus<XPUType>);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "softplus");
  }
};

template <typename T>
struct XPUTanhFunctor : public funcs::BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  template <typename Context>
  void operator()(const Context& dev_ctx,
                  const DenseTensor& x,
                  DenseTensor* out) const {
    int r = xpu_activation_func_with_max_x_y<Context, T, XPUType>(
        dev_ctx, x, out, xpu::tanh<XPUType>);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "tanh");
  }
};

template <typename T>
struct XPUFloorFunctor : public funcs::BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  template <typename Context>
  void operator()(const Context& dev_ctx,
                  const DenseTensor& x,
                  DenseTensor* out) const {
    int r = xpu_activation_func<Context, T, XPUType>(
        dev_ctx, x, out, xpu::floor<XPUType>);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "floor");
  }
};

template <typename T>
struct XPUSinFunctor : public funcs::BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  template <typename Context>
  void operator()(const Context& dev_ctx,
                  const DenseTensor& x,
                  DenseTensor* out) const {
    int r = xpu_activation_func<Context, T, XPUType>(
        dev_ctx, x, out, xpu::sin<XPUType>);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "sin");
  }
};

template <typename T>
struct XPUCosFunctor : public funcs::BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  template <typename Context>
  void operator()(const Context& dev_ctx,
                  const DenseTensor& x,
                  DenseTensor* out) const {
    int r = xpu_activation_func<Context, T, XPUType>(
        dev_ctx, x, out, xpu::cos<XPUType>);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cos");
  }
};

template <typename T>
struct XPURsqrtFunctor : public funcs::BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  template <typename Context>
  void operator()(const Context& dev_ctx,
                  const DenseTensor& x,
                  DenseTensor* out) const {
    int ret = xpu_activation_func<Context, T, XPUType>(
        dev_ctx, x, out, xpu::rsqrt<XPUType>);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "rsqrt");
  }
};

template <typename T>
struct XPUTanFunctor : public funcs::BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  template <typename Context>
  void operator()(const Context& dev_ctx,
                  const DenseTensor& x,
                  DenseTensor* out) const {
    int ret = xpu_activation_func<Context, T, XPUType>(
        dev_ctx, x, out, xpu::tan<XPUType>);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "tan");
  }
};

template <typename T>
struct XPUAcosFunctor : public funcs::BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  template <typename Context>
  void operator()(const Context& dev_ctx,
                  const DenseTensor& x,
                  DenseTensor* out) const {
    int ret = xpu_activation_func<Context, T, XPUType>(
        dev_ctx, x, out, xpu::arccos<XPUType>);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "arccos");
  }
};

DEFINE_XPU_ACTIVATION_KERNEL(Exp, XPUExpFunctor)
DEFINE_XPU_ACTIVATION_KERNEL(Floor, XPUFloorFunctor)
DEFINE_XPU_ACTIVATION_KERNEL(Log, XPULogFunctor)
DEFINE_XPU_ACTIVATION_KERNEL(Reciprocal, XPUReciprocalFunctor)
DEFINE_XPU_ACTIVATION_KERNEL(Relu, XPUReluFunctor)
DEFINE_XPU_ACTIVATION_KERNEL(Sigmoid, XPUSigmoidFunctor)
DEFINE_XPU_ACTIVATION_KERNEL(Square, XPUSquareFunctor)
DEFINE_XPU_ACTIVATION_KERNEL(Sqrt, XPUSqrtFunctor)
DEFINE_XPU_ACTIVATION_KERNEL(Tanh, XPUTanhFunctor)
DEFINE_XPU_ACTIVATION_KERNEL(Silu, XPUSiluFunctor)
DEFINE_XPU_ACTIVATION_KERNEL(Sin, XPUSinFunctor)
DEFINE_XPU_ACTIVATION_KERNEL(Cos, XPUCosFunctor)
DEFINE_XPU_ACTIVATION_KERNEL(Rsqrt, XPURsqrtFunctor)
DEFINE_XPU_ACTIVATION_KERNEL(Tan, XPUTanFunctor)
DEFINE_XPU_ACTIVATION_KERNEL(Acos, XPUAcosFunctor)

DEFINE_XPU_ACTIVATION_KERNEL_WITH_ONE_ATTRS(Mish, XPUMishFunctor, threshold)
DEFINE_XPU_ACTIVATION_KERNEL_WITH_ONE_ATTRS(LeakyRelu,
                                            XPULeakyReluFunctor,
                                            alpha)
DEFINE_XPU_ACTIVATION_KERNEL_WITH_TWO_ATTRS(Softplus,
                                            XPUSoftplusFunctor,
                                            beta,
                                            threshold)
DEFINE_XPU_ACTIVATION_KERNEL_WITH_TWO_ATTRS(HardSigmoid,
                                            XPUHardSigmoidFunctor,
                                            slope,
                                            offset)

template <typename T, typename Context>
void HardSwishKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     DenseTensor* out) {
  XPUHardSwishFunctor<T> functor;
  float threshold = 6;
  float scale = 6;
  float offset = 3;
  auto attrs = functor.GetAttrs();
  *(attrs[0].second) = threshold;
  *(attrs[1].second) = scale;
  *(attrs[2].second) = offset;
  ActivationXPUImpl<T, Context, XPUHardSwishFunctor<T>>(
      dev_ctx, x, out, functor);
}

template <typename T, typename Context>
void RoundKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const int decimals,
                 DenseTensor* out) {
  XPURoundFunctor<T> functor;
  auto attrs = functor.GetAttrs();
  *(attrs[0].second) = decimals;
  ActivationXPUImpl<T, Context, XPURoundFunctor<T>>(dev_ctx, x, out, functor);
}

}  // namespace phi

PD_REGISTER_KERNEL(relu,
                   XPU,
                   ALL_LAYOUT,
                   phi::ReluKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(silu,
                   XPU,
                   ALL_LAYOUT,
                   phi::SiluKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(
    elu, XPU, ALL_LAYOUT, phi::EluKernel, float, phi::dtype::float16) {}
PD_REGISTER_KERNEL(
    sigmoid, XPU, ALL_LAYOUT, phi::SigmoidKernel, float, phi::dtype::float16) {}
PD_REGISTER_KERNEL(swish,
                   XPU,
                   ALL_LAYOUT,
                   phi::SwishKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(hardsigmoid,
                   XPU,
                   ALL_LAYOUT,
                   phi::HardSigmoidKernel,
                   float,
                   phi::dtype::float16) {}
PD_REGISTER_KERNEL(hardswish,
                   XPU,
                   ALL_LAYOUT,
                   phi::HardSwishKernel,
                   float,
                   phi::dtype::float16) {}
PD_REGISTER_KERNEL(leaky_relu,
                   XPU,
                   ALL_LAYOUT,
                   phi::LeakyReluKernel,
                   float,
                   phi::dtype::float16) {}
PD_REGISTER_KERNEL(sqrt,
                   XPU,
                   ALL_LAYOUT,
                   phi::SqrtKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(
    tanh, XPU, ALL_LAYOUT, phi::TanhKernel, float, phi::dtype::float16) {}

PD_REGISTER_KERNEL(square,
                   XPU,
                   ALL_LAYOUT,
                   phi::SquareKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(log,
                   XPU,
                   ALL_LAYOUT,
                   phi::LogKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(
    relu6, XPU, ALL_LAYOUT, phi::Relu6Kernel, float, phi::dtype::float16) {}

PD_REGISTER_KERNEL(sin,
                   XPU,
                   ALL_LAYOUT,
                   phi::SinKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(cos,
                   XPU,
                   ALL_LAYOUT,
                   phi::CosKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(pow,
                   XPU,
                   ALL_LAYOUT,
                   phi::PowKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(rsqrt,
                   XPU,
                   ALL_LAYOUT,
                   phi::RsqrtKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(exp,
                   XPU,
                   ALL_LAYOUT,
                   phi::ExpKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(
    round, XPU, ALL_LAYOUT, phi::RoundKernel, float, phi::dtype::float16) {}

PD_REGISTER_KERNEL(
    tan, XPU, ALL_LAYOUT, phi::TanKernel, float, phi::dtype::float16) {}

PD_REGISTER_KERNEL(acos,
                   XPU,
                   ALL_LAYOUT,
                   phi::AcosKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

#define PD_REGISTER_ACTIVATION_KERNEL(name, func) \
  PD_REGISTER_KERNEL(name, XPU, ALL_LAYOUT, phi::func, float) {}

PD_REGISTER_ACTIVATION_KERNEL(floor, FloorKernel)
PD_REGISTER_ACTIVATION_KERNEL(mish, MishKernel)
PD_REGISTER_ACTIVATION_KERNEL(reciprocal, ReciprocalKernel)
PD_REGISTER_ACTIVATION_KERNEL(softplus, SoftplusKernel)
