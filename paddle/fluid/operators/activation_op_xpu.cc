/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_XPU

#include <string>

#include "paddle/fluid/operators/activation_op.h"
#include "paddle/fluid/platform/device/device_wrapper.h"
#include "paddle/fluid/platform/device/xpu/xpu_header.h"

namespace paddle {
namespace operators {

using paddle::framework::Tensor;

template <typename Functor>
class XPUActivationKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    Functor functor;

    auto attrs = functor.GetAttrs();
    for (auto &attr : attrs) {
      *attr.second = context.Attr<float>(attr.first);
    }
    functor(context);
  }
};

template <typename Functor>
class XPUActivationGradKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    Functor functor;

    auto attrs = functor.GetAttrs();
    for (auto &attr : attrs) {
      *attr.second = context.Attr<float>(attr.first);
    }
    functor(context);
  }
};

template <typename DeviceContext, typename T, typename XPUT>
void xpu_activation_forward(
    const framework::ExecutionContext &ctx,
    std::function<int(xpu::Context *, const XPUT *, XPUT *, int)> func) {
  const auto *x = ctx.Input<Tensor>("X");
  auto *y = ctx.Output<Tensor>("Out");
  const XPUT *x_data = reinterpret_cast<const XPUT *>(x->data<T>());
  XPUT *y_data = reinterpret_cast<XPUT *>(y->mutable_data<T>(ctx.GetPlace()));

  auto xpu_context = ctx.device_context<DeviceContext>().x_context();
  int r = func(xpu_context, x_data, y_data, x->numel());
  PADDLE_ENFORCE_EQ(
      r, xpu::Error_t::SUCCESS,
      platform::errors::External("XPU activation op return wrong value[%d %s].",
                                 r, XPUAPIErrorMsg[r]));
}

template <typename DeviceContext, typename T, typename XPUT>
void xpu_activation_backward(
    const framework::ExecutionContext &ctx,
    std::function<int(xpu::Context *, const XPUT *, const XPUT *, const XPUT *,
                      XPUT *, int)>
        func) {
  /* TODO: relu tanh sigmoid are inplace */
  const auto *x = ctx.Input<Tensor>("X");
  auto *y = ctx.Input<Tensor>("Out");
  auto *dOut = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
  auto *dX = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
  const XPUT *x_data = nullptr;
  const XPUT *y_data = nullptr;
  const XPUT *y_grad = nullptr;
  if (x != nullptr) x_data = reinterpret_cast<const XPUT *>(x->data<T>());
  if (y != nullptr) y_data = reinterpret_cast<const XPUT *>(y->data<T>());
  if (dOut != nullptr) y_grad = reinterpret_cast<const XPUT *>(dOut->data<T>());
  XPUT *x_grad = reinterpret_cast<XPUT *>(dX->mutable_data<T>(ctx.GetPlace()));
  auto xpu_context = ctx.device_context<DeviceContext>().x_context();

  int r = func(xpu_context, x_data, y_data, y_grad, x_grad, dX->numel());
  PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                    platform::errors::External(
                        "XPU activation grad op return wrong value[%d %s].", r,
                        XPUAPIErrorMsg[r]));
}

template <typename T>
struct XPUAbsFunctor : public BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  void operator()(const framework::ExecutionContext &ctx) const {
    xpu_activation_forward<paddle::platform::XPUDeviceContext, T, XPUType>(
        ctx, xpu::abs<XPUType>);
  }
};

template <typename T>
struct XPUAbsGradFunctor : public BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  void operator()(const framework::ExecutionContext &ctx) const {
    xpu_activation_backward<paddle::platform::XPUDeviceContext, T, XPUType>(
        ctx, xpu::abs_grad<XPUType>);
  }
};

template <typename T>
struct XPUExpFunctor : public BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  void operator()(const framework::ExecutionContext &ctx) const {
    xpu_activation_forward<paddle::platform::XPUDeviceContext, T, XPUType>(
        ctx, xpu::exp<XPUType>);
  }
};

template <typename T>
struct XPULogFunctor : public BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  void operator()(const framework::ExecutionContext &ctx) const {
    xpu_activation_forward<paddle::platform::XPUDeviceContext, T, XPUType>(
        ctx, xpu::log<XPUType>);
  }
};

template <typename T>
struct XPUReciprocalFunctor : public BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  void operator()(const framework::ExecutionContext &ctx) const {
    xpu_activation_forward<paddle::platform::XPUDeviceContext, T, XPUType>(
        ctx, xpu::reciprocal<XPUType>);
  }
};

template <typename T>
struct XPUReciprocalGradFunctor : public BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  void operator()(const framework::ExecutionContext &ctx) const {
    xpu_activation_backward<paddle::platform::XPUDeviceContext, T, XPUType>(
        ctx, xpu::reciprocal_grad<XPUType>);
  }
};

template <typename T>
struct XPUReluFunctor : public BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  void operator()(const framework::ExecutionContext &ctx) const {
    xpu_activation_forward<paddle::platform::XPUDeviceContext, T, XPUType>(
        ctx, xpu::relu<XPUType>);
  }
};

template <typename T>
struct XPUReluGradFunctor : public BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  void operator()(const framework::ExecutionContext &ctx) const {
    xpu_activation_backward<paddle::platform::XPUDeviceContext, T, XPUType>(
        ctx, xpu::relu_grad<XPUType>);
  }
};

template <typename T>
struct XPUSigmoidFunctor : public BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  void operator()(const framework::ExecutionContext &ctx) const {
    xpu_activation_forward<paddle::platform::XPUDeviceContext, T, XPUType>(
        ctx, xpu::sigmoid<XPUType>);
  }
};

template <typename T>
struct XPUSigmoidGradFunctor : public BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  void operator()(const framework::ExecutionContext &ctx) const {
    xpu_activation_backward<paddle::platform::XPUDeviceContext, T, XPUType>(
        ctx, xpu::sigmoid_grad<XPUType>);
  }
};

template <typename T>
struct XPUSqrtFunctor : public BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  void operator()(const framework::ExecutionContext &ctx) const {
    xpu_activation_forward<paddle::platform::XPUDeviceContext, T, XPUType>(
        ctx, xpu::sqrt<XPUType>);
  }
};

template <typename T>
struct XPUSqrtGradFunctor : public BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  void operator()(const framework::ExecutionContext &ctx) const {
    xpu_activation_backward<paddle::platform::XPUDeviceContext, T, XPUType>(
        ctx, xpu::sqrt_grad<XPUType>);
  }
};

template <typename T>
struct XPUSquareFunctor : public BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  void operator()(const framework::ExecutionContext &ctx) const {
    xpu_activation_forward<paddle::platform::XPUDeviceContext, T, XPUType>(
        ctx, xpu::square<XPUType>);
  }
};

template <typename T>
struct XPUSquareGradFunctor : public BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  void operator()(const framework::ExecutionContext &ctx) const {
    xpu_activation_backward<paddle::platform::XPUDeviceContext, T, XPUType>(
        ctx, xpu::square_grad<XPUType>);
  }
};

template <typename T>
struct XPUTanhFunctor : public BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  void operator()(const framework::ExecutionContext &ctx) const {
    xpu_activation_forward<paddle::platform::XPUDeviceContext, T, XPUType>(
        ctx, xpu::tanh<XPUType>);
  }
};

template <typename T>
struct XPUTanhGradFunctor : public BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  void operator()(const framework::ExecutionContext &ctx) const {
    xpu_activation_backward<paddle::platform::XPUDeviceContext, T, XPUType>(
        ctx, xpu::tanh_grad<XPUType>);
  }
};

template <typename T>
struct XPUHardSwishFunctor : public BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  void operator()(const framework::ExecutionContext &ctx) const {
    float threshold = ctx.Attr<float>("threshold");
    float scale = ctx.Attr<float>("scale");
    float offset = ctx.Attr<float>("offset");
    PADDLE_ENFORCE_EQ(threshold, 6.0f,
                      platform::errors::External(
                          "Not support threshold [%f] in XPU", threshold));
    PADDLE_ENFORCE_EQ(scale, 6.0f, platform::errors::External(
                                       "Not support scale [%f] in XPU", scale));
    PADDLE_ENFORCE_EQ(
        offset, 3.0f,
        platform::errors::External("Not support offset [%f] in XPU", offset));
    xpu_activation_forward<paddle::platform::XPUDeviceContext, T, XPUType>(
        ctx, xpu::hard_swish<XPUType>);
  }
};

template <typename T>
struct XPUHardSwishGradFunctor : public BaseActivationFunctor<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  void operator()(const framework::ExecutionContext &ctx) const {
    float threshold = ctx.Attr<float>("threshold");
    float scale = ctx.Attr<float>("scale");
    float offset = ctx.Attr<float>("offset");
    PADDLE_ENFORCE_EQ(threshold, 6.0f,
                      platform::errors::External(
                          "Not support threshold [%f] in XPU", threshold));
    PADDLE_ENFORCE_EQ(scale, 6.0f, platform::errors::External(
                                       "Not support scale [%f] in XPU", scale));
    PADDLE_ENFORCE_EQ(
        offset, 3.0f,
        platform::errors::External("Not support offset [%f] in XPU", offset));
    xpu_activation_backward<paddle::platform::XPUDeviceContext, T, XPUType>(
        ctx, xpu::hard_swish_grad<XPUType>);
  }
};

template <typename T>
struct XPULeakyReluFunctor : public BaseActivationFunctor<T> {
  void operator()(const framework::ExecutionContext &ctx) const {
    const auto *x = ctx.Input<Tensor>("X");
    auto *y = ctx.Output<Tensor>("Out");
    float alpha = ctx.Attr<float>("alpha");
    const T *x_data = x->data<T>();
    T *y_data = y->mutable_data<T>(ctx.GetPlace());

    auto xpu_context =
        ctx.device_context<paddle::platform::XPUDeviceContext>().x_context();
    int r = xpu::leaky_relu(xpu_context, x_data, y_data, x->numel(), alpha);
    PADDLE_ENFORCE_EQ(
        r, xpu::Error_t::SUCCESS,
        platform::errors::External("XPU leaky_relu return wrong value[%d %s].",
                                   r, XPUAPIErrorMsg[r]));
  }
};

template <typename T>
struct XPULeakyReluGradFunctor : public BaseActivationFunctor<T> {
  void operator()(const framework::ExecutionContext &ctx) const {
    const auto *x = ctx.Input<Tensor>("X");
    auto *dOut = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto *dX = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    float alpha = ctx.Attr<float>("alpha");
    const T *x_data = nullptr;
    const T *y_grad = nullptr;
    if (x != nullptr) x_data = x->data<T>();
    if (dOut != nullptr) y_grad = dOut->data<T>();
    T *x_grad = dX->mutable_data<T>(ctx.GetPlace());
    auto xpu_context =
        ctx.device_context<paddle::platform::XPUDeviceContext>().x_context();

    // The signs of x and y are the same,
    // y == nullptr here,
    // so we give 2 x to the api
    int r = xpu::leaky_relu_grad(
        xpu_context, reinterpret_cast<const float *>(x_data),
        reinterpret_cast<const float *>(x_data),
        reinterpret_cast<const float *>(y_grad),
        reinterpret_cast<float *>(x_grad), dX->numel(), alpha);
    PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                      platform::errors::External(
                          "XPU leaky_relu_grad return wrong value[%d %s].", r,
                          XPUAPIErrorMsg[r]));
  }
};

template <typename T>
struct XPUPowFunctor : public BaseActivationFunctor<T> {
  void operator()(const framework::ExecutionContext &ctx) const {
    const auto *x = ctx.Input<Tensor>("X");
    auto *y = ctx.Output<Tensor>("Out");
    auto pow_factor = ctx.Attr<float>("factor");
    const T *x_data = x->data<T>();
    T *y_data = y->mutable_data<T>(ctx.GetPlace());

    // allocate temp memory for factor on xpu
    auto xpu_context =
        ctx.device_context<paddle::platform::XPUDeviceContext>().x_context();
    xpu::ctx_guard RAII_GUARD(xpu_context);
    T *factor_data = RAII_GUARD.alloc_l3_or_gm<T>(1);
    PADDLE_ENFORCE_NOT_NULL(
        factor_data,
        platform::errors::External("XPU alloc_l3_or_gm returns nullptr"));
    memory::Copy(ctx.GetPlace(), static_cast<void *>(factor_data),
                 platform::CPUPlace(), static_cast<void *>(&pow_factor),
                 sizeof(T));

    // broadcast_pow(Context* ctx, const T* x, const T* y, T* z, const
    // std::vector<int>& xshape, const std::vector<int>& yshape);
    auto x_dims = phi::vectorize<int>(x->dims());
    int r = xpu::broadcast_pow(xpu_context, x_data, factor_data, y_data, x_dims,
                               {1});
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_pow");
  }
};

template <typename T>
struct XPUPowGradFunctor : public BaseActivationFunctor<T> {
  void operator()(const framework::ExecutionContext &ctx) const {
    const auto *x = ctx.Input<Tensor>("X");
    auto *dOut = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto *dX = ctx.Output<framework::Tensor>(framework::GradVarName("X"));

    const T *x_data = x->data<T>();
    const T *y_grad = dOut->data<T>();
    T *x_grad = dX->mutable_data<T>(ctx.GetPlace());

    // check dims: all dims should equal
    auto x_dims = phi::vectorize<int>(x->dims());
    auto dy_dims = phi::vectorize<int>(dOut->dims());
    auto dx_dims = phi::vectorize<int>(dX->dims());
    PADDLE_ENFORCE_EQ(x_dims, dy_dims, platform::errors::PreconditionNotMet(
                                           "x_dims should match dy_dims."));
    PADDLE_ENFORCE_EQ(x_dims, dx_dims, platform::errors::PreconditionNotMet(
                                           "x_dims should match dx_dims."));
    float pow_factor = ctx.Attr<float>("factor");

    auto xpu_context =
        ctx.device_context<paddle::platform::XPUDeviceContext>().x_context();
    // int pow_grad(Context* ctx, const T* x, const T* dy, T* dx, int len, float
    // factor);
    int r = xpu::pow_grad(xpu_context, x_data, y_grad, x_grad, x->numel(),
                          pow_factor);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "pow_grad");
  }
};

template <typename T>
struct XPUSoftPlusFunctor : public BaseActivationFunctor<T> {
  void operator()(const framework::ExecutionContext &ctx) const {
    const auto *x = ctx.Input<Tensor>("X");
    auto *y = ctx.Output<Tensor>("Out");
    const T *x_data = x->data<T>();
    T *y_data = y->mutable_data<T>(ctx.GetPlace());

    float beta = ctx.Attr<float>("beta");
    float threshold = ctx.Attr<float>("threshold");

    auto xpu_context =
        ctx.device_context<paddle::platform::XPUDeviceContext>().x_context();
    int r =
        xpu::softplus(xpu_context, x_data, y_data, x->numel(), beta, threshold);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "softplus");
  }
};

template <typename T>
struct XPUSoftPlusGradFunctor : public BaseActivationFunctor<T> {
  void operator()(const framework::ExecutionContext &ctx) const {
    const auto *x = ctx.Input<Tensor>("X");
    auto *dOut = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto *dX = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    const T *x_data = x->data<T>();
    const T *y_grad = dOut->data<T>();
    T *x_grad = dX->mutable_data<T>(ctx.GetPlace());

    float beta = ctx.Attr<float>("beta");
    float threshold = ctx.Attr<float>("threshold");

    auto xpu_context =
        ctx.device_context<paddle::platform::XPUDeviceContext>().x_context();
    int r = xpu::softplus_grad(
        xpu_context, reinterpret_cast<const float *>(x_data),
        reinterpret_cast<const float *>(
            x_data),  // softplus_grad do not need y_data
        reinterpret_cast<const float *>(y_grad),
        reinterpret_cast<float *>(x_grad), dX->numel(), beta, threshold);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "softplus_grad");
  }
};

template <typename T>
struct XPUSwishFunctor : public BaseActivationFunctor<T> {
  void operator()(const framework::ExecutionContext &ctx) const {
    const auto *x = ctx.Input<Tensor>("X");
    auto *y = ctx.Output<Tensor>("Out");
    const T *x_data = x->data<T>();
    T *y_data = y->mutable_data<T>(ctx.GetPlace());

    auto xpu_context =
        ctx.device_context<paddle::platform::XPUDeviceContext>().x_context();
    // int swish(Context* ctx, const T* x, T* y, int len);
    int r = xpu::swish(xpu_context, x_data, y_data, x->numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "swish");
  }
};

template <typename T>
struct XPUSwishGradFunctor : public BaseActivationFunctor<T> {
  void operator()(const framework::ExecutionContext &ctx) const {
    const auto *x = ctx.Input<Tensor>("X");
    auto *dOut = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto *dX = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    const T *x_data = x->data<T>();
    const T *y_grad = dOut->data<T>();
    T *x_grad = dX->mutable_data<T>(ctx.GetPlace());

    auto xpu_context =
        ctx.device_context<paddle::platform::XPUDeviceContext>().x_context();
    // int swish_grad(Context* ctx, const T* x, const T* dy, T* dx, int len);
    int r = xpu::swish_grad(xpu_context, x_data, y_grad, x_grad, dX->numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "swish_grad");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

#define REGISTER_ACTIVATION_XPU_KERNEL(act_type, functor, grad_functor)  \
  REGISTER_OP_XPU_KERNEL(act_type,                                       \
                         ops::XPUActivationKernel<ops::functor<float>>); \
  REGISTER_OP_XPU_KERNEL(                                                \
      act_type##_grad,                                                   \
      ops::XPUActivationGradKernel<ops::grad_functor<float>>);

REGISTER_ACTIVATION_XPU_KERNEL(abs, XPUAbsFunctor, XPUAbsGradFunctor)
REGISTER_ACTIVATION_XPU_KERNEL(hard_swish, XPUHardSwishFunctor,
                               XPUHardSwishGradFunctor)
REGISTER_ACTIVATION_XPU_KERNEL(leaky_relu, XPULeakyReluFunctor,
                               XPULeakyReluGradFunctor)
REGISTER_ACTIVATION_XPU_KERNEL(reciprocal, XPUReciprocalFunctor,
                               XPUReciprocalGradFunctor)
REGISTER_ACTIVATION_XPU_KERNEL(sigmoid, XPUSigmoidFunctor,
                               XPUSigmoidGradFunctor)
REGISTER_ACTIVATION_XPU_KERNEL(sqrt, XPUSqrtFunctor, XPUSqrtGradFunctor)
REGISTER_ACTIVATION_XPU_KERNEL(square, XPUSquareFunctor, XPUSquareGradFunctor)
REGISTER_ACTIVATION_XPU_KERNEL(softplus, XPUSoftPlusFunctor,
                               XPUSoftPlusGradFunctor)
REGISTER_ACTIVATION_XPU_KERNEL(swish, XPUSwishFunctor, XPUSwishGradFunctor)
REGISTER_ACTIVATION_XPU_KERNEL(pow, XPUPowFunctor, XPUPowGradFunctor)

REGISTER_OP_XPU_KERNEL(
    relu, ops::XPUActivationKernel<ops::XPUReluFunctor<float>>,
    ops::XPUActivationKernel<ops::XPUReluFunctor<paddle::platform::float16>>);
REGISTER_OP_XPU_KERNEL(
    relu_grad, ops::XPUActivationGradKernel<ops::XPUReluGradFunctor<float>>,
    ops::XPUActivationGradKernel<
        ops::XPUReluGradFunctor<paddle::platform::float16>>);
REGISTER_OP_XPU_KERNEL(
    tanh, ops::XPUActivationKernel<ops::XPUTanhFunctor<float>>,
    ops::XPUActivationKernel<ops::XPUTanhFunctor<paddle::platform::float16>>);
REGISTER_OP_XPU_KERNEL(
    tanh_grad, ops::XPUActivationGradKernel<ops::XPUTanhGradFunctor<float>>,
    ops::XPUActivationGradKernel<
        ops::XPUTanhGradFunctor<paddle::platform::float16>>);

REGISTER_OP_XPU_KERNEL(exp,
                       ops::XPUActivationKernel<ops::XPUExpFunctor<float>>);
REGISTER_OP_XPU_KERNEL(log,
                       ops::XPUActivationKernel<ops::XPULogFunctor<float>>);

#endif  // PADDLE_WITH_XPU
