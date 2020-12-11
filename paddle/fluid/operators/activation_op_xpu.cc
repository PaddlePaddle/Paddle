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

#include "paddle/fluid/operators/activation_op.h"
#include <string>
#include "paddle/fluid/platform/xpu_header.h"

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

template <typename DeviceContext, typename T>
void xpu_activation_forward(const framework::ExecutionContext &ctx,
                            xpu::Activation_t type) {
  const auto *x = ctx.Input<Tensor>("X");
  auto *y = ctx.Output<Tensor>("Out");
  const T *x_data = x->data<T>();
  T *y_data = y->mutable_data<T>(ctx.GetPlace());
  int r = 0;
  auto xpu_context = ctx.device_context<DeviceContext>().x_context();

  switch (type.type) {
    case xpu::Activation_t::HARD_SWISH: {
      float threshold = ctx.Attr<float>("threshold");
      float scale = ctx.Attr<float>("scale");
      float offset = ctx.Attr<float>("offset");
      PADDLE_ENFORCE_EQ(threshold, 6.0f,
                        platform::errors::External(
                            "Not support threshold [%f] in XPU", threshold));
      PADDLE_ENFORCE_EQ(
          scale, 6.0f,
          platform::errors::External("Not support scale [%f] in XPU", scale));
      PADDLE_ENFORCE_EQ(
          offset, 3.0f,
          platform::errors::External("Not support offset [%f] in XPU", offset));

      r = xpu::hard_swish(xpu_context, reinterpret_cast<const float *>(x_data),
                          reinterpret_cast<float *>(y_data), x->numel());
      break;
    }
    case xpu::Activation_t::ACT_POW: {
      type.pow_factor = ctx.Attr<float>("factor");
    }
    default: {
      r = xpu::activation_forward(xpu_context, type, x->numel(),
                                  reinterpret_cast<const float *>(x_data),
                                  reinterpret_cast<float *>(y_data));
      break;
    }
  }

  PADDLE_ENFORCE_EQ(r, XPU_SUCCESS,
                    platform::errors::External(
                        "XPU API return wrong value[%d], please check whether "
                        "Baidu Kunlun Card is properly installed.",
                        r));
}

template <typename DeviceContext, typename T>
void xpu_activation_backward(const framework::ExecutionContext &ctx,
                             xpu::Activation_t type) {
  /* TODO: relu tanh sigmoid are inplace */
  const auto *x = ctx.Input<Tensor>("X");
  auto *y = ctx.Input<Tensor>("Out");
  auto *dOut = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
  auto *dX = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
  const T *x_data = nullptr;
  const T *y_data = nullptr;
  const T *y_grad = nullptr;
  if (x != nullptr) x_data = x->data<T>();
  if (y != nullptr) y_data = y->data<T>();
  if (dOut != nullptr) y_grad = dOut->data<T>();
  T *x_grad = dX->mutable_data<T>(ctx.GetPlace());
  int r = 0;
  auto xpu_context = ctx.device_context<DeviceContext>().x_context();

  switch (type.type) {
    case xpu::Activation_t::HARD_SWISH: {
      float threshold = ctx.Attr<float>("threshold");
      float scale = ctx.Attr<float>("scale");
      float offset = ctx.Attr<float>("offset");
      PADDLE_ENFORCE_EQ(threshold, 6.0f,
                        platform::errors::External(
                            "Not support threshold [%f] in XPU", threshold));
      PADDLE_ENFORCE_EQ(
          scale, 6.0f,
          platform::errors::External("Not support scale [%f] in XPU", scale));
      PADDLE_ENFORCE_EQ(
          offset, 3.0f,
          platform::errors::External("Not support offset [%f] in XPU", offset));
      r = xpu::hard_swish_grad(xpu_context,
                               reinterpret_cast<const float *>(x_data),
                               reinterpret_cast<const float *>(y_data),
                               reinterpret_cast<const float *>(y_grad),
                               reinterpret_cast<float *>(x_grad), dX->numel());
      break;
    }
    default: {
      r = xpu::activation_backward(xpu_context, type, dX->numel(),
                                   reinterpret_cast<const float *>(x_data),
                                   reinterpret_cast<const float *>(y_data),
                                   reinterpret_cast<const float *>(y_grad),
                                   reinterpret_cast<float *>(x_grad));
      break;
    }
  }

  PADDLE_ENFORCE_EQ(r, XPU_SUCCESS,
                    platform::errors::External(
                        "XPU API return wrong value[%d], please check whether "
                        "Baidu Kunlun Card is properly installed.",
                        r));
}

template <typename T, xpu::Activation_t::act_enum algorithm>
struct XPUActivationFunc : public BaseActivationFunctor<T> {
  void operator()(const framework::ExecutionContext &ctx) const {
    xpu_activation_forward<paddle::platform::XPUDeviceContext, T>(ctx,
                                                                  algorithm);
  }
};

template <typename T, xpu::Activation_t::act_enum algorithm>
struct XPUActivationGradFunc : public BaseActivationFunctor<T> {
  void operator()(const framework::ExecutionContext &ctx) const {
    xpu_activation_backward<paddle::platform::XPUDeviceContext, T>(ctx,
                                                                   algorithm);
  }
};

template <typename T>
using XPUReluFunctor = XPUActivationFunc<T, xpu::Activation_t::RELU>;
template <typename T>
using XPUSigmoidFunctor = XPUActivationFunc<T, xpu::Activation_t::SIGMOID>;
template <typename T>
using XPUTanhFunctor = XPUActivationFunc<T, xpu::Activation_t::TANH>;
template <typename T>
using XPUGeluFunctor = XPUActivationFunc<T, xpu::Activation_t::GELU>;
template <typename T>
using XPULogFunctor = XPUActivationFunc<T, xpu::Activation_t::LOG>;
template <typename T>
using XPUSquareFunctor = XPUActivationFunc<T, xpu::Activation_t::SQUARE>;
template <typename T>
using XPUHardSwishFunctor = XPUActivationFunc<T, xpu::Activation_t::HARD_SWISH>;
template <typename T>
using XPUSuareGradFunctor = XPUActivationGradFunc<T, xpu::Activation_t::SQUARE>;
template <typename T>
using XPUReluGradFunctor = XPUActivationGradFunc<T, xpu::Activation_t::RELU>;
template <typename T>
using XPUSigmoidGradFunctor =
    XPUActivationGradFunc<T, xpu::Activation_t::SIGMOID>;
template <typename T>
using XPUTanhGradFunctor = XPUActivationGradFunc<T, xpu::Activation_t::TANH>;
template <typename T>
using XPUGeluGradFunctor = XPUActivationGradFunc<T, xpu::Activation_t::GELU>;
template <typename T>
using XPUSqrtFunctor = XPUActivationFunc<T, xpu::Activation_t::SQRT>;
template <typename T>
using XPUSqrtGradFunctor = XPUActivationGradFunc<T, xpu::Activation_t::SQRT>;
template <typename T>
using XPUHardSwishGradFunctor =
    XPUActivationGradFunc<T, xpu::Activation_t::HARD_SWISH>;
template <typename T>
using XPUACTPowFunctor = XPUActivationFunc<T, xpu::Activation_t::ACT_POW>;
template <typename T>
using XPUABSFunctor = XPUActivationFunc<T, xpu::Activation_t::ABS>;
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

#define REGISTER_ACTIVATION_XPU_KERNEL(act_type, functor, grad_functor)  \
  REGISTER_OP_XPU_KERNEL(act_type,                                       \
                         ops::XPUActivationKernel<ops::functor<float>>); \
  REGISTER_OP_XPU_KERNEL(                                                \
      act_type##_grad,                                                   \
      ops::XPUActivationGradKernel<ops::grad_functor<float>>);

REGISTER_ACTIVATION_XPU_KERNEL(relu, XPUReluFunctor, XPUReluGradFunctor)
REGISTER_ACTIVATION_XPU_KERNEL(tanh, XPUTanhFunctor, XPUTanhGradFunctor)
REGISTER_ACTIVATION_XPU_KERNEL(sigmoid, XPUSigmoidFunctor,
                               XPUSigmoidGradFunctor)
REGISTER_ACTIVATION_XPU_KERNEL(gelu, XPUGeluFunctor, XPUGeluGradFunctor)
REGISTER_ACTIVATION_XPU_KERNEL(sqrt, XPUSqrtFunctor, XPUSqrtGradFunctor)
REGISTER_ACTIVATION_XPU_KERNEL(square, XPUSquareFunctor, XPUSuareGradFunctor)
REGISTER_ACTIVATION_XPU_KERNEL(hard_swish, XPUHardSwishFunctor,
                               XPUHardSwishGradFunctor)
REGISTER_OP_XPU_KERNEL(log,
                       ops::XPUActivationKernel<ops::XPULogFunctor<float>>);
REGISTER_OP_XPU_KERNEL(pow,
                       ops::XPUActivationKernel<ops::XPUACTPowFunctor<float>>);
REGISTER_OP_XPU_KERNEL(abs,
                       ops::XPUActivationKernel<ops::XPUABSFunctor<float>>);

#endif  // PADDLE_WITH_XPU
