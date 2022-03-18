/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <cmath>
#include "paddle/fluid/operators/elementwise/elementwise_op.h"

#ifdef __xpu__
#include <memory>
#include <string>
#include "paddle/fluid/operators/elementwise/elementwise_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_broadcast.cu.h"
#include "paddle/fluid/operators/elementwise/elementwise_xpu.h"
#include "paddle/fluid/platform/device/device_wrapper.h"
#endif

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class ElementwiseMaxKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(__NVCC__) || defined(__xpu__)
    std::vector<const framework::Tensor*> ins;
    std::vector<framework::Tensor*> outs;
#ifdef __NVCC__
    const auto& dev_ctx =
        ctx.template device_context<platform::CUDADeviceContext>();
#else
    const auto& dev_ctx =
        ctx.template device_context<platform::XPUDeviceContext>();
#endif
    int axis = PackTensorsIntoVector<T>(ctx, &ins, &outs);
    paddle::operators::LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T,
                                                   T>(dev_ctx, ins, &outs, axis,
                                                      MaxFunctor<T>());
#else
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* y = ctx.Input<framework::LoDTensor>("Y");
    auto* z = ctx.Output<framework::LoDTensor>("Out");

    z->mutable_data<T>(ctx.GetPlace());
    int axis = ctx.Attr<int>("axis");
    ElementwiseComputeEx<MaxFunctor<T>, DeviceContext, T>(ctx, x, y, axis,
                                                          MaxFunctor<T>(), z);
#endif
  }
};

template <typename T>
struct MaxGradDx {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const {
    return dout * static_cast<T>(x > y);
  }
};

template <typename T>
struct MaxGradDy {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const {
    return dout * static_cast<T>(x <= y);
  }
};

template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_same<DeviceContext, platform::CPUDeviceContext>::value>::type
ElementwiseMaxGrad(const framework::ExecutionContext& ctx,
                   const framework::Tensor* x, const framework::Tensor* y,
                   const framework::Tensor* out, const framework::Tensor* dout,
                   framework::Tensor* dx, framework::Tensor* dy) {
  int axis = ctx.Attr<int>("axis");
  ElemwiseGradCompute<DeviceContext, T, MaxGradDx<T>, MaxGradDy<T>>(
      ctx, *x, *y, *out, *dout, axis, dx, dy, MaxGradDx<T>(), MaxGradDy<T>());
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_same<DeviceContext, platform::CUDADeviceContext>::value>::type
ElementwiseMaxGrad(const framework::ExecutionContext& ctx,
                   const framework::Tensor* x, const framework::Tensor* y,
                   const framework::Tensor* out, const framework::Tensor* dout,
                   framework::Tensor* dx, framework::Tensor* dy);
#endif

template <typename DeviceContext, typename T>
class ElementwiseMaxGradKernel : public ElemwiseGradKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    ElemwiseGradKernel<T>::Compute(ctx);
    using Tensor = framework::Tensor;

    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* out = dout;  // out is not necessary
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));

    ElementwiseMaxGrad<DeviceContext, T>(ctx, x, y, out, dout, dx, dy);
  }
};

}  // namespace operators
}  // namespace paddle
