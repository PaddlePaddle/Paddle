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

#include "paddle/fluid/operators/mean_op.h"
#ifdef PADDLE_WITH_XPU
#include <memory>
#include <string>
#include <unordered_map>

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class MeanXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<Tensor>("X");
    auto* output = context.Output<Tensor>("Out");
    output->mutable_data<T>(context.GetPlace());
    auto& dev_ctx = context.template device_context<DeviceContext>();
    const T* x_data = input->data<T>();
    T* y_data = output->data<T>();
    std::vector<int> x_shape;
    x_shape.push_back(1);
    x_shape.push_back(input->numel());
    std::vector<int> rdims = {1};
    int r = xpu::reduce_mean(
        dev_ctx.x_context(), reinterpret_cast<const XPUType*>(x_data),
        reinterpret_cast<XPUType*>(y_data), x_shape, rdims);
    PADDLE_ENFORCE_EQ(r, XPU_SUCCESS,
                      platform::errors::External(
                          "XPU reduce_mean kernel return wrong value[%d %s]", r,
                          XPUAPIErrorMsg[r]));
  }
};
template <typename DeviceContext, typename T>
class MeanGradXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto OG = context.Input<Tensor>(framework::GradVarName("Out"));
    PADDLE_ENFORCE_EQ(OG->numel(), 1, platform::errors::InvalidArgument(
                                          "Mean Gradient should be scalar"));
    auto IG = context.Output<Tensor>(framework::GradVarName("X"));
    IG->mutable_data<T>(context.GetPlace());
    auto& dev_ctx = context.template device_context<DeviceContext>();

    XPUType* dx = reinterpret_cast<XPUType*>(IG->data<T>());

    const T* dy = OG->data<T>();
    T dy0_value;
    xpu_wait(dev_ctx.x_context()->xpu_stream);
    memory::Copy(platform::CPUPlace(), &dy0_value, OG->place(), dy, sizeof(T));
    float dy0_fp32 = static_cast<float>(dy0_value);
    dy0_fp32 = dy0_fp32 / static_cast<float>(IG->numel());

    int r = xpu::constant(dev_ctx.x_context(), dx, IG->numel(),
                          static_cast<XPUType>(dy0_fp32));
    PADDLE_ENFORCE_EQ(r, XPU_SUCCESS,
                      platform::errors::External(
                          "XPU constant kernel return wrong value[%d %s]", r,
                          XPUAPIErrorMsg[r]));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    mean, ops::MeanXPUKernel<paddle::platform::XPUDeviceContext, float>,
    ops::MeanXPUKernel<paddle::platform::XPUDeviceContext,
                       paddle::platform::float16>);
REGISTER_OP_XPU_KERNEL(
    mean_grad,
    ops::MeanGradXPUKernel<paddle::platform::XPUDeviceContext, float>,
    ops::MeanGradXPUKernel<paddle::platform::XPUDeviceContext,
                           paddle::platform::float16>);
#endif
