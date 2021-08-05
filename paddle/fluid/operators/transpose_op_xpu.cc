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
#include "paddle/fluid/operators/transpose_op.h"
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/platform/xpu/xpu_header.h"

namespace paddle {
namespace operators {

using framework::Tensor;

template <typename DeviceContext, typename T>
class TransposeXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto x = context.Input<framework::Tensor>("X");
    auto out = context.Output<framework::Tensor>("Out");

    // axis is permute
    auto axis = context.Attr<std::vector<int>>("axis");
    int ndims = axis.size();
    const auto x_dims = x->dims();
    const T* x_data = x->data<T>();
    T* y_data = out->mutable_data<T>(context.GetPlace());
    if (out->numel() == 0) {
      return;
    }

    std::vector<int> x_shape_host(ndims, 0);
    for (int i = 0; i < ndims; ++i) {
      x_shape_host[i] = x_dims[i];
    }
    auto& dev_ctx = context.template device_context<DeviceContext>();
    int r = xpu::transpose<T>(dev_ctx.x_context(), x_data, y_data, x_shape_host,
                              axis);
    PADDLE_ENFORCE_EQ(
        r, xpu::Error_t::SUCCESS,
        platform::errors::External("XPU kernel error! error code=%d", r));
  }
};

template <typename DeviceContext, typename T>
class TransposeGradXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* out_grad =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* x_grad =
        context.Output<framework::Tensor>(framework::GradVarName("X"));
    if (!x_grad) return;

    x_grad->mutable_data<T>(context.GetPlace());
    std::vector<int> axis = context.Attr<std::vector<int>>("axis");
    std::vector<int> reversed_axis(axis);
    for (size_t i = 0; i < axis.size(); i++) {
      reversed_axis[axis[i]] = i;
    }

    int ndims = axis.size();
    std::vector<int> out_shape_host(ndims, 0);
    for (int i = 0; i < ndims; ++i) {
      out_shape_host[i] = out_grad->dims()[i];
    }
    auto& dev_ctx = context.template device_context<DeviceContext>();
    int r = xpu::transpose<T>(dev_ctx.x_context(), out_grad->data<T>(),
                              x_grad->data<T>(), out_shape_host, reversed_axis);
    PADDLE_ENFORCE_EQ(
        r, xpu::Error_t::SUCCESS,
        platform::errors::External("XPU kernel error! error code=%d", r));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_XPU_KERNEL(
    transpose,
    ops::TransposeXPUKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
    transpose_grad,
    ops::TransposeGradXPUKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
    transpose2,
    ops::TransposeXPUKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
    transpose2_grad,
    ops::TransposeGradXPUKernel<paddle::platform::XPUDeviceContext, float>);

#endif  // PADDLE_WITH_XPU
