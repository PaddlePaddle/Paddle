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

#include "paddle/fluid/operators/softmax_op.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DDim = framework::DDim;

template <typename DeviceContext, typename T>
class SoftmaxXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* out = context.Output<Tensor>("Out");
    const int rank = x->dims().size();
    int axis = CanonicalAxis(context.Attr<int>("axis"), rank);

    // allocate memory on device.
    out->mutable_data<T>(context.GetPlace());

    std::vector<int> x_dims;
    for (int i = 0; i < rank; i++) {
      x_dims.push_back(x->dims()[i]);
    }
    if (axis < 0) {
      axis += rank;
    }

    auto& dev_ctx = context.template device_context<DeviceContext>();
    int r = xpu::softmax<T>(dev_ctx.x_context(), x->data<float>(),
                            out->data<float>(), x_dims, axis);
    PADDLE_ENFORCE_EQ(
        r, XPU_SUCCESS,
        platform::errors::External("XPU API(softmax2d_forward) return wrong "
                                   "value[%d %s]",
                                   r, XPUAPIErrorMsg[r]));
  }
};

template <typename DeviceContext, typename T>
class SoftmaxGradXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* out = context.Input<Tensor>("Out");
    auto* dout = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = context.Output<Tensor>(framework::GradVarName("X"));
    const int rank = dx->dims().size();
    int axis = CanonicalAxis(context.Attr<int>("axis"), rank);

    // allocate memory on device.
    dx->mutable_data<T>(context.GetPlace());

    std::vector<int> x_dims;
    for (int i = 0; i < rank; i++) {
      x_dims.push_back(dx->dims()[i]);
    }
    if (axis < 0) {
      axis += rank;
    }

    auto& dev_ctx = context.template device_context<DeviceContext>();
    int r = xpu::softmax_grad<T>(dev_ctx.x_context(), out->data<float>(),
                                 dout->data<float>(), dx->data<float>(), x_dims,
                                 axis);
    PADDLE_ENFORCE_EQ(
        r, XPU_SUCCESS,
        platform::errors::External("XPU API(softmax2d_backward) return wrong "
                                   "value[%d %s]",
                                   r, XPUAPIErrorMsg[r]));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_XPU_KERNEL(
    softmax, ops::SoftmaxXPUKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
    softmax_grad,
    ops::SoftmaxGradXPUKernel<paddle::platform::XPUDeviceContext, float>);

#endif  // PADDLE_WITH_XPU
