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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device/device_wrapper.h"
#include "paddle/fluid/platform/device/xpu/xpu_header.h"
namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class SignXPUKernel : public framework::OpKernel<T> {
 public:
  virtual void Compute(const framework::ExecutionContext& context) const {
    auto* out = context.Output<framework::Tensor>("Out");
    auto* in = context.Input<framework::Tensor>("X");
    out->mutable_data<T>(in->place());
    auto xpu_context = context.device_context<DeviceContext>().x_context();
    // int sign(Context* ctx, const T* x , T* y, int len);
    int r = xpu::sign(xpu_context, in->data<T>(), out->data<T>(), in->numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "sign");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    sign, ops::SignXPUKernel<paddle::platform::XPUDeviceContext, float>);

#endif
