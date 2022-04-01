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

#ifdef PADDLE_WITH_XPU
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class LabelSmoothXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto* out_t = ctx.Output<framework::LoDTensor>("Out");
    auto* in_t = ctx.Input<framework::LoDTensor>("X");
    auto* dist_t = ctx.Input<framework::Tensor>("PriorDist");
    auto label_dim = in_t->dims()[in_t->dims().size() - 1];
    auto ptr = out_t->mutable_data<T>(ctx.GetPlace());

    auto epsilon = ctx.Attr<float>("epsilon");
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    if (dist_t) {
      PADDLE_THROW(
          platform::errors::External("XPU doesn't support dist label smooth"));
    } else {
      int r = xpu::label_smooth<T>(dev_ctx.x_context(), in_t->data<T>(), ptr,
                                   in_t->numel(), epsilon, label_dim);
      PADDLE_ENFORCE_EQ(
          r, XPU_SUCCESS,
          platform::errors::External("XPU API(label_smooth) return wrong "
                                     "value[%d %s]",
                                     r, XPUAPIErrorMsg[r]));
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_XPU_KERNEL(
    label_smooth,
    ops::LabelSmoothXPUKernel<paddle::platform::XPUDeviceContext, float>);
#endif
