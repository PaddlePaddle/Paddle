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
#include <memory>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/cast_op.h"
#include "paddle/fluid/platform/float16.h"
#include "xpu/refactor/math.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename InT>
class CastXPUKernel : public framework::OpKernel<InT> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<framework::Tensor>("X");
    auto* out = context.Output<framework::Tensor>("Out");
    auto in_type = static_cast<framework::proto::VarType::Type>(
        context.Attr<int>("in_dtype"));
    auto out_type = static_cast<framework::proto::VarType::Type>(
        context.Attr<int>("out_dtype"));
    auto* in_data = in->data<InT>();
    auto numel = in->numel();
    auto& dev_ctx = context.template device_context<DeviceContext>();
    int r = -1;
    if (out_type == framework::proto::VarType::FP32) {
      auto* out_data = out->mutable_data<float>(context.GetPlace());
      r = xpu::cast_v2<InT, float>(dev_ctx.x_context(), in_data, out_data,
                                   numel);
    } else if (out_type == framework::proto::VarType::INT32) {
      auto* out_data = out->mutable_data<int>(context.GetPlace());
      r = xpu::cast_v2<InT, int32_t>(dev_ctx.x_context(), in_data, out_data,
                                     numel);
    } else if (out_type == framework::proto::VarType::INT64) {
      auto* out_data = out->mutable_data<int64_t>(context.GetPlace());
      r = xpu::cast_v2<InT, int64_t>(dev_ctx.x_context(), in_data, out_data,
                                     numel);
    } else {
      PADDLE_THROW(platform::errors::Unavailable("Not supported cast %d -> %d",
                                                 in_type, out_type));
    }
    PADDLE_ENFORCE_EQ(
        r, XPU_SUCCESS,
        platform::errors::External(
            "XPU API return wrong value[%d], please check whether "
            "Baidu Kunlun Card is properly installed.",
            r));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    cast, ops::CastXPUKernel<paddle::platform::XPUDeviceContext, int32_t>,
    ops::CastXPUKernel<paddle::platform::XPUDeviceContext, float>,
    ops::CastXPUKernel<paddle::platform::XPUDeviceContext, int64_t>);
#endif
