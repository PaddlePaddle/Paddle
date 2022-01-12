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

using var_type = framework::proto::VarType;
namespace plat = paddle::platform;

template <typename DeviceContext, typename InT>
class CastXPUKernel : public framework::OpKernel<InT> {
  using XPUInTDType = typename XPUTypeTrait<InT>::Type;
  using float16 = typename XPUTypeTrait<paddle::platform::float16>::Type;

 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<framework::Tensor>("X");
    auto* out = context.Output<framework::Tensor>("Out");
    auto in_type = static_cast<var_type::Type>(context.Attr<int>("in_dtype"));
    auto out_type = static_cast<var_type::Type>(context.Attr<int>("out_dtype"));
    auto* in_data = in->data<InT>();

    auto numel = in->numel();
    auto& dev_ctx = context.template device_context<DeviceContext>();
    int r = -1;
    switch (out_type) {
      case var_type::FP32:
        r = xpu::cast_v2<XPUInTDType, float>(
            dev_ctx.x_context(), reinterpret_cast<const XPUInTDType*>(in_data),
            out->mutable_data<float>(context.GetPlace()), numel);
        break;
      case var_type::FP16:
        r = xpu::cast_v2<XPUInTDType, float16>(
            dev_ctx.x_context(), reinterpret_cast<const XPUInTDType*>(in_data),
            reinterpret_cast<float16*>(
                out->mutable_data<plat::float16>(context.GetPlace())),
            numel);
        break;
      case var_type::INT64:
        r = xpu::cast_v2<XPUInTDType, int64_t>(
            dev_ctx.x_context(), reinterpret_cast<const XPUInTDType*>(in_data),
            out->mutable_data<int64_t>(context.GetPlace()), numel);
        break;
      case var_type::INT32:
        r = xpu::cast_v2<XPUInTDType, int32_t>(
            dev_ctx.x_context(), reinterpret_cast<const XPUInTDType*>(in_data),
            out->mutable_data<int>(context.GetPlace()), numel);
        break;
      case var_type::BOOL:
        r = xpu::cast_v2<XPUInTDType, bool>(
            dev_ctx.x_context(), reinterpret_cast<const XPUInTDType*>(in_data),
            out->mutable_data<bool>(context.GetPlace()), numel);
        break;
      default:
        PADDLE_THROW(platform::errors::Unavailable(
            "Not supported cast %d -> %d", in_type, out_type));
    }
    PADDLE_ENFORCE_EQ(
        r, XPU_SUCCESS,
        platform::errors::External("XPU CAST API return wrong value[%d %s].", r,
                                   XPUAPIErrorMsg[r]));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    cast, ops::CastXPUKernel<paddle::platform::XPUDeviceContext, int32_t>,
    ops::CastXPUKernel<paddle::platform::XPUDeviceContext, float>,
    ops::CastXPUKernel<paddle::platform::XPUDeviceContext,
                       paddle::platform::float16>,
    ops::CastXPUKernel<paddle::platform::XPUDeviceContext, int64_t>,
    ops::CastXPUKernel<paddle::platform::XPUDeviceContext, bool>);
#endif
