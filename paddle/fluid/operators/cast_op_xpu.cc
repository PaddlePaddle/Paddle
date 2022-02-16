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

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/cast_op.h"
#include "paddle/fluid/platform/float16.h"
#include "xpu/refactor/math.h"

#include "paddle/pten/kernels/cast_kernel.h"

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
    auto out_dtype =
        static_cast<var_type::Type>(context.Attr<int>("out_dtype"));

    auto& dev_ctx = context.template device_context<DeviceContext>();

    out->mutable_data(dev_ctx.GetPlace(),
                      static_cast<framework::proto::VarType::Type>(out_dtype));

    auto pt_out_dtype = framework::TransToPtenDataType(
        static_cast<framework::proto::VarType::Type>(out_dtype));
    // call pten kernel
    pten::CastKernel<InT>(
        static_cast<const typename paddle::framework::ConvertToPtenContext<
            DeviceContext>::TYPE&>(dev_ctx),
        *in, pt_out_dtype, out);
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
