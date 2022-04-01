/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/operators/arg_min_max_op_base.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class ArgMaxXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* out = ctx.Output<framework::LoDTensor>("Out");
    auto dtype = ctx.Attr<int>("dtype");
    PADDLE_ENFORCE_EQ(
        (dtype < 0 || dtype == 3), true,
        platform::errors::InvalidArgument(
            "The attribute of dtype in xpu argmin/argmax must be [%s], but "
            "received [%s]",
            paddle::framework::DataTypeToString(
                framework::proto::VarType::INT64),
            paddle::framework::DataTypeToString(
                static_cast<framework::proto::VarType::Type>(dtype))));

    out->template mutable_data<int64_t>(ctx.GetPlace());
    auto axis = ctx.Attr<int64_t>("axis");
    const bool& flatten = ctx.Attr<bool>("flatten");
    framework::DDim x_dims;
    if (flatten) {
      x_dims = phi::make_ddim({x->numel()});
      // if flatten, the axis just as 0
      axis = 0;
    } else {
      x_dims = x->dims();
      if (axis < 0) axis += x_dims.size();
    }
    auto xdims_vec = phi::vectorize<int>(x_dims);
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    int r = xpu::argmax(dev_ctx.x_context(), x->data<T>(), out->data<int64_t>(),
                        xdims_vec, axis);
    PADDLE_ENFORCE_EQ(r, XPU_SUCCESS,
                      platform::errors::External(
                          "XPU argmax kernel return wrong value[%d %s].", r,
                          XPUAPIErrorMsg[r]));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    arg_max, ops::ArgMaxXPUKernel<paddle::platform::XPUDeviceContext, float>);

#endif
