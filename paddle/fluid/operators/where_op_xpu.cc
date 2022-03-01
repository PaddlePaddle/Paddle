// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifdef PADDLE_WITH_XPU

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class WhereXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* condition = context.Input<framework::Tensor>("Condition");
    auto* X = context.Input<framework::Tensor>("X");
    auto* Y = context.Input<framework::Tensor>("Y");
    auto* out = context.Output<framework::Tensor>("Out");

    const bool* cond_data = condition->data<bool>();
    const T* x_data = X->data<T>();
    const T* y_data = Y->data<T>();
    T* out_data = out->mutable_data<T>(context.GetPlace());

    auto cond_dims = phi::vectorize<int>(condition->dims());
    auto input_dims = phi::vectorize<int>(X->dims());

    auto& dev_ctx = context.template device_context<DeviceContext>();
    int ret = xpu::select(dev_ctx.x_context(), cond_data, x_data, y_data,
                          out_data, cond_dims, input_dims);
    PADDLE_ENFORCE_EQ(ret, XPU_SUCCESS,
                      platform::errors::External(
                          "XPU select kernel return wrong value[%d %s]", ret,
                          XPUAPIErrorMsg[ret]));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_XPU_KERNEL(
    where, ops::WhereXPUKernel<paddle::platform::XPUDeviceContext, float>,
    ops::WhereXPUKernel<paddle::platform::XPUDeviceContext, int>,
    ops::WhereXPUKernel<paddle::platform::XPUDeviceContext, int64_t>);
#endif
