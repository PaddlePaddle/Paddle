// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/operators/stack_op.h"
#include <string>
#include <vector>
#include "paddle/fluid/operators/concat_op.h"
#include "paddle/fluid/platform/device/xpu/xpu_header.h"

namespace paddle {
namespace operators {

using framework::Tensor;
template <typename DeviceContext, typename T>
class StackXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto x = ctx.MultiInput<Tensor>("X");
    auto* y = ctx.Output<Tensor>("Y");
    int axis = ctx.Attr<int>("axis");
    if (axis < 0) {
      axis += x[0]->dims().size() + 1;
    }
    auto* y_data = y->mutable_data<T>(ctx.GetPlace());

    auto& dim = x[0]->dims();
    std::vector<int> xdims;
    for (auto i = 0; i < dim.size(); ++i) {
      xdims.push_back(dim[i]);
    }
    xdims.push_back(1);
    std::vector<std::vector<int>> xdims_list;
    int n = static_cast<int>(x.size());
    for (int i = 0; i < n; i++) {
      xdims_list.push_back(xdims);
    }

    std::vector<const T*> x_list;
    for (int i = 0; i < n; i++) {
      x_list.push_back(x[i]->data<T>());
    }

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    int r =
        xpu::concat<T>(dev_ctx.x_context(), x_list, y_data, xdims_list, axis);
    PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                      platform::errors::External(
                          "The stack XPU API return wrong value[%d %s]", r,
                          XPUAPIErrorMsg[r]));
  }
};

template <typename DeviceContext, typename T>
class StackGradXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* dy = ctx.Input<Tensor>(framework::GradVarName("Y"));
    auto dx = ctx.MultiOutput<Tensor>(framework::GradVarName("X"));
    auto axis = ctx.Attr<int>("axis");
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto dy_dims = dy->dims();

    if (axis < 0) axis += dy_dims.size() + 1;
    auto dy_shape = framework::vectorize<int>(dy_dims);

    std::vector<int> dx_dims_list(dx.size(), 1);
    std::vector<T*> dx_lists;
    for (auto out : dx) {
      dx_lists.push_back(out->mutable_data<T>(ctx.GetPlace()));
    }

    int r = xpu::split<T>(dev_ctx.x_context(), dy->data<T>(), dx_lists,
                          dy_shape, dx_dims_list, axis);
    PADDLE_ENFORCE_EQ(r, XPU_SUCCESS,
                      platform::errors::External(
                          "The stack_grad XPU kernel return wrong value[%d %s]",
                          r, XPUAPIErrorMsg[r]));
  }
};

}  // namespace operators
}  // namespace paddle

namespace plat = paddle::platform;
namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(stack,
                       ops::StackXPUKernel<plat::XPUDeviceContext, float>,
                       ops::StackXPUKernel<plat::XPUDeviceContext, int>,
                       ops::StackXPUKernel<plat::XPUDeviceContext, int64_t>);
REGISTER_OP_XPU_KERNEL(stack_grad,
                       ops::StackGradXPUKernel<plat::XPUDeviceContext, float>,
                       ops::StackGradXPUKernel<plat::XPUDeviceContext, int>);
#endif
