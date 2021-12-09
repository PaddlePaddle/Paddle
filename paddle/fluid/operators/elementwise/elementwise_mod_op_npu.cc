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

#include "paddle/fluid/operators/elementwise/elementwise_mod_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_npu.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class ElementwiseModNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx =
        ctx.template device_context<paddle::platform::NPUDeviceContext>();
    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* out = ctx.Output<Tensor>("Out");
    int axis = ctx.Attr<int>("axis");

    auto x_dims = x->dims();
    auto y_dims = y->dims();

    axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);

    bool direct_compute = false;
    if (x_dims.size() >= y_dims.size()) {
      direct_compute =
          y_dims == framework::slice_ddim(x_dims, axis, x_dims.size());
    } else {
      direct_compute =
          x_dims == framework::slice_ddim(y_dims, axis, y_dims.size());
    }

    Tensor transformed_x, transformed_y;
    if (direct_compute) {
      transformed_x.ShareDataWith(*x);
      transformed_y.ShareDataWith(*y);
    } else {
      NpuElementWiseOpBroadcast<T>(dev_ctx, x, y, axis, &transformed_x,
                                   &transformed_y);
    }
    out->mutable_data<T>(ctx.GetPlace());
    const auto& runner =
        NpuOpRunner("FloorMod", {transformed_x, transformed_y}, {*out}, {});
    auto stream = dev_ctx.stream();
    runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    elementwise_mod,
    ops::ElementwiseModNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::ElementwiseModNPUKernel<paddle::platform::NPUDeviceContext, double>,
    ops::ElementwiseModNPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::ElementwiseModNPUKernel<paddle::platform::NPUDeviceContext, int64_t>,
    ops::ElementwiseModNPUKernel<paddle::platform::NPUDeviceContext,
                                 paddle::platform::float16>);
