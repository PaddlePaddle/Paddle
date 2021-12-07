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

#include "paddle/fluid/operators/p_norm_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class PnormNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in_x = ctx.Input<framework::Tensor>("X");
    auto* out_norm = ctx.Output<framework::Tensor>("Out");
    out_norm->mutable_data<T>(ctx.GetPlace());

    float porder = ctx.Attr<float>("porder");
    int axis = ctx.Attr<int>("axis");
    bool keepdim = ctx.Attr<bool>("keepdim");

    auto xdim = in_x->dims();
    if (axis < 0) axis = xdim.size() + axis;

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    int p = 0;
    bool combine_op =
        !(porder == 0 || porder == INFINITY || porder == -INFINITY);
    if (porder == INFINITY) {
      p = INT_MAX;
    } else if (porder == -INFINITY) {
      p = INT_MIN;
    } else {
      p = static_cast<int>(porder);
      float t = 0;
      float diff = abs(std::modf(porder, &t));
      if (diff < 1e-5) {
        combine_op = false;
      }
    }

    if (!combine_op) {
      const auto& runner = NpuOpRunner("LpNorm", {*in_x}, {*out_norm},
                                       {{"p", p},
                                        {"axes", std::vector<int32_t>({axis})},
                                        {"keep_dims", keepdim}});
      runner.Run(stream);
    } else {
      Tensor tmp_x;
      tmp_x.mutable_data<T>(xdim, ctx.GetPlace());

      const auto& power_runner1 =
          NpuOpRunner("Power", {*in_x}, {tmp_x},
                      {{"power", porder}, {"scale", 1.0f}, {"shift", 0.0f}});
      power_runner1.Run(stream);

      const auto& reduce_runner = NpuOpRunner(
          "ReduceSumD", {tmp_x}, {*out_norm},
          {{"axes", std::vector<int32_t>({axis})}, {"keep_dims", keepdim}});
      reduce_runner.Run(stream);

      const auto& power_runner2 = NpuOpRunner(
          "Power", {*out_norm}, {*out_norm},
          {{"power", 1 / porder}, {"scale", 1.0f}, {"shift", 0.0f}});
      power_runner2.Run(stream);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(
    p_norm, ops::PnormNPUKernel<plat::NPUDeviceContext, float>,
    ops::PnormNPUKernel<plat::NPUDeviceContext, plat::float16>);
