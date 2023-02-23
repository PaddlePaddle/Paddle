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

#include "paddle/fluid/operators/elementwise/elementwise_mul_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_npu.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

using NPUDeviceContext = platform::NPUDeviceContext;

template <typename T>
static void ReduceDims(const framework::ExecutionContext& ctx,
                       const aclrtStream& stream,
                       const int axis,
                       const framework::DDim& ddims,
                       const framework::DDim& brd_ddims,
                       const phi::DenseTensor& in,
                       phi::DenseTensor* out) {
  std::vector<int64_t> axes;
  int64_t brd_size = brd_ddims.size();
  int64_t org_size = ddims.size();
  // int64_t diff = brd_dims.size() - dims.size();
  for (int64_t i = 0; i < brd_size; ++i) {
    if (i < axis || i >= org_size + axis) {
      axes.push_back(i);
      continue;
    }
    if (brd_ddims[i] > ddims[i - axis]) {
      axes.push_back(i);
    }
  }
  // LOG(INFO) << "axes = " << phi::make_ddim(axes).to_str();
  out->mutable_data<T>(ctx.GetPlace());
  const auto& runner = NpuOpRunner(
      "ReduceSumD", {in}, {*out}, {{"axes", axes}, {"keep_dims", false}});
  runner.Run(stream);
}

template <typename T>
class ElementwiseMulNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<NPUDeviceContext>();
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* y = ctx.Input<phi::DenseTensor>("Y");
    auto* out = ctx.Output<phi::DenseTensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    int axis = ctx.Attr<int>("axis");

    bool direct_compute = false;
    auto x_dims = x->dims();
    auto y_dims = y->dims();
    axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
    if (x_dims.size() >= y_dims.size()) {
      direct_compute = x_dims.size() == (y_dims.size() + axis);
    } else {
      direct_compute = y_dims.size() == (x_dims.size() + axis);
    }

    auto stream = ctx.template device_context<NPUDeviceContext>().stream();

    if (direct_compute) {
      const auto& runner = NpuOpRunner("Mul", {*x, *y}, {*out}, {});
      runner.Run(stream);
    } else {
      phi::DenseTensor trans_x, trans_y;
      NpuElementWiseOpBroadcast<T>(dev_ctx, x, y, axis, &trans_x, &trans_y);
      const auto& runner = NpuOpRunner("Mul", {trans_x, trans_y}, {*out}, {});
      runner.Run(stream);
    }
  }
};

template <typename T>
class ElementwiseMulGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<NPUDeviceContext>();
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* y = ctx.Input<phi::DenseTensor>("Y");
    auto* dout = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<phi::DenseTensor>(framework::GradVarName("Y"));
    int axis = ctx.Attr<int>("axis");

    axis = (axis == -1 ? std::abs(x->dims().size() - y->dims().size()) : axis);
    auto stream = ctx.template device_context<NPUDeviceContext>().stream();

    phi::DenseTensor trans_x, trans_y;
    NpuElementWiseOpBroadcast<T>(dev_ctx, x, y, axis, &trans_x, &trans_y);

    if (dx) {
      if (dx->dims() == dout->dims()) {
        dx->mutable_data<T>(ctx.GetPlace());
        const auto& runner_dx = NpuOpRunner("Mul", {*dout, trans_y}, {*dx}, {});
        runner_dx.Run(stream);
      } else {
        phi::DenseTensor dx_temp(x->type());
        dx_temp.Resize(trans_x.dims());
        dx_temp.mutable_data<T>(ctx.GetPlace());
        const auto& runner_dx =
            NpuOpRunner("Mul", {*dout, trans_y}, {dx_temp}, {});
        runner_dx.Run(stream);
        ReduceDims<T>(
            ctx, stream, axis, dx->dims(), trans_x.dims(), dx_temp, dx);
      }
    }
    if (dy) {
      if (dy->dims() == dout->dims()) {
        dy->mutable_data<T>(ctx.GetPlace());
        const auto& runner_dy = NpuOpRunner("Mul", {trans_x, *dout}, {*dy}, {});
        runner_dy.Run(stream);
      } else {
        phi::DenseTensor dy_temp(y->type());
        dy_temp.Resize(trans_y.dims());
        dy_temp.mutable_data<T>(ctx.GetPlace());
        const auto& runner_dy =
            NpuOpRunner("Mul", {trans_x, *dout}, {dy_temp}, {});
        runner_dy.Run(stream);
        ReduceDims<T>(
            ctx, stream, axis, dy->dims(), trans_y.dims(), dy_temp, dy);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(elementwise_mul,
                       ops::ElementwiseMulNPUKernel<float>,
                       ops::ElementwiseMulNPUKernel<paddle::platform::float16>,
#ifdef PADDLE_WITH_ASCEND_INT64
                       ops::ElementwiseMulNPUKernel<int64_t>,
#endif
                       ops::ElementwiseMulNPUKernel<int>);

REGISTER_OP_NPU_KERNEL(
    elementwise_mul_grad,
    ops::ElementwiseMulGradNPUKernel<float>,
    ops::ElementwiseMulGradNPUKernel<paddle::platform::float16>,
#ifdef PADDLE_WITH_ASCEND_INT64
    ops::ElementwiseMulGradNPUKernel<int64_t>,
#endif
    ops::ElementwiseMulGradNPUKernel<int>);
