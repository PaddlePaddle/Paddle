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
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class ElementwiseMulNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx =
        ctx.template device_context<paddle::platform::NPUDeviceContext>();
    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");

    auto* out = ctx.Output<Tensor>("Out");
    int axis = ctx.Attr<int>("axis");

    bool direct_compute = false;
    auto x_dims = x->dims();
    auto y_dims = y->dims();
    axis =
        (axis < 0 ? std::abs(x_dims.size() - y_dims.size()) + axis + 1 : axis);
    if (x_dims.size() >= y_dims.size()) {
      direct_compute =
          y_dims == framework::slice_ddim(x_dims, axis, x_dims.size());
    } else {
      direct_compute =
          x_dims == framework::slice_ddim(y_dims, axis, y_dims.size());
    }

    auto place = ctx.GetPlace();

    out->mutable_data<T>(place);

    auto stream = dev_ctx.stream();

    if (direct_compute) {
      const auto& runner = NpuOpRunner("Mul", {*x, *y}, {*out}, {});
      runner.Run(stream);
    } else {
      Tensor transformed_x, transformed_y;
      NpuElementWiseOpBroadcast<T>(dev_ctx, x, y, axis, &transformed_x,
                                   &transformed_y);
      const auto& runner =
          NpuOpRunner("Mul", {transformed_x, transformed_y}, {*out}, {});
      runner.Run(stream);
    }
  }
};

template <typename DeviceContext, typename T>
class ElementwiseMulGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx =
        ctx.template device_context<paddle::platform::NPUDeviceContext>();
    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    int axis = ctx.Attr<int>("axis");

    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));

    auto place = ctx.GetPlace();
    auto stream = dev_ctx.stream();

    auto x_dims = x->dims();
    auto y_dims = y->dims();
    axis =
        (axis < 0 ? std::abs(x_dims.size() - y_dims.size()) + axis + 1 : axis);
    Tensor transformed_x, transformed_y;
    NpuElementWiseOpBroadcast<T>(dev_ctx, x, y, axis, &transformed_x,
                                 &transformed_y);
    auto dout_dims = dout->dims();
    if (dx) {
      dx->mutable_data<T>(place);

      Tensor tmp_dx(dx->type());
      tmp_dx.mutable_data<T>(dout_dims, place);

      const auto& runner_dx =
          NpuOpRunner("Mul", {*dout, transformed_y}, {tmp_dx}, {});
      runner_dx.Run(stream);

      if (x_dims != dout_dims) {
        std::vector<int> reduce_axes;
        int src_axis = (x_dims.size() < dout_dims.size() ? axis : 0);
        for (int ax = 0; ax < dout_dims.size(); ++ax) {
          if ((ax < src_axis || ax >= src_axis + x_dims.size()) ||
              (dout_dims[ax] > 1 && x_dims[ax - src_axis] == 1)) {
            reduce_axes.push_back(ax);
          }
        }
        if (!reduce_axes.empty()) {
          const auto& runner =
              NpuOpRunner("ReduceSumD", {tmp_dx}, {*dx},
                          {{"axes", reduce_axes}, {"keep_dims", false}});
          runner.Run(stream);
        }
      } else {
        framework::TensorCopy(tmp_dx, place, dev_ctx, dx);
      }
    }

    if (dy) {
      dy->mutable_data<T>(place);

      Tensor tmp_dy(dy->type());
      tmp_dy.mutable_data<T>(dout_dims, place);

      const auto& runner_dy =
          NpuOpRunner("Mul", {*dout, transformed_x}, {tmp_dy}, {});
      runner_dy.Run(stream);

      if (y_dims != dout_dims) {
        std::vector<int> reduce_axes;
        int src_axis = (y_dims.size() < dout_dims.size() ? axis : 0);
        for (int ax = 0; ax < dout_dims.size(); ++ax) {
          if ((ax < src_axis || ax >= src_axis + y_dims.size()) ||
              (dout_dims[ax] > 1 && y_dims[ax - src_axis] == 1)) {
            reduce_axes.push_back(ax);
          }
        }
        if (!reduce_axes.empty()) {
          const auto& runner =
              NpuOpRunner("ReduceSumD", {tmp_dy}, {*dy},
                          {{"axes", reduce_axes}, {"keep_dims", false}});
          runner.Run(stream);
        }
      } else {
        framework::TensorCopy(tmp_dy, place, dev_ctx, dy);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    elementwise_mul,
    ops::ElementwiseMulNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::ElementwiseMulNPUKernel<paddle::platform::NPUDeviceContext,
                                 paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(
    elementwise_mul_grad,
    ops::ElementwiseMulGradNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::ElementwiseMulGradNPUKernel<paddle::platform::NPUDeviceContext,
                                     paddle::platform::float16>);
