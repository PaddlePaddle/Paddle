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

#include <memory>
#include <string>

#include "paddle/fluid/operators/elementwise/elementwise_npu.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;

template <typename DeviceContext, typename T>
class ElementwisePowNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx =
        ctx.template device_context<paddle::platform::NPUDeviceContext>();

    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* y = ctx.Input<phi::DenseTensor>("Y");
    auto* out = ctx.Output<phi::DenseTensor>("Out");

    auto place = ctx.GetPlace();
    int axis = ctx.Attr<int>("axis");

    out->mutable_data<T>(place);

    bool direct_compute = false;
    auto x_dims = x->dims();
    auto y_dims = y->dims();
    axis =
        (axis < 0 ? std::abs(x_dims.size() - y_dims.size()) + axis + 1 : axis);
    if (x_dims.size() >= y_dims.size()) {
      direct_compute = y_dims == phi::slice_ddim(x_dims, axis, x_dims.size());
    } else {
      direct_compute = x_dims == phi::slice_ddim(y_dims, axis, y_dims.size());
    }

    auto stream = dev_ctx.stream();

    if (direct_compute) {
      const auto& runner = NpuOpRunner("Pow", {*x, *y}, {*out}, {});
      runner.Run(stream);
    } else {
      Tensor transformed_x, transformed_y;
      NpuElementWiseOpBroadcast<T>(
          dev_ctx, x, y, axis, &transformed_x, &transformed_y);
      const auto& runner =
          NpuOpRunner("Pow", {transformed_x, transformed_y}, {*out}, {});
      runner.Run(stream);
    }
  }
};

template <typename DeviceContext, typename T>
class ElementwisePowGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx =
        ctx.template device_context<paddle::platform::NPUDeviceContext>();
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* y = ctx.Input<phi::DenseTensor>("Y");
    auto* dout = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<phi::DenseTensor>(framework::GradVarName("Y"));
    int axis = ctx.Attr<int>("axis");
    auto place = ctx.GetPlace();

    auto x_dims = x->dims();
    auto y_dims = y->dims();
    axis =
        (axis < 0 ? std::abs(x_dims.size() - y_dims.size()) + axis + 1 : axis);
    Tensor transformed_x, transformed_y;
    NpuElementWiseOpBroadcast<T>(
        dev_ctx, x, y, axis, &transformed_x, &transformed_y);

    auto dout_dims = dout->dims();
    auto stream = dev_ctx.stream();
    // Reshape info vector.
    std::vector<int> reduce_axes;
    if (dx) {
      Tensor zero_tensor(dout->type());
      zero_tensor.mutable_data<T>(dout_dims, place);
      FillNpuTensorWithConstant<T>(&zero_tensor, static_cast<T>(0));

      dx->mutable_data<T>(place);
      Tensor tmp_dx;
      tmp_dx.mutable_data<T>(dout_dims, place);

      // dx = dout * y * pow(x, y - 1);
      Tensor PowGrad_dx_temp1(dout->type());
      PowGrad_dx_temp1.mutable_data<T>(dout->dims(), place);
      const auto& runner_PowGrad_dx_temp1 =
          NpuOpRunner("Mul", {*dout, transformed_y}, {PowGrad_dx_temp1}, {});
      runner_PowGrad_dx_temp1.Run(stream);

      Tensor one_dx(transformed_y.type());
      one_dx.mutable_data<T>(transformed_y.dims(), place);
      const auto& runner_one_dx =
          NpuOpRunner("OnesLike", {transformed_y}, {one_dx}, {});
      runner_one_dx.Run(stream);

      Tensor sub_dx(transformed_y.type());
      sub_dx.mutable_data<T>(transformed_y.dims(), place);
      const auto& runner_sub_dx =
          NpuOpRunner("Sub", {transformed_y, one_dx}, {sub_dx}, {});
      runner_sub_dx.Run(stream);

      Tensor PowGrad_dx_temp2(transformed_x.type());
      PowGrad_dx_temp2.mutable_data<T>(transformed_x.dims(), place);
      const auto& runner_PowGrad_dx_temp2 =
          NpuOpRunner("Pow", {transformed_x, sub_dx}, {PowGrad_dx_temp2}, {});
      runner_PowGrad_dx_temp2.Run(stream);

      const auto& runner_dx = NpuOpRunner(
          "Mul", {PowGrad_dx_temp1, PowGrad_dx_temp2}, {tmp_dx}, {});
      runner_dx.Run(stream);

      if (x_dims != dout_dims) {
        reduce_axes.clear();

        int src_axis = (x_dims.size() < dout_dims.size() ? axis : 0);
        for (int ax = 0; ax < dout_dims.size(); ++ax) {
          if ((ax < src_axis || ax >= src_axis + x_dims.size()) ||
              (dout_dims[ax] > 1 && x_dims[ax - src_axis] == 1)) {
            reduce_axes.push_back(ax);
          }
        }
        if (!reduce_axes.empty()) {
          const auto& runner =
              NpuOpRunner("ReduceSumD",
                          {tmp_dx},
                          {*dx},
                          {{"axes", reduce_axes}, {"keep_dims", false}});
          runner.Run(stream);
        }
      } else {
        framework::TensorCopy(tmp_dx, place, dev_ctx, dx);
      }
    }
    if (dy) {
      Tensor zero_tensor(dout->type());
      zero_tensor.mutable_data<T>(dout_dims, place);
      FillNpuTensorWithConstant<T>(&zero_tensor, static_cast<T>(0));

      dy->mutable_data<T>(place);
      Tensor tmp_dy;
      tmp_dy.mutable_data<T>(dout_dims, place);

      // dy = dout * log(x) * pow(x, y)
      Tensor PowGrad_dy_temp1(transformed_x.type());
      PowGrad_dy_temp1.mutable_data<T>(transformed_x.dims(), place);
      const auto& runner_PowGrad_dy_temp1 = NpuOpRunner(
          "Pow", {transformed_x, transformed_y}, {PowGrad_dy_temp1}, {});
      runner_PowGrad_dy_temp1.Run(stream);

      Tensor one_dy(transformed_x.type());
      one_dy.mutable_data<T>(transformed_x.dims(), place);
      const auto& runner_one_dy =
          NpuOpRunner("OnesLike", {transformed_x}, {one_dy}, {});
      runner_one_dy.Run(stream);

      Tensor sub_dy(transformed_x.type());
      sub_dy.mutable_data<T>(transformed_x.dims(), place);
      const auto& runner_sub_dy =
          NpuOpRunner("Sub", {transformed_x, one_dy}, {sub_dy}, {});
      runner_sub_dy.Run(stream);

      Tensor log_dy(transformed_x.type());
      log_dy.mutable_data<T>(transformed_x.dims(), place);
      const auto& runner_log_dy = NpuOpRunner("Log1p", {sub_dy}, {log_dy}, {});
      runner_log_dy.Run(stream);

      Tensor PowGrad_dy_temp2(transformed_x.type());
      PowGrad_dy_temp2.mutable_data<T>(transformed_x.dims(), place);
      const auto& runner_PowGrad_dy_temp2 = NpuOpRunner(
          "Mul", {log_dy, PowGrad_dy_temp1}, {PowGrad_dy_temp2}, {});
      runner_PowGrad_dy_temp2.Run(stream);

      const auto& runner_dy =
          NpuOpRunner("Mul", {*dout, PowGrad_dy_temp2}, {tmp_dy}, {});
      runner_dy.Run(stream);

      if (y_dims != dout_dims) {
        reduce_axes.clear();

        int src_axis = (y_dims.size() < dout_dims.size() ? axis : 0);
        for (int ax = 0; ax < dout_dims.size(); ++ax) {
          if ((ax < src_axis || ax >= src_axis + y_dims.size()) ||
              (dout_dims[ax] > 1 && y_dims[ax - src_axis] == 1)) {
            reduce_axes.push_back(ax);
          }
        }
        if (!reduce_axes.empty()) {
          const auto& runner =
              NpuOpRunner("ReduceSumD",
                          {tmp_dy},
                          {*dy},
                          {{"axes", reduce_axes}, {"keep_dims", false}});
          runner.Run(stream);
        }
      } else {
        framework::TensorCopy(tmp_dy, place, dev_ctx, dy);
      }
    }
    if (!dx && !dy) {
      PADDLE_THROW(platform::errors::Unavailable(
          "Not support all outputs to be empty."));
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(
    elementwise_pow,
    ops::ElementwisePowNPUKernel<plat::NPUDeviceContext, plat::float16>,
    ops::ElementwisePowNPUKernel<plat::NPUDeviceContext, float>,
    ops::ElementwisePowNPUKernel<plat::NPUDeviceContext, double>,
    ops::ElementwisePowNPUKernel<plat::NPUDeviceContext, int>);

REGISTER_OP_NPU_KERNEL(
    elementwise_pow_grad,
    ops::ElementwisePowGradNPUKernel<plat::NPUDeviceContext, plat::float16>,
    ops::ElementwisePowGradNPUKernel<plat::NPUDeviceContext, float>,
    ops::ElementwisePowGradNPUKernel<plat::NPUDeviceContext, double>,
    ops::ElementwisePowGradNPUKernel<plat::NPUDeviceContext, int>);
