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

#include "paddle/fluid/operators/elementwise/elementwise_npu.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;

template <typename DeviceContext, typename T>
class ElementwiseMaxNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx =
        ctx.template device_context<paddle::platform::NPUDeviceContext>();

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
      direct_compute = y_dims == phi::slice_ddim(x_dims, axis, x_dims.size());
    } else {
      direct_compute = x_dims == phi::slice_ddim(y_dims, axis, y_dims.size());
    }

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    if (direct_compute) {
      const auto& runner = NpuOpRunner("Maximum", {*x, *y}, {*out}, {});
      runner.Run(stream);
    } else {
      Tensor transformed_x, transformed_y;
      NpuElementWiseOpBroadcast<T>(
          dev_ctx, x, y, axis, &transformed_x, &transformed_y);
      const auto& runner =
          NpuOpRunner("Maximum", {transformed_x, transformed_y}, {*out}, {});
      runner.Run(stream);
    }
  }
};

template <typename DeviceContext, typename T>
class ElementwiseMaxGradNPUKernel : public framework::OpKernel<T> {
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

    // The ascend elementwise_max_grad op only supports broadcast
    // when axis is -1, and requires all the inputs must have the
    // same shape when axis is not -1. For convenience, we should
    // broadcast the original input x and y to transformed_x and
    // transformed_x firstly, then use tmp tensor to get the op
    // output, last reduce the tmp tensor shape to match the
    // paddle output.

    auto x_dims = x->dims();
    auto y_dims = y->dims();
    axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
    Tensor transformed_x, transformed_y;
    NpuElementWiseOpBroadcast<T>(
        dev_ctx, x, y, axis, &transformed_x, &transformed_y);

    auto dout_dims = dout->dims();
    auto stream = dev_ctx.stream();
    framework::NPUAttributeMap attr_input = {{"grad_x", true},
                                             {"grad_y", true}};
    // Reshape info vector.
    std::vector<int> reduce_axes;

    if (dx && dy) {
      dx->mutable_data<T>(ctx.GetPlace());
      dy->mutable_data<T>(ctx.GetPlace());
      Tensor tmp_dx;
      tmp_dx.mutable_data<T>(dout_dims, ctx.GetPlace());
      Tensor tmp_dy;
      tmp_dy.mutable_data<T>(dout_dims, ctx.GetPlace());

      const auto& runner = NpuOpRunner("MaximumGrad",
                                       {*dout, transformed_x, transformed_y},
                                       {tmp_dx, tmp_dy},
                                       attr_input);
      runner.Run(stream);

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
        framework::TensorCopy(tmp_dx, ctx.GetPlace(), dev_ctx, dx);
      }

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
        framework::TensorCopy(tmp_dy, ctx.GetPlace(), dev_ctx, dy);
      }

    } else if (dx) {
      Tensor zero_tensor(dout->type());
      zero_tensor.mutable_data<T>(dout_dims, ctx.GetPlace());
      FillNpuTensorWithConstant<T>(&zero_tensor, static_cast<T>(0));

      dx->mutable_data<T>(ctx.GetPlace());
      Tensor tmp_dx;
      tmp_dx.mutable_data<T>(dout_dims, ctx.GetPlace());

      const auto& runner = NpuOpRunner("MaximumGrad",
                                       {*dout, transformed_x, transformed_y},
                                       {tmp_dx, zero_tensor},
                                       attr_input);
      runner.Run(stream);

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
        framework::TensorCopy(tmp_dx, ctx.GetPlace(), dev_ctx, dx);
      }

    } else if (dy) {
      Tensor zero_tensor(dout->type());
      zero_tensor.mutable_data<T>(dout_dims, ctx.GetPlace());
      FillNpuTensorWithConstant<T>(&zero_tensor, static_cast<T>(0));

      dy->mutable_data<T>(ctx.GetPlace());
      Tensor tmp_dy;
      tmp_dy.mutable_data<T>(dout_dims, ctx.GetPlace());

      const auto& runner = NpuOpRunner("MaximumGrad",
                                       {*dout, transformed_x, transformed_y},
                                       {zero_tensor, tmp_dy},
                                       attr_input);
      runner.Run(stream);

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
        framework::TensorCopy(tmp_dy, ctx.GetPlace(), dev_ctx, dy);
      }
    } else {
      PADDLE_THROW(platform::errors::Unavailable(
          "Do not support all outputs to be empty."));
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(
    elementwise_max,
    ops::ElementwiseMaxNPUKernel<plat::NPUDeviceContext, plat::float16>,
    ops::ElementwiseMaxNPUKernel<plat::NPUDeviceContext, float>,
    ops::ElementwiseMaxNPUKernel<plat::NPUDeviceContext, double>,
    ops::ElementwiseMaxNPUKernel<plat::NPUDeviceContext, int>,
    ops::ElementwiseMaxNPUKernel<plat::NPUDeviceContext, int64_t>);

REGISTER_OP_NPU_KERNEL(
    elementwise_max_grad,
    ops::ElementwiseMaxGradNPUKernel<plat::NPUDeviceContext, plat::float16>,
    ops::ElementwiseMaxGradNPUKernel<plat::NPUDeviceContext, float>,
    ops::ElementwiseMaxGradNPUKernel<plat::NPUDeviceContext, double>,
    ops::ElementwiseMaxGradNPUKernel<plat::NPUDeviceContext, int>);
