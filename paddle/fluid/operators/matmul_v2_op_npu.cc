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

#include "paddle/fluid/operators/matmul_v2_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class MatMulV2NPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::Tensor>("X");
    auto* y = ctx.Input<framework::Tensor>("Y");
    auto* out = ctx.Output<framework::Tensor>("Out");
    bool transpose_x = ctx.Attr<bool>("trans_x");
    bool transpose_y = ctx.Attr<bool>("trans_y");

    /*auto stream =
          ctx.template device_context<paddle::platform::NPUDeviceContext>()
              .stream();

    Tensor tmp_x(x->type());
    tmp_x.Resize(x->dims());
    tmp_x.mutable_data<T>(ctx.GetPlace());
    tmp_x.set_layout(DataLayout::kFractalNZ);
    // framework::TensorCopy(
    //       *x, ctx.GetPlace(),
    //       ctx.template device_context<platform::DeviceContext>(), &tmp_x);
    if (x->layout() != tmp_x.layout()) {
      auto runner_cast_x = NpuOpRunner(
          "TransData", {*x}, {tmp_x},
          {{"src_format", framework::DataLayoutToString(x->layout())},
    {"dst_format", framework::DataLayoutToString(tmp_x.layout())}, {"groups",
    1}});
      runner_cast_x.Run(stream);
    } else {
     tmp_x.ShareDataWith(*x);
    }

    Tensor tmp_y(y->type());
    tmp_y.Resize(y->dims());
    tmp_y.mutable_data<T>(ctx.GetPlace());
    tmp_y.set_layout(DataLayout::kFractalNZ);
    // framework::TensorCopy(
    //       *y, ctx.GetPlace(),
    //       ctx.template device_context<platform::DeviceContext>(), &tmp_y);
    if (y->layout() != tmp_y.layout()) {
      auto runner_cast_y = NpuOpRunner(
          "TransData", {*y}, {tmp_y},
          {{"src_format", framework::DataLayoutToString(y->layout())},
    {"dst_format", framework::DataLayoutToString(tmp_y.layout())}, {"groups",
    1}});
      runner_cast_y.Run(stream);
    } else {
     tmp_y.ShareDataWith(*y);
    }*/

    Tensor tmp_x(src_tensor.type());
    tmp_x.Resize(src_tensor->dims());
    tmp_x.mutable_data<T>(ctx.GetPlace());
    framework::TensorCopy(
        src_tensor, ctx.GetPlace(),
        ctx.template device_context<platform::DeviceContext>(), &tmp_x);
    tmp_x = CastNPUFormat(*x, 29);

    Tensor tmp_y(src_tensor.type());
    tmp_y.Resize(src_tensor->dims());
    tmp_y.mutable_data<T>(ctx.GetPlace());
    framework::TensorCopy(
        src_tensor, ctx.GetPlace(),
        ctx.template device_context<platform::DeviceContext>(), &tmp_y);
    tmp_y = CastNPUFormat(*y, 29);

    if (x->dims().size() == 2) {
      out->mutable_data<T>(ctx.GetPlace());
      out->set_layout(DataLayout::kFractalNZ);

      const auto& runner = NpuOpRunner(
          "MatMul", {tmp_x, tmp_y}, {*out},
          {{"transpose_x1", transpose_x}, {"transpose_x2", transpose_y}});

      auto stream =
          ctx.template device_context<paddle::platform::NPUDeviceContext>()
              .stream();
      runner.Run(stream);

    } else if (x->dims().size() > 2) {
      out->mutable_data<T>(ctx.GetPlace());

      const auto& runner =
          NpuOpRunner("BatchMatMul", {tmp_x, tmp_y}, {*out},
                      {{"adj_x1", transpose_x}, {"adj_x2", transpose_y}});

      auto stream =
          ctx.template device_context<paddle::platform::NPUDeviceContext>()
              .stream();
      runner.Run(stream);
    }
  }
};

template <typename DeviceContext, typename T>
class MatMulV2GradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::Tensor>("X");
    auto* y = ctx.Input<framework::Tensor>("Y");
    auto* dout = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<framework::Tensor>(framework::GradVarName("Y"));
    bool transpose_y = ctx.Attr<bool>("trans_y");
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    if (x->dims().size() == 2) {
      if (transpose_y) {
        if (dx) {
          dx->mutable_data<T>(ctx.GetPlace());
          const auto& runner_dx =
              NpuOpRunner("MatMul", {*dout, *y}, {*dx},
                          {{"transpose_x1", false}, {"transpose_x2", false}});

          runner_dx.Run(stream);
        }
        if (dy) {
          dy->mutable_data<T>(ctx.GetPlace());
          const auto& runner_dy =
              NpuOpRunner("MatMul", {*dout, *x}, {*dy},
                          {{"transpose_x1", true}, {"transpose_x2", false}});

          runner_dy.Run(stream);
        }

      } else {
        if (dx) {
          dx->mutable_data<T>(ctx.GetPlace());
          const auto& runner_dx =
              NpuOpRunner("MatMul", {*dout, *y}, {*dx},
                          {{"transpose_x1", false}, {"transpose_x2", true}});

          runner_dx.Run(stream);
        }
        if (dy) {
          dy->mutable_data<T>(ctx.GetPlace());
          const auto& runner_dy =
              NpuOpRunner("MatMul", {*x, *dout}, {*dy},
                          {{"transpose_x1", true}, {"transpose_x2", false}});

          runner_dy.Run(stream);
        }
      }
    } else if (x->dims().size() > 2) {
      if (transpose_y) {
        if (dx) {
          dx->mutable_data<T>(ctx.GetPlace());
          const auto& runner_dx =
              NpuOpRunner("BatchMatMul", {*dout, *y}, {*dx},
                          {{"adj_x1", false}, {"adj_x2", false}});

          runner_dx.Run(stream);
        }
        if (dy) {
          dy->mutable_data<T>(ctx.GetPlace());
          const auto& runner_dy =
              NpuOpRunner("BatchMatMul", {*dout, *x}, {*dy},
                          {{"adj_x1", true}, {"adj_x2", false}});

          runner_dy.Run(stream);
        }
      } else {
        if (dx) {
          dx->mutable_data<T>(ctx.GetPlace());
          const auto& runner_dx =
              NpuOpRunner("BatchMatMul", {*dout, *y}, {*dx},
                          {{"adj_x1", false}, {"adj_x2", true}});

          runner_dx.Run(stream);
        }
        if (dy) {
          dy->mutable_data<T>(ctx.GetPlace());
          const auto& runner_dy =
              NpuOpRunner("BatchMatMul", {*x, *dout}, {*dy},
                          {{"adj_x1", true}, {"adj_x2", false}});
          runner_dy.Run(stream);
        }
      }
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    matmul_v2,
    ops::MatMulV2NPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::MatMulV2NPUKernel<paddle::platform::NPUDeviceContext,
                           paddle::platform::float16>);
REGISTER_OP_NPU_KERNEL(
    matmul_v2_grad,
    ops::MatMulV2GradNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::MatMulV2GradNPUKernel<paddle::platform::NPUDeviceContext,
                               paddle::platform::float16>);
