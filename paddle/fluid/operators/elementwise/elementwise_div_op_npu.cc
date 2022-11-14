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

#include "paddle/fluid/operators/elementwise/elementwise_div_op.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;

template <typename DeviceContext, typename T>
class ElementwiseDivNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* y = ctx.Input<phi::DenseTensor>("Y");

    auto* out = ctx.Output<phi::DenseTensor>("Out");

    auto place = ctx.GetPlace();

    out->mutable_data<T>(place);

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    const auto& runner = NpuOpRunner("Div", {*x, *y}, {*out}, {});
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class ElementwiseDivGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* out = ctx.Input<phi::DenseTensor>("Out");
    auto* dout = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* y = ctx.Input<phi::DenseTensor>("Y");

    auto* dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<phi::DenseTensor>(framework::GradVarName("Y"));

    auto place = ctx.GetPlace();

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    if (dx) {
      dx->mutable_data<T>(place);

      Tensor tensor_one(y->type());
      tensor_one.mutable_data<float>({1}, place);
      FillNpuTensorWithConstant<float>(&tensor_one, static_cast<float>(1.0));

      // Use `Div` CANN OP to achieve `1/y` instead of `Power` CANN OP.
      // Because `Power` will cause precision overflow, that is, `float_status`
      // will be set to 1.
      Tensor y_div(y->type());
      y_div.mutable_data<T>(y->dims(), place);
      const auto& runner_one_div_y =
          NpuOpRunner("Div", {tensor_one, *y}, {y_div}, {});
      runner_one_div_y.Run(stream);

      Tensor tensor_zeros(x->type());
      tensor_zeros.mutable_data<T>(x->dims(), place);
      const auto& runner_tensor_zeros =
          NpuOpRunner("ZerosLike", {*x}, {tensor_zeros}, {});
      runner_tensor_zeros.Run(stream);

      Tensor x_zero(experimental::DataType::BOOL);
      x_zero.mutable_data<bool>(x->dims(), place);
      const auto& runner_x_zero =
          NpuOpRunner("Equal", {*x, tensor_zeros}, {x_zero}, {});
      runner_x_zero.Run(stream);

      Tensor x_nozero(experimental::DataType::BOOL);
      x_nozero.mutable_data<bool>(x->dims(), place);
      const auto& runner_x_nonzero =
          NpuOpRunner("LogicalNot", {x_zero}, {x_nozero}, {});
      runner_x_nonzero.Run(stream);

      Tensor x_nozero_f(x->type());
      x_nozero_f.mutable_data<T>(x->dims(), place);
      const auto& runner_x_nonzero_f =
          NpuOpRunner("Cast",
                      {x_nozero},
                      {x_nozero_f},
                      {{"dst_type", static_cast<int32_t>(0)}});
      runner_x_nonzero_f.Run(stream);

      Tensor x_grad_w(x->type());
      x_grad_w.mutable_data<T>(x->dims(), place);
      const auto& runner_x_grad_w =
          NpuOpRunner("Mul", {x_nozero_f, y_div}, {x_grad_w}, {});
      runner_x_grad_w.Run(stream);

      const auto& runner_x_grad =
          NpuOpRunner("Mul", {x_grad_w, *dout}, {*dx}, {});
      runner_x_grad.Run(stream);
    }

    if (dy) {
      dy->mutable_data<T>(place);

      Tensor neg_out(out->type());
      neg_out.mutable_data<T>(out->dims(), place);
      const auto& runner_neg_out = NpuOpRunner("Neg", {*out}, {neg_out}, {});
      runner_neg_out.Run(stream);

      Tensor tmp_mul(out->type());
      tmp_mul.mutable_data<T>(out->dims(), place);
      const auto& runner_mul =
          NpuOpRunner("Mul", {neg_out, *dout}, {tmp_mul}, {});
      runner_mul.Run(stream);

      if (dy->dims() != dout->dims()) {
        Tensor reduced_tmp_mul(y->type());
        reduced_tmp_mul.mutable_data<T>(y->dims(), place);

        std::vector<int64_t> axes;
        int64_t diff = dout->dims().size() - dy->dims().size();
        for (int64_t i = 0; i < dout->dims().size(); ++i) {
          if (i < diff) {
            axes.push_back(i);
            continue;
          }
          if (dout->dims()[i] > dy->dims()[i - diff]) {
            axes.push_back(i);
          }
        }
        const auto& runner_reduce =
            NpuOpRunner("ReduceSumD",
                        {tmp_mul},
                        {reduced_tmp_mul},
                        {{"axes", axes}, {"keep_dims", false}});
        runner_reduce.Run(stream);

        const auto& runner_y_grad =
            NpuOpRunner("Div", {reduced_tmp_mul, *y}, {*dy}, {});
        runner_y_grad.Run(stream);
      } else {
        const auto& runner_y_grad =
            NpuOpRunner("Div", {tmp_mul, *y}, {*dy}, {});
        runner_y_grad.Run(stream);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    elementwise_div,
    ops::ElementwiseDivNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::ElementwiseDivNPUKernel<paddle::platform::NPUDeviceContext,
                                 paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(
    elementwise_div_grad,
    ops::ElementwiseDivGradNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::ElementwiseDivGradNPUKernel<paddle::platform::NPUDeviceContext,
                                     paddle::platform::float16>);
