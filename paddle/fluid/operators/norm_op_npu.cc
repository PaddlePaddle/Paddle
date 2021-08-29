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

#include "paddle/fluid/operators/norm_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

using DDim = framework::DDim;
using Tensor = framework::Tensor;

// @TODO
// This function may be no correct !!!
inline void GetRemainDims(const framework::DDim &dim, int axis,
                          std::vector<int> *remainVec) {
  if (axis < 0) axis = dim.size() + axis;

  for (int i = 0; i < axis; ++i) {
    remainVec->push_back(dim[i]);
  }

  remainVec->push_back(1);

  for (int i = axis + 1; i < dim.size(); ++i) {
    remainVec->push_back(dim[i]);
  }
}

template <typename DeviceContext, typename T>
class NormNPUKernel : public framework::OpKernel<T> {
 private:
  void CheckAxis(int axis, int rank) const {
    // check the axis is in [-rank, rank-1]
    if (axis <= rank - 1 && axis >= -rank) return;
    PADDLE_THROW(platform::errors::InvalidArgument(
        "axis in norm operator must between (%d) and (%d)"
        "but got (%d).",
        -rank, rank - 1, axis));
  }

 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    VLOG(4) << "Launch Norm Op Kernel on NPU." << std::endl;
    auto *in_x = ctx.Input<framework::Tensor>("X");
    auto *out_y = ctx.Output<framework::Tensor>("Out");
    auto *out_norm = ctx.Output<framework::Tensor>("Norm");
    out_y->mutable_data<T>(ctx.GetPlace());
    out_norm->mutable_data<T>(ctx.GetPlace());
    auto xdim = in_x->dims();
    float eps = ctx.Attr<float>("epsilon");
    int axis = ctx.Attr<int>("axis");
    CheckAxis(axis, xdim.size());
    if (axis < 0) axis = xdim.size() + axis;

    framework::NPUAttributeMap attr_input_norm;
    attr_input_norm["axes"] = std::vector<int>({axis});
    attr_input_norm["p"] = 2;
    attr_input_norm["keepdim"] = true;
    attr_input_norm["epsilon"] = eps;
    const auto &runner =
        NpuOpRunner("LpNorm", {*in_x}, {*out_norm}, attr_input_norm);
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
    NpuOpRunner("Div", {*in_x, *out_norm}, {*out_y}, {}).Run(stream);
  }
};

template <typename DeviceContext, typename T>
class NormGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    LOG(WARNING) << "NormGradNPUKernel";
    LOG(WARNING) << "op type: " << ctx.Type();

    float epsilon = ctx.Attr<float>("epsilon");
    int axis = ctx.Attr<int>("axis");

    LOG(WARNING) << "epsilon: " << epsilon;
    LOG(WARNING) << "axis: " << axis;

    auto *x = ctx.Input<Tensor>("X");
    auto *norm = ctx.Input<framework::Tensor>("Norm");
    auto *dy = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto *dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));

    auto place = ctx.GetPlace();

    LOG(WARNING) << "x dims: " << x->dims();
    LOG(WARNING) << "x numel: " << x->numel();
    LOG(WARNING) << "norm dims: " << norm->dims();
    LOG(WARNING) << "norm numel: " << norm->numel();
    LOG(WARNING) << "dy dims: " << dy->dims();
    LOG(WARNING) << "dy numel: " << dy->numel();
    LOG(WARNING) << "dx dims: " << dx->dims();
    LOG(WARNING) << "dx numel: " << dx->numel();

    // dx = ( dy/sqrt(sum(x*x)) ) * [1 - x*sum(x) / (sum(x*x) + e)]
    //    = [dy - dy * x * sum(x) / (sum(x*x) + e)] / sqrt(sum(x*x))
    //    = [dy - x * sum(x*dy) / (sum(x*x) + e)] / sqrt(sum(x*x))

    std::vector<int> remainVec;
    GetRemainDims(x->dims(), axis, &remainVec);
    auto remainDim = framework::make_ddim(remainVec);
    // framework::DDim remainDim = {2, 1, 4, 5};
    LOG(WARNING) << "remainDim: " << remainDim;

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    std::vector<int> axes = {axis};

    Tensor x_mul_dy;
    {
      x_mul_dy.Resize(x->dims());
      x_mul_dy.mutable_data<float>(place);

      const auto &runner = NpuOpRunner("Mul", {*x, *dy}, {x_mul_dy}, {});
      runner.Run(stream);

      LOG(WARNING) << "x_mul_dy: ";
      PrintTensor<float>(x_mul_dy, ctx);
    }

    Tensor x_mul_dy_sum;
    {
      framework::NPUAttributeMap attr_input = {{"keep_dims", true},
                                               {"axes", axes}};

      x_mul_dy_sum.Resize(remainDim);
      x_mul_dy_sum.mutable_data<float>(place);

      const auto &runner =
          NpuOpRunner("ReduceSumD", {x_mul_dy}, {x_mul_dy_sum}, attr_input);
      runner.Run(stream);

      LOG(WARNING) << "x_mul_dy_sum: ";
      PrintTensor<float>(x_mul_dy_sum, ctx);
    }

    Tensor x_mul_dy_sum_broadcast;
    {
      framework::NPUAttributeMap attr_input = {
          {"shape", framework::vectorize(x->dims())}};

      x_mul_dy_sum_broadcast.Resize(x->dims());
      x_mul_dy_sum_broadcast.mutable_data<float>(place);

      const auto &runner = NpuOpRunner("BroadcastToD", {x_mul_dy_sum},
                                       {x_mul_dy_sum_broadcast}, attr_input);
      runner.Run(stream);

      LOG(WARNING) << "x_mul_dy_sum_broadcast: ";
      PrintTensor<float>(x_mul_dy_sum_broadcast, ctx);
    }

    Tensor x_square;
    {
      x_square.Resize(x->dims());
      x_square.mutable_data<T>(place);

      const auto &runner = NpuOpRunner("Square", {*x}, {x_square}, {});
      runner.Run(stream);

      LOG(WARNING) << "x_square: ";
      PrintTensor<float>(x_square, ctx);
    }

    Tensor x_square_sum;
    {
      framework::NPUAttributeMap attr_input = {{"keep_dims", true},
                                               {"axes", axes}};

      x_square_sum.Resize(remainDim);
      x_square_sum.mutable_data<T>(place);

      const auto &runner =
          NpuOpRunner("ReduceSumD", {x_square}, {x_square_sum}, attr_input);
      runner.Run(stream);

      LOG(WARNING) << "x_square_sum: ";
      PrintTensor<float>(x_square_sum, ctx);
    }

    Tensor x_square_sum_broadcast;
    {
      framework::NPUAttributeMap attr_input = {
          {"shape", framework::vectorize(x->dims())}};

      x_square_sum_broadcast.Resize(x->dims());
      x_square_sum_broadcast.mutable_data<T>(place);

      const auto &runner = NpuOpRunner("BroadcastToD", {x_square_sum},
                                       {x_square_sum_broadcast}, attr_input);
      runner.Run(stream);

      LOG(WARNING) << "x_square_sum_broadcast: ";
      PrintTensor<float>(x_square_sum_broadcast, ctx);
    }

    // @TODO broadcast
    // x_square_sum -> x_square_sum_broadcast
    Tensor x_square_sum_plus_epsilon;
    {
      framework::NPUAttributeMap attr_input = {{"value", epsilon}};

      x_square_sum_plus_epsilon.Resize(x->dims());
      x_square_sum_plus_epsilon.mutable_data<T>(place);

      const auto &runner = NpuOpRunner("Adds", {x_square_sum_broadcast},
                                       {x_square_sum_plus_epsilon}, attr_input);
      runner.Run(stream);

      LOG(WARNING) << "x_square_sum_plus_epsilon: ";
      PrintTensor<float>(x_square_sum_plus_epsilon, ctx);
    }

    // @TODO broadcast
    // x_square_sum -> x_square_sum_broadcast
    Tensor x_square_sum_sqrt;
    {
      x_square_sum_sqrt.Resize(x->dims());
      x_square_sum_sqrt.mutable_data<T>(place);

      const auto &runner = NpuOpRunner("Sqrt", {x_square_sum_broadcast},
                                       {x_square_sum_sqrt}, {});
      runner.Run(stream);

      LOG(WARNING) << "x_square_sum_sqrt: ";
      PrintTensor<float>(x_square_sum_sqrt, ctx);
    }

    // x * sum(x*dy)
    // x * x_mul_dy_sum
    // @TODO broadcast
    Tensor tmp1;
    {
      tmp1.Resize(x->dims());
      tmp1.mutable_data<T>(place);

      const auto &runner = NpuOpRunner("Mul", {*x, x_mul_dy_sum}, {tmp1}, {});
      runner.Run(stream);

      LOG(WARNING) << "tmp1: ";
      PrintTensor<float>(tmp1, ctx);
    }

    // x * sum(x*dy) / (sum(x*x) + e)
    // tmp1 / x_square_sum_plus_epsilon
    // @TODO broadcast
    Tensor tmp2;
    {
      tmp2.Resize(x->dims());
      tmp2.mutable_data<T>(place);

      const auto &runner =
          NpuOpRunner("Div", {tmp1, x_square_sum_plus_epsilon}, {tmp2}, {});
      runner.Run(stream);

      LOG(WARNING) << "tmp2: ";
      PrintTensor<float>(tmp2, ctx);
    }

    // dy - x * sum(x*dy) / (sum(x*x) + e)
    // dy - tmp2
    // @TODO broadcast
    Tensor tmp3;
    {
      tmp3.Resize(x->dims());
      tmp3.mutable_data<T>(place);

      const auto &runner = NpuOpRunner("Sub", {*dy, tmp2}, {tmp3}, {});
      runner.Run(stream);

      LOG(WARNING) << "tmp3: ";
      PrintTensor<float>(tmp3, ctx);
    }

    // at last, we get dx
    // tmp3 / x_square_sum_sqrt
    // @TODO broadcast
    {
      dx->mutable_data<T>(place);

      const auto &runner =
          NpuOpRunner("Div", {tmp3, x_square_sum_sqrt}, {*dx}, {});
      runner.Run(stream);

      LOG(WARNING) << "dx: ";
      PrintTensor<float>(*dx, ctx);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(
    norm, ops::NormNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::NormNPUKernel<paddle::platform::NPUDeviceContext,
                       paddle::platform::float16>)

REGISTER_OP_NPU_KERNEL(
    norm_grad, ops::NormGradNPUKernel<plat::NPUDeviceContext, float>,
    ops::NormGradNPUKernel<plat::NPUDeviceContext, plat::float16>);
