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

#include "paddle/fluid/operators/batch_norm_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename T>
class NPUBatchNormOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const float epsilon = ctx.Attr<float>("epsilon");
    float momentum = ctx.Attr<float>("momentum");
    const bool is_test = ctx.Attr<bool>("is_test");
    const bool use_global_stats = ctx.Attr<bool>("use_global_stats");
    const bool trainable_stats = ctx.Attr<bool>("trainable_statistics");

    bool test_mode = is_test && (!trainable_stats);
    bool training = !test_mode && !use_global_stats;

    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
    DataLayout data_layout = framework::StringToDataLayout(data_layout_str);

    const auto *x = ctx.Input<Tensor>("X");
    const auto &x_dims = x->dims();
    PADDLE_ENFORCE_EQ(x_dims.size(), 4,
                      platform::errors::InvalidArgument(
                          "The input tensor X's dimension must equal to 4. But "
                          "received X's shape = [%s], X's dimension = [%d].",
                          x_dims, x_dims.size()));
    const auto *running_mean = ctx.Input<Tensor>("Mean");
    const auto *running_var = ctx.Input<Tensor>("Variance");
    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *bias = ctx.Input<Tensor>("Bias");

    auto *y = ctx.Output<Tensor>("Y");
    y->mutable_data<T>(ctx.GetPlace());

    PADDLE_ENFORCE_EQ(
        x->layout(), data_layout,
        platform::errors::InvalidArgument(
            "BatchNorm OP's input tensor x layout should equal to attr "
            "data_layout, but got tensor layout <%s>, attr layout <%s>",
            framework::DataLayoutToString(x->layout()), data_layout_str));
    PADDLE_ENFORCE_EQ(
        y->layout(), data_layout,
        platform::errors::InvalidArgument(
            "BatchNorm OP's output tensor y layout should equal to attr "
            "data_layout, but got tensor layout <%s>, attr layout <%s>",
            framework::DataLayoutToString(y->layout()), data_layout_str));

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    if (!training) {
      const auto &runner_infer = NpuOpRunner(
          "BNInfer", {*x, *scale, *bias, *running_mean, *running_var}, {*y},
          {{"epsilon", epsilon}});
      runner_infer.Run(stream);
    } else {
      auto *mean_out = ctx.Output<Tensor>("MeanOut");
      auto *variance_out = ctx.Output<Tensor>("VarianceOut");
      auto *saved_mean = ctx.Output<Tensor>("SavedMean");
      auto *saved_variance = ctx.Output<Tensor>("SavedVariance");
      mean_out->mutable_data<T>(ctx.GetPlace());
      variance_out->mutable_data<T>(ctx.GetPlace());
      saved_mean->mutable_data<T>(ctx.GetPlace());
      saved_variance->mutable_data<T>(ctx.GetPlace());

      // if MomentumTensor is set, use MomentumTensor value, momentum
      // is only used in this training branch
      if (ctx.HasInput("MomentumTensor")) {
        const auto *mom_tensor = ctx.Input<Tensor>("MomentumTensor");
        Tensor mom_cpu;
        TensorCopySync(*mom_tensor, platform::CPUPlace(), &mom_cpu);
        momentum = mom_cpu.data<float>()[0];
      }

      framework::Tensor sum, square_sum;
      sum.mutable_data<float>(running_mean->dims(), ctx.GetPlace());
      square_sum.mutable_data<float>(running_mean->dims(), ctx.GetPlace());

      const auto &runner_reduce = NpuOpRunner(
          "BNTrainingReduce", {*x}, {sum, square_sum}, {{"epsilon", epsilon}});
      runner_reduce.Run(stream);

      const auto &runner_update = NpuOpRunner(
          "BNTrainingUpdate",
          {*x, sum, square_sum, *scale, *bias, *running_mean, *running_var},
          {*y, *mean_out, *variance_out, *saved_mean, *saved_variance},
          {{"factor", momentum}, {"epsilon", epsilon}});
      runner_update.Run(stream);
    }
  }
};

template <typename T>
class NPUBatchNormGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto *x = ctx.Input<Tensor>("X");
    const auto *d_y = ctx.Input<Tensor>(framework::GradVarName("Y"));
    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *bias = ctx.Input<Tensor>("Bias");
    const auto *saved_mean = ctx.Input<Tensor>("SavedMean");
    // SavedVariance have been reverted in forward operator
    const auto *saved_inv_variance = ctx.Input<Tensor>("SavedVariance");
    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
    bool use_global_stats = ctx.Attr<bool>("use_global_stats");
    const bool is_test = ctx.Attr<bool>("is_test");
    const float epsilon = ctx.Attr<float>("epsilon");
    DataLayout data_layout = framework::StringToDataLayout(data_layout_str);

    auto *d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *d_scale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto *d_bias = ctx.Output<Tensor>(framework::GradVarName("Bias"));

    PADDLE_ENFORCE_EQ(
        x->layout(), data_layout,
        platform::errors::InvalidArgument(
            "BatchNorm OP's input tensor x layout should equal to attr "
            "data_layout, but got tensor layout <%s>, attr layout <%s>",
            framework::DataLayoutToString(x->layout()), data_layout_str));
    PADDLE_ENFORCE_EQ(
        d_y->layout(), data_layout,
        platform::errors::InvalidArgument(
            "BatchNorm OP's input tensor d_y layout should equal to attr "
            "data_layout, but got tensor layout <%s>, attr layout <%s>",
            framework::DataLayoutToString(d_y->layout()), data_layout_str));

    use_global_stats = is_test || use_global_stats;

    Tensor scale_grad_tmp(scale->type());
    Tensor bias_grad_tmp(bias->type());
    if (d_scale == nullptr) {
      scale_grad_tmp.Resize(scale->dims());
      d_scale = &scale_grad_tmp;
    }
    if (d_bias == nullptr) {
      bias_grad_tmp.Resize(bias->dims());
      d_bias = &bias_grad_tmp;
    }

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    if (d_scale && d_bias) {
      d_scale->mutable_data<T>(ctx.GetPlace());
      d_bias->mutable_data<T>(ctx.GetPlace());
      const auto &runner_update = NpuOpRunner(
          "BNTrainingUpdateGrad", {*d_y, *x, *saved_mean, *saved_inv_variance},
          {*d_scale, *d_bias}, {{"epsilon", epsilon}});
      runner_update.Run(stream);
    }
    if (d_x) {
      d_x->mutable_data<T>(ctx.GetPlace());
      if (use_global_stats) {
        const auto *running_var = ctx.Input<Tensor>("Variance");
        const auto &runner_infer =
            NpuOpRunner("BNInferGrad", {*d_y, *scale, *running_var}, {*d_x},
                        {{"epsilon", epsilon}});
        runner_infer.Run(stream);
      } else {
        const auto &runner_reduce = NpuOpRunner(
            "BNTrainingReduceGrad", {*d_y, *x, *d_scale, *d_bias, *scale,
                                     *saved_mean, *saved_inv_variance},
            {*d_x}, {{"epsilon", epsilon}});
        runner_reduce.Run(stream);
      }
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(batch_norm, ops::NPUBatchNormOpKernel<float>,
                       ops::NPUBatchNormOpKernel<plat::float16>);
REGISTER_OP_NPU_KERNEL(batch_norm_grad, ops::NPUBatchNormGradOpKernel<float>,
                       ops::NPUBatchNormGradOpKernel<plat::float16>);
