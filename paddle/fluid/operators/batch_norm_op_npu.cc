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

using NPUDeviceContext = platform::NPUDeviceContext;

using Tensor = framework::Tensor;

template <typename T>
std::string outputVector(const std::vector<T> vec) {
  std::ostringstream oss;
  // for (auto ele : vec) oss << ele << ' ';
  for (size_t i = 0; i < vec.size() && i < 10; ++i) {
    oss << vec[i] << ' ';
  }
  return oss.str();
}
template <typename T>
void PrintTensor(const framework::Tensor &src,
                 const framework::ExecutionContext &ctx) {
  std::vector<T> vec(src.numel());
  TensorToVector(src, ctx.device_context(), &vec);
  LOG(WARNING) << "vec: " << outputVector<T>(vec);
}

template <typename T>
class NPUBatchNormOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    LOG(WARNING) << "NPUBatchNormOpKernel";

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

    LOG(WARNING) << "Input Tensor | x: ";
    PrintTensor<float>(*x, ctx);
    LOG(WARNING) << "Input Tensor | scale: ";
    PrintTensor<float>(*scale, ctx);
    LOG(WARNING) << "Input Tensor | bias: ";
    PrintTensor<float>(*bias, ctx);
    LOG(WARNING) << "Input Tensor | mean: ";
    PrintTensor<float>(*running_mean, ctx);
    LOG(WARNING) << "Input Tensor | variance: ";
    PrintTensor<float>(*running_var, ctx);

    auto *y = ctx.Output<Tensor>("Y");
    y->mutable_data<T>(ctx.GetPlace());

    Tensor x_tensor(x->type());
    Tensor y_tesnor(y->type());
    x_tensor.ShareDataWith(*x);
    y_tesnor.ShareDataWith(*y);
    if (data_layout == DataLayout::kNHWC) {
      x_tensor.set_layout(DataLayout::kNHWC);
      y_tesnor.set_layout(DataLayout::kNHWC);
    }

    auto stream = ctx.template device_context<NPUDeviceContext>().stream();
    if (!training) {
      const auto &runner_infer = NpuOpRunner(
          "BNInfer", {x_tensor, *scale, *bias, *running_mean, *running_var},
          {y_tesnor}, {{"epsilon", epsilon}});
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

      const auto &runner_reduce =
          NpuOpRunner("BNTrainingReduce", {x_tensor}, {sum, square_sum},
                      {{"epsilon", epsilon}});
      runner_reduce.Run(stream);

      const auto &runner_update = NpuOpRunner(
          "BNTrainingUpdate", {x_tensor, sum, square_sum, *scale, *bias,
                               *running_mean, *running_var},
          {y_tesnor, *mean_out, *variance_out, *saved_mean, *saved_variance},
          {{"factor", momentum}, {"epsilon", epsilon}});
      runner_update.Run(stream);

      LOG(WARNING) << "Output Tensor | y: ";
      PrintTensor<float>(*y, ctx);
      LOG(WARNING) << "Output Tensor | mean_out: ";
      PrintTensor<float>(*mean_out, ctx);
      LOG(WARNING) << "Output Tensor | variance_out: ";
      PrintTensor<float>(*variance_out, ctx);
      LOG(WARNING) << "Output Tensor | saved_mean: ";
      PrintTensor<float>(*saved_mean, ctx);
      LOG(WARNING) << "Output Tensor | saved_variance: ";
      PrintTensor<float>(*saved_variance, ctx);
    }
  }
};

template <typename T>
class NPUBatchNormGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    LOG(WARNING) << "NPUBatchNormGradOpKernel";

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

    LOG(WARNING) << "Input Tensor | x: ";
    PrintTensor<T>(*x, ctx);
    LOG(WARNING) << "Input Tensor | dy: ";
    PrintTensor<T>(*d_y, ctx);
    LOG(WARNING) << "Input Tensor | scale: ";
    PrintTensor<T>(*scale, ctx);
    LOG(WARNING) << "Input Tensor | bias: ";
    PrintTensor<T>(*bias, ctx);
    LOG(WARNING) << "Input Tensor | saved_mean: ";
    PrintTensor<T>(*saved_mean, ctx);
    LOG(WARNING) << "Input Tensor | saved_variance: ";
    PrintTensor<T>(*saved_inv_variance, ctx);

    auto *d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *d_scale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto *d_bias = ctx.Output<Tensor>(framework::GradVarName("Bias"));

    use_global_stats = is_test || use_global_stats;

    Tensor x_tensor(x->type());
    Tensor dy_tensor(d_y->type());
    x_tensor.ShareDataWith(*x);
    dy_tensor.ShareDataWith(*d_y);
    if (data_layout == DataLayout::kNHWC) {
      x_tensor.set_layout(DataLayout::kNHWC);
      dy_tensor.set_layout(DataLayout::kNHWC);
    }

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

    auto stream = ctx.template device_context<NPUDeviceContext>().stream();
    if (d_scale && d_bias) {
      d_scale->mutable_data<T>(ctx.GetPlace());
      d_bias->mutable_data<T>(ctx.GetPlace());
      if (use_global_stats) {
        const auto *running_mean = ctx.Input<Tensor>("Mean");
        const auto *running_variance = ctx.Input<Tensor>("Variance");
        const auto &runner_update =
            NpuOpRunner("BNTrainingUpdateGrad",
                        {dy_tensor, x_tensor, *running_mean, *running_variance},
                        {*d_scale, *d_bias}, {{"epsilon", epsilon}});
        runner_update.Run(stream);
      } else {
        const auto &runner_update =
            NpuOpRunner("BNTrainingUpdateGrad",
                        {dy_tensor, x_tensor, *saved_mean, *saved_inv_variance},
                        {*d_scale, *d_bias}, {{"epsilon", epsilon}});
        runner_update.Run(stream);
      }
    }
    if (d_x) {
      d_x->mutable_data<T>(ctx.GetPlace());
      Tensor dx_tensor(d_x->type());
      dx_tensor.ShareDataWith(*d_x);
      if (use_global_stats) {
        const auto *running_var = ctx.Input<Tensor>("Variance");
        const auto &runner_infer =
            NpuOpRunner("BNInferGrad", {dy_tensor, *scale, *running_var},
                        {dx_tensor}, {{"epsilon", epsilon}});
        runner_infer.Run(stream);
      } else {
        const auto &runner_reduce = NpuOpRunner(
            "BNTrainingReduceGrad", {dy_tensor, x_tensor, *d_scale, *d_bias,
                                     *scale, *saved_mean, *saved_inv_variance},
            {dx_tensor}, {{"epsilon", epsilon}});
        runner_reduce.Run(stream);
      }
    }

    LOG(WARNING) << "Output Tensor | d_x: ";
    PrintTensor<float>(*d_x, ctx);
    LOG(WARNING) << "Output Tensor | d_scale: ";
    PrintTensor<float>(*d_scale, ctx);
    LOG(WARNING) << "Output Tensor | d_bias: ";
    PrintTensor<float>(*d_bias, ctx);
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
