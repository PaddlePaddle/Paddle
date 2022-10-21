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
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

using NPUDeviceContext = platform::NPUDeviceContext;

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
    DataLayout data_layout = phi::StringToDataLayout(data_layout_str);

    const auto *x = ctx.Input<phi::DenseTensor>("X");
    const auto &x_dims = x->dims();
    PADDLE_ENFORCE_EQ(
        (x_dims.size() == 4UL || x_dims.size() == 3UL),
        true,
        platform::errors::InvalidArgument(
            "The input tensor X's dimension must equal to 3 or 4. "
            " But got X's shape = [%s], X's dimension = [%d].",
            x_dims.to_str(),
            x_dims.size()));

    const auto *running_mean = ctx.Input<phi::DenseTensor>("Mean");
    const auto *running_var = ctx.Input<phi::DenseTensor>("Variance");
    const auto *scale = ctx.Input<phi::DenseTensor>("Scale");
    const auto *bias = ctx.Input<phi::DenseTensor>("Bias");

    auto *y = ctx.Output<phi::DenseTensor>("Y");
    y->mutable_data<T>(ctx.GetPlace());

    auto &dev_ctx = ctx.template device_context<NPUDeviceContext>();
    auto x_tensor =
        ctx.AllocateTmpTensor<T, NPUDeviceContext>(x->dims(), dev_ctx);
    auto y_tesnor =
        ctx.AllocateTmpTensor<T, NPUDeviceContext>(y->dims(), dev_ctx);
    x_tensor.ShareDataWith(*x);
    y_tesnor.ShareDataWith(*y);
    if (data_layout == DataLayout::kNHWC) {
      x_tensor.set_layout(DataLayout::kNHWC);
      y_tesnor.set_layout(DataLayout::kNHWC);
    }

    auto stream = ctx.template device_context<NPUDeviceContext>().stream();
    if (!training) {
      const auto &runner_infer =
          NpuOpRunner("BNInfer",
                      {x_tensor, *scale, *bias, *running_mean, *running_var},
                      {y_tesnor},
                      {{"epsilon", epsilon}});
      runner_infer.Run(stream);
    } else {
      auto *mean_out = ctx.Output<phi::DenseTensor>("MeanOut");
      auto *variance_out = ctx.Output<phi::DenseTensor>("VarianceOut");
      auto *saved_mean = ctx.Output<phi::DenseTensor>("SavedMean");
      auto *saved_variance = ctx.Output<phi::DenseTensor>("SavedVariance");
      mean_out->mutable_data<float>(ctx.GetPlace());
      variance_out->mutable_data<float>(ctx.GetPlace());
      saved_mean->mutable_data<float>(ctx.GetPlace());
      saved_variance->mutable_data<float>(ctx.GetPlace());

      // if MomentumTensor is set, use MomentumTensor value, momentum
      // is only used in this training branch
      if (ctx.HasInput("MomentumTensor")) {
        const auto *mom_tensor = ctx.Input<phi::DenseTensor>("MomentumTensor");
        Tensor mom_cpu;
        paddle::framework::TensorCopySync(
            *mom_tensor, platform::CPUPlace(), &mom_cpu);
        momentum = mom_cpu.data<float>()[0];
      }

      phi::DenseTensor sum, square_sum;
      sum.mutable_data<float>(running_mean->dims(), ctx.GetPlace());
      square_sum.mutable_data<float>(running_mean->dims(), ctx.GetPlace());

      // BNTrainingReduce ONLY support rank = 4
      if (x->dims().size() == 3) {
        auto x_shape_vec = phi::vectorize(x->dims());
        if (data_layout == DataLayout::kNCHW) {
          x_shape_vec.push_back(1);  // expand NCL -> NCL1
        } else {
          x_shape_vec.insert(x_shape_vec.begin() + 2, 1);  // expand NLC -> NL1C
        }
        auto x_new_shape = phi::make_ddim(x_shape_vec);
        x_tensor.Resize(x_new_shape);
        x_tensor.Resize(x_new_shape);
      }
      const auto &runner_reduce = NpuOpRunner("BNTrainingReduce",
                                              {x_tensor},
                                              {sum, square_sum},
                                              {{"epsilon", epsilon}});
      runner_reduce.Run(stream);

      const auto &runner_update = NpuOpRunner(
          "BNTrainingUpdate",
          {x_tensor,
           sum,
           square_sum,
           *scale,
           *bias,
           *running_mean,
           *running_var},
          {y_tesnor, *mean_out, *variance_out, *saved_mean, *saved_variance},
          {{"factor", momentum}, {"epsilon", epsilon}});
      runner_update.Run(stream);
    }
  }
};

template <typename T>
class NPUBatchNormGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto *x = ctx.Input<phi::DenseTensor>("X");
    const auto *d_y = ctx.Input<phi::DenseTensor>(framework::GradVarName("Y"));
    const auto *scale = ctx.Input<phi::DenseTensor>("Scale");
    const auto *bias = ctx.Input<phi::DenseTensor>("Bias");
    const auto *saved_mean = ctx.Input<phi::DenseTensor>("SavedMean");
    // SavedVariance have been reverted in forward operator
    const auto *saved_inv_variance =
        ctx.Input<phi::DenseTensor>("SavedVariance");
    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
    bool use_global_stats = ctx.Attr<bool>("use_global_stats");
    const bool is_test = ctx.Attr<bool>("is_test");
    const float epsilon = ctx.Attr<float>("epsilon");
    DataLayout data_layout = phi::StringToDataLayout(data_layout_str);

    auto *d_x = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto *d_scale =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("Scale"));
    auto *d_bias = ctx.Output<phi::DenseTensor>(framework::GradVarName("Bias"));

    use_global_stats = is_test || use_global_stats;

    auto &dev_ctx = ctx.template device_context<NPUDeviceContext>();
    auto x_tensor =
        ctx.AllocateTmpTensor<T, NPUDeviceContext>(x->dims(), dev_ctx);
    auto dy_tensor =
        ctx.AllocateTmpTensor<T, NPUDeviceContext>(d_y->dims(), dev_ctx);
    x_tensor.ShareDataWith(*x);
    dy_tensor.ShareDataWith(*d_y);
    if (data_layout == DataLayout::kNHWC) {
      x_tensor.set_layout(DataLayout::kNHWC);
      dy_tensor.set_layout(DataLayout::kNHWC);
    }

    auto scale_grad_tmp =
        ctx.AllocateTmpTensor<T, NPUDeviceContext>(scale->dims(), dev_ctx);
    auto bias_grad_tmp =
        ctx.AllocateTmpTensor<T, NPUDeviceContext>(bias->dims(), dev_ctx);
    if (d_scale == nullptr) {
      d_scale = &scale_grad_tmp;
    }
    if (d_bias == nullptr) {
      d_bias = &bias_grad_tmp;
    }

    auto stream = ctx.template device_context<NPUDeviceContext>().stream();
    if (d_scale && d_bias) {
      d_scale->mutable_data<float>(ctx.GetPlace());
      d_bias->mutable_data<float>(ctx.GetPlace());
      if (use_global_stats) {
        const auto *running_mean = ctx.Input<phi::DenseTensor>("Mean");
        const auto *running_variance = ctx.Input<phi::DenseTensor>("Variance");
        const auto &runner_update =
            NpuOpRunner("BNTrainingUpdateGrad",
                        {dy_tensor, x_tensor, *running_mean, *running_variance},
                        {*d_scale, *d_bias},
                        {{"epsilon", epsilon}});
        runner_update.Run(stream);
      } else {
        const auto &runner_update =
            NpuOpRunner("BNTrainingUpdateGrad",
                        {dy_tensor, x_tensor, *saved_mean, *saved_inv_variance},
                        {*d_scale, *d_bias},
                        {{"epsilon", epsilon}});
        runner_update.Run(stream);
      }
    }
    if (d_x) {
      d_x->mutable_data<T>(ctx.GetPlace());
      auto dx_tensor =
          ctx.AllocateTmpTensor<T, NPUDeviceContext>(d_x->dims(), dev_ctx);
      dx_tensor.ShareDataWith(*d_x);
      if (data_layout == DataLayout::kNHWC) {
        dx_tensor.set_layout(DataLayout::kNHWC);
      }
      if (use_global_stats) {
        if (x->dims().size() == 3) {
          // BNInferGrad only support x rank = 4,
          auto x_shape_vec = phi::vectorize(d_x->dims());
          if (data_layout == DataLayout::kNCHW) {
            x_shape_vec.push_back(1);  // expand NCL -> NCL1
          } else {
            x_shape_vec.insert(x_shape_vec.begin() + 2,
                               1);  // expand NLC -> NL1C
          }
          auto x_new_shape = phi::make_ddim(x_shape_vec);
          dx_tensor.Resize(x_new_shape);
          dy_tensor.Resize(x_new_shape);
        }
        const auto *running_var = ctx.Input<phi::DenseTensor>("Variance");
        const auto &runner_infer =
            NpuOpRunner("BNInferGrad",
                        {dy_tensor, *scale, *running_var},
                        {dx_tensor},
                        {{"epsilon", epsilon}});
        runner_infer.Run(stream);
      } else {
        const auto &runner_reduce = NpuOpRunner("BNTrainingReduceGrad",
                                                {dy_tensor,
                                                 x_tensor,
                                                 *d_scale,
                                                 *d_bias,
                                                 *scale,
                                                 *saved_mean,
                                                 *saved_inv_variance},
                                                {dx_tensor},
                                                {{"epsilon", epsilon}});
        runner_reduce.Run(stream);
      }
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(batch_norm,
                       ops::NPUBatchNormOpKernel<float>,
                       ops::NPUBatchNormOpKernel<plat::float16>);
REGISTER_OP_NPU_KERNEL(batch_norm_grad,
                       ops::NPUBatchNormGradOpKernel<float>,
                       ops::NPUBatchNormGradOpKernel<plat::float16>);
