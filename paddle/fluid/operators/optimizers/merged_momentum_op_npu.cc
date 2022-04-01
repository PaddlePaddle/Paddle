// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/optimizers/merged_momentum_op.h"

#include "paddle/fluid/platform/device/npu/npu_op_runner.h"
#include "paddle/phi/kernels/impl/momentum_kernel_impl.h"

namespace paddle {
namespace operators {

template <typename T>
class NPUMergedMomentumOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto params = ctx.MultiInput<framework::Tensor>("Param");
    auto params_out = ctx.MultiOutput<framework::Tensor>("ParamOut");
    size_t n = params.size();
    PADDLE_ENFORCE_EQ(n, params_out.size(),
                      platform::errors::InvalidArgument(
                          "The size of Output(ParamOut) must be equal to "
                          "Input(Param), but got the size of Output(ParamOut) "
                          "is %d, the size of Input(Param) is %d.",
                          params_out.size(), n));
    for (size_t i = 0; i < n; ++i) {
      PADDLE_ENFORCE_EQ(params[i], params_out[i],
                        platform::errors::InvalidArgument(
                            "The size of Input(Param) and Output(ParamOut) "
                            "must be the same Tensors."));
    }

    auto grads = ctx.MultiInput<framework::Tensor>("Grad");
    PADDLE_ENFORCE_EQ(
        n, grads.size(),
        platform::errors::InvalidArgument(
            "The size of Input(Grad) must be equal to Input(Param), but got "
            "the size of Input(Grad) is %d, the size of Input(Param) is %d.",
            grads.size(), n));

    auto velocitys = ctx.MultiInput<framework::Tensor>("Velocity");
    PADDLE_ENFORCE_EQ(n, velocitys.size(),
                      platform::errors::InvalidArgument(
                          "The size of Input(Velocity) must be equal to "
                          "Input(Param), but got the size of Input(Velocity) "
                          "is %d, the size of Input(Param) is %d.",
                          velocitys.size(), n));

    auto velocitys_out = ctx.MultiOutput<framework::Tensor>("VelocityOut");
    PADDLE_ENFORCE_EQ(
        n, velocitys_out.size(),
        platform::errors::InvalidArgument(
            "The size of Output(VelocityOut) must be "
            "equal to Input(Param), but got the size of Output(VelocityOut) is "
            "%d, the size of Input(Param) is %d.",
            velocitys_out.size(), n));
    for (size_t i = 0; i < n; ++i) {
      PADDLE_ENFORCE_EQ(velocitys[i], velocitys_out[i],
                        platform::errors::InvalidArgument(
                            "Input(Velocity) and Output(VelocityOut) must be "
                            "the same Tensors."));
    }

    T mu = static_cast<T>(ctx.Attr<float>("mu"));
    auto lrs = ctx.MultiInput<framework::Tensor>("LearningRate");
    if (lrs.size() != 1) {
      PADDLE_ENFORCE_EQ(
          n, lrs.size(),
          platform::errors::InvalidArgument(
              "If the size of Input(LearningRate) is not 1, the size of "
              "Input(LearningRate) must be "
              "equal to Input(Param), but got the size of Input(LearningRate) "
              "is %d, the size of Input(Param) is %d.",
              lrs.size(), n));
    }
    auto use_nesterov = ctx.Attr<bool>("use_nesterov");
    auto regularization_methods =
        ctx.Attr<std::vector<std::string>>("regularization_method");
    auto regularization_coeffs =
        ctx.Attr<std::vector<float>>("regularization_coeff");
    if (regularization_methods.size() != 0) {
      PADDLE_ENFORCE_EQ(
          n, regularization_methods.size(),
          platform::errors::InvalidArgument(
              "The size of Attr(regularization_method) must be equal "
              "to Input(Param), but got the size of "
              "Attr(regularization_method) is %d, the size of Input(Param) is "
              "%d.",
              regularization_methods.size(), n));
      PADDLE_ENFORCE_EQ(
          n, regularization_coeffs.size(),
          platform::errors::InvalidArgument(
              "The size of Attr(regularization_coeff) must be equal "
              "to Input(Param), but got the size of Attr(regularization_coeff) "
              "is %d, the size of Input(Param) is %d.",
              regularization_coeffs.size(), n));
    }

    VLOG(5) << "use_nesterov: " << use_nesterov
            << ",  regularization_methods.size(): "
            << regularization_methods.size()
            << ",  regularization_coeffs.size(): "
            << regularization_coeffs.size();

    auto& dev_ctx = ctx.template device_context<platform::NPUDeviceContext>();

    Tensor mu_tensor;
    mu_tensor.mutable_data<T>(phi::make_ddim({1}), ctx.GetPlace());
    FillNpuTensorWithConstant<T>(&mu_tensor, mu);

    for (size_t idx = 0; idx < n; ++idx) {
      phi::RegularizationType regularization_flag =
          regularization_methods.size() > 0 &&
                  regularization_methods[idx] == "l2_decay"
              ? phi::RegularizationType::kL2DECAY
              : phi::RegularizationType::kNONE;
      float regularization_coeff = 0.0;
      if (regularization_coeffs.size() != 0) {
        regularization_coeff = regularization_coeffs[idx];
      }

      auto learning_rate = lrs.size() > 1 ? lrs[idx] : lrs[0];
      auto param = params[idx];
      auto param_out = params_out[idx];
      auto velocity = velocitys[idx];
      auto velocity_out = velocitys_out[idx];

      auto grad = grads[idx];
      Tensor regularized_grad;
      if (regularization_flag == phi::RegularizationType::kL2DECAY) {
        regularized_grad.mutable_data<T>(grad->dims(), ctx.GetPlace());
        const auto& runner1 = NpuOpRunner("Muls", {*param}, {regularized_grad},
                                          {{"value", regularization_coeff}});
        runner1.Run(dev_ctx.stream());
        const auto& runner2 = NpuOpRunner("Add", {regularized_grad, *grad},
                                          {regularized_grad}, {});
        runner2.Run(dev_ctx.stream());
      } else {
        regularized_grad.ShareDataWith(*grad);
      }
      framework::TensorCopy(*param, ctx.GetPlace(), dev_ctx, param_out);
      framework::TensorCopy(*velocity, ctx.GetPlace(), dev_ctx, velocity_out);
      // NOTE: ApplyMomentum will change the input
      const auto& runner = NpuOpRunner(
          "ApplyMomentum", {*param_out, *velocity_out, *learning_rate,
                            regularized_grad, mu_tensor},
          {*param_out}, {{"use_nesterov", use_nesterov}});
      runner.Run(dev_ctx.stream());
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_NPU_KERNEL(merged_momentum, ops::NPUMergedMomentumOpKernel<float>,
                       ops::NPUMergedMomentumOpKernel<plat::float16>);
