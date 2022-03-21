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
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {

template <typename T>
class MLUMergedMomentumOpKernel : public framework::OpKernel<T> {
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

    auto mu = ctx.Attr<float>("mu");
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

    auto& dev_ctx = ctx.template device_context<platform::MLUDeviceContext>();

    Tensor mu_tensor = ctx.AllocateTmpTensor<T, MLUDeviceContext>({1}, dev_ctx);
    MLUCnnlTensorDesc mu_tensor_desc(mu_tensor);
    MLUCnnl::Fill(ctx, mu, mu_tensor_desc.get(), GetBasePtr(&mu_tensor));

    for (size_t idx = 0; idx < n; ++idx) {
      RegularizationType regularization_flag =
          regularization_methods.size() > 0 &&
                  regularization_methods[idx] == "l2_decay"
              ? RegularizationType::kL2DECAY
              : RegularizationType::kNONE;
      T regularization_coeff = static_cast<T>(0.0);
      if (regularization_coeffs.size() != 0) {
        regularization_coeff = static_cast<T>(regularization_coeffs[idx]);
      }

      auto learning_rate = lrs.size() > 1 ? lrs[idx] : lrs[0];
      auto param_out = params_out[idx];
      auto velocity_out = velocitys_out[idx];

      auto grad = grads[idx];
      Tensor regularized_grad;
      MLUCnnlTensorDesc param_desc(*param_out);
      if (regularization_flag == RegularizationType::kL2DECAY) {
        regularized_grad = ctx.AllocateTmpTensor<T, MLUDeviceContext>(
            param_out->dims(), dev_ctx);
        MLUCnnlOpTensorDesc op_tensor_desc(
            CNNL_OP_TENSOR_ADD, ToCnnlDataType<T>(), CNNL_NOT_PROPAGATE_NAN);
        MLUCnnl::OpTensor(ctx, op_tensor_desc.get(), param_desc.get(),
                          GetBasePtr(param_out), param_desc.get(),
                          GetBasePtr(grad), param_desc.get(),
                          GetBasePtr(&regularized_grad), ToCnnlDataType<T>(),
                          regularization_coeff);
      } else {
        regularized_grad = *grad;
      }
      MLUCnnl::ApplyMomentum(ctx, param_desc.get(),
                             GetBasePtr(&regularized_grad), use_nesterov,
                             GetBasePtr(learning_rate), GetBasePtr(&mu_tensor),
                             GetBasePtr(param_out), GetBasePtr(velocity_out));
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_MLU_KERNEL(merged_momentum, ops::MLUMergedMomentumOpKernel<float>,
                       ops::MLUMergedMomentumOpKernel<plat::float16>);
