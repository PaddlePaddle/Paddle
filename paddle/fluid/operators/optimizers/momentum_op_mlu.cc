/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/mlu/mlu_baseop.h"
#include "paddle/fluid/operators/optimizers/momentum_op.h"
#include "paddle/phi/kernels/impl/momentum_kernel_impl.h"

namespace paddle {
namespace operators {

template <typename T>
class MLUMomentumOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<platform::MLUDeviceContext>();

    std::string regularization_method =
        ctx.Attr<std::string>("regularization_method");
    auto regularization_coeff = ctx.Attr<float>("regularization_coeff");
    phi::RegularizationType regularization_flag{
        phi::RegularizationType::kNONE};  // disable regularization
    if (regularization_method == "l2_decay") {
      regularization_flag = phi::RegularizationType::kL2DECAY;
    }

    T mu = static_cast<T>(ctx.Attr<float>("mu"));
    bool use_nesterov = ctx.Attr<bool>("use_nesterov");

    auto learning_rate = ctx.Input<phi::DenseTensor>("LearningRate");
    auto param = ctx.Input<phi::DenseTensor>("Param");
    auto velocity = ctx.Input<phi::DenseTensor>("Velocity");

    auto param_out = ctx.Output<phi::DenseTensor>("ParamOut");
    auto velocity_out = ctx.Output<phi::DenseTensor>("VelocityOut");

    param_out->mutable_data<T>(ctx.GetPlace());
    velocity_out->mutable_data<T>(ctx.GetPlace());

    auto* grad_var = ctx.InputVar("Grad");
    if (grad_var->IsType<framework::LoDTensor>()) {
      auto grad = ctx.Input<phi::DenseTensor>("Grad");
      Tensor mu_tensor =
          ctx.AllocateTmpTensor<T, MLUDeviceContext>({1}, dev_ctx);
      MLUCnnlTensorDesc mu_tensor_desc(mu_tensor);
      MLUCnnl::Fill(ctx,
                    CNNL_POINTER_MODE_HOST,
                    &mu,
                    mu_tensor_desc.get(),
                    GetBasePtr(&mu_tensor));

      Tensor regularized_grad;
      MLUCnnlTensorDesc param_desc(*param);
      if (regularization_flag == phi::RegularizationType::kL2DECAY) {
        regularized_grad =
            ctx.AllocateTmpTensor<T, MLUDeviceContext>(param->dims(), dev_ctx);
        MLUCnnlOpTensorDesc op_tensor_desc(
            CNNL_OP_TENSOR_ADD, ToCnnlDataType<T>(), CNNL_NOT_PROPAGATE_NAN);
        MLUCnnl::OpTensor(ctx,
                          op_tensor_desc.get(),
                          param_desc.get(),
                          GetBasePtr(param),
                          param_desc.get(),
                          GetBasePtr(grad),
                          param_desc.get(),
                          GetBasePtr(&regularized_grad),
                          ToCnnlDataType<T>(),
                          regularization_coeff);
      } else {
        regularized_grad = *grad;
      }
      framework::TensorCopy(*param, ctx.GetPlace(), dev_ctx, param_out);
      framework::TensorCopy(*velocity, ctx.GetPlace(), dev_ctx, velocity_out);
      MLUCnnl::ApplyMomentum(ctx,
                             param_desc.get(),
                             GetBasePtr(&regularized_grad),
                             use_nesterov,
                             GetBasePtr(learning_rate),
                             GetBasePtr(&mu_tensor),
                             GetBasePtr(param_out),
                             GetBasePtr(velocity_out));
    } else if (grad_var->IsType<phi::SelectedRows>()) {
      PADDLE_ENFORCE_EQ(
          false,
          true,
          platform::errors::PermissionDenied("Unsupport SparseMomentum"));
    } else {
      PADDLE_ENFORCE_EQ(false,
                        true,
                        platform::errors::PermissionDenied(
                            "Unsupported Variable Type of Grad "
                            "in MomentumOp. Excepted LodTensor "
                            "or SelectedRows, But received [%s]",
                            paddle::framework::ToTypeName(grad_var->Type())));
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_MLU_KERNEL(momentum,
                       ops::MLUMomentumOpKernel<float>,
                       ops::MLUMomentumOpKernel<plat::float16>);
