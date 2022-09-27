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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op_mlu.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

template <typename T>
class AdamMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto* param_var = ctx.InputVar("Param");
    PADDLE_ENFORCE_EQ(param_var->IsType<framework::LoDTensor>(),
                      true,
                      platform::errors::InvalidArgument(
                          "The Var(%s)'s type should be LoDTensor, "
                          "but the received is %s",
                          ctx.InputNames("Param").front(),
                          framework::ToTypeName(param_var->Type())));
    auto* param = ctx.Input<LoDTensor>("Param");
    auto* grad_var = ctx.InputVar("Grad");
    PADDLE_ENFORCE_EQ(grad_var->IsType<framework::LoDTensor>(),
                      true,
                      platform::errors::InvalidArgument(
                          "The Grad(%s)'s type should be LoDTensor, "
                          "but the received is %s",
                          ctx.InputNames("Grad").front(),
                          framework::ToTypeName(param_var->Type())));
    auto* grad = ctx.Input<LoDTensor>("Grad");
    auto* mom1 = ctx.Input<LoDTensor>("Moment1");
    auto* mom2 = ctx.Input<LoDTensor>("Moment2");
    auto* lr = ctx.Input<LoDTensor>("LearningRate");

    auto* beta1_pow = ctx.Input<Tensor>("Beta1Pow");
    auto* beta2_pow = ctx.Input<Tensor>("Beta2Pow");

    auto* param_out = ctx.Output<LoDTensor>("ParamOut");
    auto* mom1_out = ctx.Output<LoDTensor>("Moment1Out");
    auto* mom2_out = ctx.Output<LoDTensor>("Moment2Out");
    auto* beta1_pow_out = ctx.Output<LoDTensor>("Beta1PowOut");
    auto* beta2_pow_out = ctx.Output<LoDTensor>("Beta2PowOut");

    bool skip_update = false;
    if (ctx.HasInput("SkipUpdate")) {
      auto* skip_update_tensor = ctx.Input<framework::Tensor>("SkipUpdate");
      PADDLE_ENFORCE_EQ(skip_update_tensor->numel(),
                        1,
                        platform::errors::InvalidArgument(
                            "Input(SkipUpdate) size must be 1, but get %d",
                            skip_update_tensor->numel()));
      std::vector<bool> skip_update_vec;
      paddle::framework::TensorToVector(
          *skip_update_tensor, ctx.device_context(), &skip_update_vec);
      ctx.device_context().Wait();
      skip_update = skip_update_vec[0];
    }
    // skip_update=true, just copy input to output, and TensorCopy will call
    // mutable_data
    if (skip_update) {
      VLOG(4) << "Adam skip update";
      framework::TensorCopy(
          *param,
          ctx.GetPlace(),
          ctx.template device_context<platform::MLUDeviceContext>(),
          param_out);
      framework::TensorCopy(
          *mom1,
          ctx.GetPlace(),
          ctx.template device_context<platform::MLUDeviceContext>(),
          mom1_out);
      framework::TensorCopy(
          *mom2,
          ctx.GetPlace(),
          ctx.template device_context<platform::MLUDeviceContext>(),
          mom2_out);
      framework::TensorCopy(
          *beta1_pow,
          beta1_pow->place(),
          ctx.template device_context<platform::MLUDeviceContext>(),
          beta1_pow_out);
      framework::TensorCopy(
          *beta2_pow,
          beta2_pow->place(),
          ctx.template device_context<platform::MLUDeviceContext>(),
          beta2_pow_out);
      return;
    }

    bool use_global_beta_pow = ctx.Attr<bool>("use_global_beta_pow");
    VLOG(4) << "use_global_beta_pow:" << use_global_beta_pow;

    param_out->ShareDataWith(*param);
    mom1_out->ShareDataWith(*mom1);
    mom2_out->ShareDataWith(*mom2);

    LoDTensor beta1_pow_tmp;
    LoDTensor beta2_pow_tmp;
    if (beta1_pow->place() == platform::CPUPlace()) {
      T beta1 = *beta1_pow->data<T>();
      beta1_pow_tmp.mutable_data<T>({1}, ctx.GetPlace());
      MLUCnnlTensorDesc beta1_pow_tmp_desc(beta1_pow_tmp);
      MLUCnnl::Fill(ctx,
                    CNNL_POINTER_MODE_HOST,
                    &beta1,
                    beta1_pow_tmp_desc.get(),
                    GetBasePtr(&beta1_pow_tmp));
      beta1_pow = &beta1_pow_tmp;
    }
    if (beta2_pow->place() == platform::CPUPlace()) {
      T beta2 = *beta2_pow->data<T>();
      beta2_pow_tmp.mutable_data<T>({1}, ctx.GetPlace());
      MLUCnnlTensorDesc beta2_pow_tmp_desc(beta2_pow_tmp);
      MLUCnnl::Fill(ctx,
                    CNNL_POINTER_MODE_HOST,
                    &beta2,
                    beta2_pow_tmp_desc.get(),
                    GetBasePtr(&beta2_pow_tmp));
      beta2_pow = &beta2_pow_tmp;
    }

    VLOG(3) << "beta1_pow.numel() : " << beta1_pow->numel()
            << "beta2_pow.numel() : " << beta2_pow->numel();
    VLOG(3) << "param.numel(): " << param->numel();

    PADDLE_ENFORCE_EQ(beta1_pow_out->numel(),
                      1,
                      platform::errors::InvalidArgument(
                          "beta1 pow output size should be 1, but received "
                          "value is:%d.",
                          beta1_pow_out->numel()));

    PADDLE_ENFORCE_EQ(beta2_pow_out->numel(),
                      1,
                      platform::errors::InvalidArgument(
                          "beta2 pow output size should be 1, but received "
                          "value is:%d.",
                          beta2_pow_out->numel()));

    const Tensor* beta1_tensor = nullptr;
    const Tensor* beta2_tensor = nullptr;
    const Tensor* epsilon_tensor = nullptr;

    Tensor beta1_tmp(experimental::DataType::FLOAT32);
    Tensor beta2_tmp(experimental::DataType::FLOAT32);
    Tensor epsilon_tmp(experimental::DataType::FLOAT32);

    if (ctx.HasInput("Beta1Tensor")) {
      beta1_tensor = ctx.Input<framework::Tensor>("Beta1Tensor");
      PADDLE_ENFORCE_EQ(beta1_tensor->numel(),
                        1,
                        platform::errors::InvalidArgument(
                            "Input(Beta1Tensor) size must be 1, but get %d",
                            beta1_tensor->numel()));
    } else {
      T beta1 = static_cast<T>(ctx.Attr<float>("beta1"));
      beta1_tmp.mutable_data<T>({1}, ctx.GetPlace());
      MLUCnnlTensorDesc beta1_tmp_desc(beta1_tmp);
      MLUCnnl::Fill(ctx,
                    CNNL_POINTER_MODE_HOST,
                    &beta1,
                    beta1_tmp_desc.get(),
                    GetBasePtr(&beta1_tmp));
      beta1_tensor = &beta1_tmp;
    }

    if (ctx.HasInput("Beta2Tensor")) {
      beta2_tensor = ctx.Input<framework::Tensor>("Beta2Tensor");
      PADDLE_ENFORCE_EQ(beta2_tensor->numel(),
                        1,
                        platform::errors::InvalidArgument(
                            "Input(Beta2Tensor) size must be 1, but get %d",
                            beta2_tensor->numel()));
    } else {
      T beta2 = static_cast<T>(ctx.Attr<float>("beta2"));
      beta2_tmp.mutable_data<T>({1}, ctx.GetPlace());
      MLUCnnlTensorDesc beta2_tmp_desc(beta2_tmp);
      MLUCnnl::Fill(ctx,
                    CNNL_POINTER_MODE_HOST,
                    &beta2,
                    beta2_tmp_desc.get(),
                    GetBasePtr(&beta2_tmp));
      beta2_tensor = &beta2_tmp;
    }

    if (ctx.HasInput("EpsilonTensor")) {
      epsilon_tensor = ctx.Input<framework::Tensor>("EpsilonTensor");
      PADDLE_ENFORCE_EQ(epsilon_tensor->numel(),
                        1,
                        platform::errors::InvalidArgument(
                            "Input(EpsilonTensor) size must be 1, but get %d",
                            epsilon_tensor->numel()));
    } else {
      T epsilon = static_cast<T>(ctx.Attr<float>("epsilon"));
      epsilon_tmp.mutable_data<T>({1}, ctx.GetPlace());
      MLUCnnlTensorDesc epsilon_tmp_desc(epsilon_tmp);
      MLUCnnl::Fill(ctx,
                    CNNL_POINTER_MODE_HOST,
                    &epsilon,
                    epsilon_tmp_desc.get(),
                    GetBasePtr(&epsilon_tmp));
      epsilon_tensor = &epsilon_tmp;
    }

    MLUCnnlTensorDesc param_desc(*param);
    MLUCnnlTensorDesc mom1_desc(*mom1);
    MLUCnnlTensorDesc mom2_desc(*mom2);
    MLUCnnlTensorDesc grad_desc(*grad);
    MLUCnnl::ApplyAdam(ctx,
                       param_desc.get(),
                       GetBasePtr(param_out),
                       mom1_desc.get(),
                       GetBasePtr(mom1_out),
                       mom2_desc.get(),
                       GetBasePtr(mom2_out),
                       grad_desc.get(),
                       GetBasePtr(grad),
                       GetBasePtr(lr),
                       GetBasePtr(beta1_tensor),
                       GetBasePtr(beta2_tensor),
                       GetBasePtr(beta1_pow),
                       GetBasePtr(beta2_pow),
                       GetBasePtr(epsilon_tensor),
                       /*use_nesterov*/ false);

    if (!use_global_beta_pow) {
      beta1_pow_out->mutable_data<T>(ctx.GetPlace());
      beta2_pow_out->mutable_data<T>(ctx.GetPlace());

      MLUCnnlTensorDesc beta1_desc(*beta1_tensor);
      MLUCnnlOpTensorDesc mul_op_desc(
          CNNL_OP_TENSOR_MUL, ToCnnlDataType<T>(), CNNL_NOT_PROPAGATE_NAN);

      MLUCnnl::OpTensor(ctx,
                        mul_op_desc.get(),
                        beta1_desc.get(),
                        GetBasePtr(beta1_pow),
                        beta1_desc.get(),
                        GetBasePtr(beta1_tensor),
                        beta1_desc.get(),
                        GetBasePtr(beta1_pow_out),
                        ToCnnlDataType<T>());

      MLUCnnl::OpTensor(ctx,
                        mul_op_desc.get(),
                        beta1_desc.get(),
                        GetBasePtr(beta2_pow),
                        beta1_desc.get(),
                        GetBasePtr(beta2_tensor),
                        beta1_desc.get(),
                        GetBasePtr(beta2_pow_out),
                        ToCnnlDataType<T>());
    }
  }
};

template <typename T>
class AdamWMLUKernel : public AdamMLUKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    VLOG(3) << "MLU AdamW Kernel";
    bool skip_update = false;
    if (ctx.HasInput("SkipUpdate")) {
      VLOG(3) << "Has SkipUpdate";
      auto* skip_update_tensor = ctx.Input<framework::Tensor>("SkipUpdate");
      PADDLE_ENFORCE_EQ(skip_update_tensor->numel(),
                        1,
                        platform::errors::InvalidArgument(
                            "Input(SkipUpdate) size must be 1, but get %d",
                            skip_update_tensor->numel()));
      std::vector<bool> skip_update_vec;
      paddle::framework::TensorToVector(
          *skip_update_tensor, ctx.device_context(), &skip_update_vec);
      ctx.device_context().Wait();
      skip_update = skip_update_vec[0];
    }
    bool with_decay = ctx.Attr<bool>("with_decay");
    VLOG(3) << "Skip update: " << skip_update << ", With decay: " << with_decay;
    if (!skip_update && with_decay) {
      if (ctx.HasInput("MasterParam")) {
        PADDLE_THROW(platform::errors::Unimplemented(
            "Master Param is not supported on MLU"));
      } else {
        const auto* param_var = ctx.InputVar("Param");
        PADDLE_ENFORCE_EQ(param_var->IsType<framework::LoDTensor>(),
                          true,
                          platform::errors::InvalidArgument(
                              "The Var(%s)'s type should be LoDTensor, "
                              "but the received is %s",
                              ctx.InputNames("Param").front(),
                              framework::ToTypeName(param_var->Type())));
        auto* param = ctx.Input<LoDTensor>("Param");
        auto* lr = ctx.Input<LoDTensor>("LearningRate");
        float coeff = ctx.Attr<float>("coeff");

        // update param with decay coeff: mul(-1 * lr, coeff * param) + param
        MLUCnnlTensorDesc lr_desc(*lr);
        MLUCnnlTensorDesc param_desc(*param);
        MLUCnnlOpTensorDesc mul_op_desc(
            CNNL_OP_TENSOR_MUL, ToCnnlDataType<T>(), CNNL_NOT_PROPAGATE_NAN);

        MLUCnnl::OpTensor(ctx,
                          mul_op_desc.get(),
                          lr_desc.get(),
                          GetBasePtr(lr),
                          param_desc.get(),
                          GetBasePtr(param),
                          param_desc.get(),
                          const_cast<void*>(GetBasePtr(param)),
                          ToCnnlDataType<T>(),
                          /*alpha1*/ -1.f,
                          /*alpha2*/ coeff,
                          /*beta*/ 1.f);
      }
    }
    AdamMLUKernel<T>::Compute(ctx);
  }
};

template <typename T>
class MergedAdamMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // Get inputs and outputs
    auto params = ctx.MultiInput<framework::Tensor>("Param");
    auto grads = ctx.MultiInput<framework::Tensor>("Grad");
    auto lrs = ctx.MultiInput<framework::Tensor>("LearningRate");
    auto mom1s = ctx.MultiInput<framework::Tensor>("Moment1");
    auto mom2s = ctx.MultiInput<framework::Tensor>("Moment2");
    auto beta1_pows = ctx.MultiInput<framework::Tensor>("Beta1Pow");
    auto beta2_pows = ctx.MultiInput<framework::Tensor>("Beta2Pow");
    auto master_params = ctx.MultiInput<framework::Tensor>("MasterParam");
    auto param_outs = ctx.MultiOutput<framework::Tensor>("ParamOut");
    auto mom1_outs = ctx.MultiOutput<framework::Tensor>("Moment1Out");
    auto mom2_outs = ctx.MultiOutput<framework::Tensor>("Moment2Out");
    auto beta1_pow_outs = ctx.MultiOutput<framework::Tensor>("Beta1PowOut");
    auto beta2_pow_outs = ctx.MultiOutput<framework::Tensor>("Beta2PowOut");

    // Check validation of inputs and outputs
    size_t param_num = params.size();
    PADDLE_ENFORCE_EQ(param_num,
                      param_outs.size(),
                      platform::errors::InvalidArgument(
                          "The size of Output(ParamOut) must be equal to "
                          "Input(Param), but got the size of Output(ParamOut) "
                          "is %d, the size of Input(Param) is %d.",
                          param_outs.size(),
                          param_num));

    bool skip_update = false;
    if (ctx.HasInput("SkipUpdate")) {
      auto* skip_update_tensor = ctx.Input<framework::Tensor>("SkipUpdate");
      PADDLE_ENFORCE_EQ(skip_update_tensor->numel(),
                        1,
                        platform::errors::InvalidArgument(
                            "Input(SkipUpdate) size must be 1, but get %d",
                            skip_update_tensor->numel()));
      std::vector<bool> skip_update_vec;
      paddle::framework::TensorToVector(
          *skip_update_tensor, ctx.device_context(), &skip_update_vec);
      ctx.device_context().Wait();
      skip_update = skip_update_vec[0];
    }
    // skip_update=true, just copy input to output, and TensorCopy will call
    // mutable_data

    if (skip_update) {
      VLOG(4) << "MergedAdam skip update";
      for (size_t i = 0; i < param_num; ++i) {
        framework::TensorCopy(
            *params[i],
            ctx.GetPlace(),
            ctx.template device_context<platform::MLUDeviceContext>(),
            param_outs[i]);
        framework::TensorCopy(
            *mom1s[i],
            ctx.GetPlace(),
            ctx.template device_context<platform::MLUDeviceContext>(),
            mom1_outs[i]);
        framework::TensorCopy(
            *mom2s[i],
            ctx.GetPlace(),
            ctx.template device_context<platform::MLUDeviceContext>(),
            mom2_outs[i]);
        framework::TensorCopy(
            *beta1_pows[i],
            beta1_pows[i]->place(),
            ctx.template device_context<platform::MLUDeviceContext>(),
            beta1_pow_outs[i]);
        framework::TensorCopy(
            *beta2_pows[i],
            beta2_pows[i]->place(),
            ctx.template device_context<platform::MLUDeviceContext>(),
            beta2_pow_outs[i]);
      }
      return;
    }

    bool use_global_beta_pow = ctx.Attr<bool>("use_global_beta_pow");
    VLOG(4) << "use_global_beta_pow:" << use_global_beta_pow;

    // Get beta1, beta2 and epsilon from attribute.
    const Tensor* beta1_tensor = nullptr;
    const Tensor* beta2_tensor = nullptr;
    const Tensor* epsilon_tensor = nullptr;

    Tensor beta1_tmp(experimental::DataType::FLOAT32);
    Tensor beta2_tmp(experimental::DataType::FLOAT32);
    Tensor epsilon_tmp(experimental::DataType::FLOAT32);

    T beta1 = static_cast<T>(ctx.Attr<float>("beta1"));
    T beta2 = static_cast<T>(ctx.Attr<float>("beta2"));
    T epsilon = static_cast<T>(ctx.Attr<float>("epsilon"));
    beta1_tmp.mutable_data<T>({1}, ctx.GetPlace());
    beta2_tmp.mutable_data<T>({1}, ctx.GetPlace());
    epsilon_tmp.mutable_data<T>({1}, ctx.GetPlace());
    MLUCnnlTensorDesc beta1_tmp_desc(beta1_tmp);
    MLUCnnlTensorDesc beta2_tmp_desc(beta2_tmp);
    MLUCnnlTensorDesc epsilon_tmp_desc(epsilon_tmp);
    MLUCnnl::Fill(ctx,
                  CNNL_POINTER_MODE_HOST,
                  &beta1,
                  beta1_tmp_desc.get(),
                  GetBasePtr(&beta1_tmp));
    MLUCnnl::Fill(ctx,
                  CNNL_POINTER_MODE_HOST,
                  &beta2,
                  beta2_tmp_desc.get(),
                  GetBasePtr(&beta2_tmp));
    MLUCnnl::Fill(ctx,
                  CNNL_POINTER_MODE_HOST,
                  &epsilon,
                  epsilon_tmp_desc.get(),
                  GetBasePtr(&epsilon_tmp));
    beta1_tensor = &beta1_tmp;
    beta2_tensor = &beta2_tmp;
    epsilon_tensor = &epsilon_tmp;

    // Loop to compute
    for (size_t i = 0; i < param_num; ++i) {
      VLOG(4) << "[MergedAdam] loop: " << i;
      param_outs[i]->ShareDataWith(*params[i]);
      mom1_outs[i]->ShareDataWith(*mom1s[i]);
      mom2_outs[i]->ShareDataWith(*mom2s[i]);

      LoDTensor beta1_pow_tmp;
      LoDTensor beta2_pow_tmp;
      if (beta1_pows[i]->place() == platform::CPUPlace()) {
        T beta1 = *beta1_pows[i]->data<T>();
        beta1_pow_tmp.mutable_data<T>({1}, ctx.GetPlace());
        MLUCnnlTensorDesc beta1_pow_tmp_desc(beta1_pow_tmp);
        MLUCnnl::Fill(ctx,
                      CNNL_POINTER_MODE_HOST,
                      &beta1,
                      beta1_pow_tmp_desc.get(),
                      GetBasePtr(&beta1_pow_tmp));
        beta1_pows[i] = &beta1_pow_tmp;
      }
      if (beta2_pows[i]->place() == platform::CPUPlace()) {
        T beta2 = *beta2_pows[i]->data<T>();
        beta2_pow_tmp.mutable_data<T>({1}, ctx.GetPlace());
        MLUCnnlTensorDesc beta2_pow_tmp_desc(beta2_pow_tmp);
        MLUCnnl::Fill(ctx,
                      CNNL_POINTER_MODE_HOST,
                      &beta2,
                      beta2_pow_tmp_desc.get(),
                      GetBasePtr(&beta2_pow_tmp));
        beta2_pows[i] = &beta2_pow_tmp;
      }

      VLOG(3) << "beta1_pow.numel() : " << beta1_pows[i]->numel()
              << "beta2_pow.numel() : " << beta2_pows[i]->numel();
      VLOG(3) << "param.numel(): " << params[i]->numel();
      PADDLE_ENFORCE_EQ(beta1_pow_outs[i]->numel(),
                        1,
                        platform::errors::InvalidArgument(
                            "beta1 pow output size should be 1, but received "
                            "value is:%d.",
                            beta1_pow_outs[i]->numel()));

      PADDLE_ENFORCE_EQ(beta2_pow_outs[i]->numel(),
                        1,
                        platform::errors::InvalidArgument(
                            "beta2 pow output size should be 1, but received "
                            "value is:%d.",
                            beta2_pow_outs[i]->numel()));
      MLUCnnlTensorDesc param_desc(*params[i]);
      MLUCnnlTensorDesc mom1_desc(*mom1s[i]);
      MLUCnnlTensorDesc mom2_desc(*mom2s[i]);
      MLUCnnlTensorDesc grad_desc(*grads[i]);
      MLUCnnl::ApplyAdam(ctx,
                         param_desc.get(),
                         GetBasePtr(param_outs[i]),
                         mom1_desc.get(),
                         GetBasePtr(mom1_outs[i]),
                         mom2_desc.get(),
                         GetBasePtr(mom2_outs[i]),
                         grad_desc.get(),
                         GetBasePtr(grads[i]),
                         GetBasePtr(lrs[i]),
                         GetBasePtr(beta1_tensor),
                         GetBasePtr(beta2_tensor),
                         GetBasePtr(beta1_pows[i]),
                         GetBasePtr(beta2_pows[i]),
                         GetBasePtr(epsilon_tensor),
                         /*use_nesterov*/ false);
      if (!use_global_beta_pow) {
        beta1_pow_outs[i]->mutable_data<T>(ctx.GetPlace());
        beta2_pow_outs[i]->mutable_data<T>(ctx.GetPlace());

        MLUCnnlTensorDesc beta1_desc(*beta1_tensor);
        MLUCnnlOpTensorDesc mul_op_desc(
            CNNL_OP_TENSOR_MUL, ToCnnlDataType<T>(), CNNL_NOT_PROPAGATE_NAN);

        MLUCnnl::OpTensor(ctx,
                          mul_op_desc.get(),
                          beta1_desc.get(),
                          GetBasePtr(beta1_pows[i]),
                          beta1_desc.get(),
                          GetBasePtr(beta1_tensor),
                          beta1_desc.get(),
                          GetBasePtr(beta1_pow_outs[i]),
                          ToCnnlDataType<T>());

        MLUCnnl::OpTensor(ctx,
                          mul_op_desc.get(),
                          beta1_desc.get(),
                          GetBasePtr(beta2_pows[i]),
                          beta1_desc.get(),
                          GetBasePtr(beta2_tensor),
                          beta1_desc.get(),
                          GetBasePtr(beta2_pow_outs[i]),
                          ToCnnlDataType<T>());
      }
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(adam,
                       ops::AdamMLUKernel<float>,
                       ops::AdamMLUKernel<plat::float16>);

REGISTER_OP_MLU_KERNEL(adamw,
                       ops::AdamWMLUKernel<float>,
                       ops::AdamWMLUKernel<plat::float16>);

REGISTER_OP_MLU_KERNEL(merged_adam,
                       ops::MergedAdamMLUKernel<float>,
                       ops::MergedAdamMLUKernel<plat::float16>);
