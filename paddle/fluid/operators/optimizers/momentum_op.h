/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <memory>
#include <string>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/algorithm.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

using framework::Tensor;
using framework::SelectedRows;
struct NoNesterov;
struct UseNesterov;

enum class RegularizationType {
  kNONE = 0,
  kL1DECAY = 1,  // do not need support right now
  kL2DECAY = 2,
};

class MomentumOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override;
};

class MomentumOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("Param"), true,
                      platform::errors::NotFound(
                          "Input(param) of Momentum should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("Grad"), true,
                      platform::errors::NotFound(
                          "Input(grad) of Momentum should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("Velocity"), true,
                      platform::errors::NotFound(
                          "Input(velocity) of Momentum should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("LearningRate"), true,
        platform::errors::NotFound(
            "Input(LearningRate) of Momentum should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->GetInputsVarType("Param").front(),
        framework::proto::VarType::LOD_TENSOR,
        platform::errors::InvalidArgument(
            "The input var's type should be LoDTensor, but the received is %s",
            ctx->GetInputsVarType("Param").front()));

    PADDLE_ENFORCE_EQ(ctx->HasOutput("ParamOut"), true,
                      platform::errors::NotFound(
                          "Output(ParamOut) of Momentum should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("VelocityOut"), true,
        platform::errors::NotFound(
            "Output(VelocityOut) of Momentum should not be null."));

    auto lr_dims = ctx->GetInputDim("LearningRate");
    PADDLE_ENFORCE_NE(framework::product(lr_dims), 0,
                      platform::errors::InvalidArgument(
                          "Maybe the Input variable LearningRate has not "
                          "been initialized. You may need to confirm "
                          "if you put exe.run(startup_program) "
                          "after optimizer.minimize function."));
    PADDLE_ENFORCE_EQ(framework::product(lr_dims), 1,
                      platform::errors::InvalidArgument(
                          "Learning_rate should be a scalar. But Received "
                          "LearningRate's dim [%s]",
                          framework::product(lr_dims)));

    auto param_dim = ctx->GetInputDim("Param");
    if (ctx->GetInputsVarType("Grad")[0] ==
        framework::proto::VarType::LOD_TENSOR) {
      PADDLE_ENFORCE_EQ(
          param_dim, ctx->GetInputDim("Grad"),
          platform::errors::InvalidArgument(
              "Param and Grad input of MomentumOp should have the same "
              "dimension. But received Param's dim [%s] and Grad's dim [%s].",
              param_dim, ctx->GetInputDim("Grad")));
      PADDLE_ENFORCE_EQ(
          param_dim, ctx->GetInputDim("Velocity"),
          platform::errors::InvalidArgument(
              "Param and Velocity of MomentumOp should have the same "
              "dimension. But received Param's dim [%s] and Velocity [%s].",
              param_dim, ctx->GetInputDim("Velocity")));
    }

    ctx->SetOutputDim("ParamOut", param_dim);
    ctx->SetOutputDim("VelocityOut", param_dim);
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_data_type =
        OperatorWithKernel::IndicateVarDataType(ctx, "Param");
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

template <typename T>
class CPUDenseMomentumFunctor {
 private:
  const Tensor* param_;
  const Tensor* grad_;
  const Tensor* velocity_;
  const Tensor* learning_rate_;
  const T mu_;
  const T use_nesterov_;
  RegularizationType regularization_flag_;
  const T regularization_coeff_;
  Tensor* param_out_;
  Tensor* velocity_out_;

 public:
  CPUDenseMomentumFunctor(const Tensor* param, const Tensor* grad,
                          const Tensor* velocity, const Tensor* learning_rate,
                          const T mu, const bool use_nesterov,
                          RegularizationType regularization_flag,
                          const T regularization_coeff, Tensor* param_out,
                          Tensor* velocity_out)
      : param_(param),
        grad_(grad),
        velocity_(velocity),
        learning_rate_(learning_rate),
        mu_(mu),
        use_nesterov_(use_nesterov),
        regularization_flag_(regularization_flag),
        regularization_coeff_(regularization_coeff),
        param_out_(param_out),
        velocity_out_(velocity_out) {}

  inline void operator()() {
    auto param_out = framework::EigenVector<T>::Flatten(*param_out_);
    auto velocity_out = framework::EigenVector<T>::Flatten(*velocity_out_);

    auto param = framework::EigenVector<T>::Flatten(*param_);
    auto velocity = framework::EigenVector<T>::Flatten(*velocity_);
    auto grad = framework::EigenVector<T>::Flatten(*grad_);
    auto* lr = learning_rate_->data<T>();

    if (regularization_flag_ == RegularizationType::kL2DECAY) {
      velocity_out = velocity * mu_ + param * regularization_coeff_ + grad;
      if (use_nesterov_) {
        param_out =
            param -
            (param * regularization_coeff_ + grad + velocity_out * mu_) * lr[0];
      } else {
        param_out = param - lr[0] * velocity_out;
      }
    } else {
      velocity_out = velocity * mu_ + grad;
      if (use_nesterov_) {
        param_out = param - (grad + velocity_out * mu_) * lr[0];
      } else {
        param_out = param - lr[0] * velocity_out;
      }
    }
  }
};

template <typename T, typename UpdateMethod>
class DenseMomentumFunctor;

// NOTE(dzh) for performance.
// avoid if/else in inside kernel, implement GPU UseNesterov/NoNesterov as two
// functor.
template <typename T>
class DenseMomentumFunctor<T, UseNesterov> {
 private:
  const T* param_;
  const T* grad_;
  const T* velocity_;
  const T* lr_;
  const T mu_;
  const int64_t num_;
  T* param_out_;
  T* velocity_out_;
  RegularizationType regularization_flag_;
  const T regularization_coeff_;

 public:
  DenseMomentumFunctor(const T* param, const T* grad, const T* velocity,
                       const T* learning_rate, const T mu, const int64_t num,
                       RegularizationType regularization_flag,
                       const T regularization_coeff, T* param_out,
                       T* velocity_out)
      : param_(param),
        grad_(grad),
        velocity_(velocity),
        lr_(learning_rate),
        mu_(mu),
        num_(num),
        param_out_(param_out),
        velocity_out_(velocity_out),
        regularization_flag_(regularization_flag),
        regularization_coeff_(regularization_coeff) {}

  inline HOSTDEVICE void operator()(size_t i) const {
    // put memory access in register
    const T param = param_[i];
    T grad = grad_[i];
    const T lr = lr_[0];
    const T velocity = velocity_[i];

    grad = regularization_flag_ == RegularizationType::kL2DECAY
               ? grad + regularization_coeff_ * param
               : grad;

    T velocity_out = velocity * mu_ + grad;
    T param_out = param - (grad + velocity_out * mu_) * lr;
    // write reigster to memory
    velocity_out_[i] = velocity_out;
    param_out_[i] = param_out;
  }
};

template <typename T>
class DenseMomentumFunctor<T, NoNesterov> {
 private:
  const T* param_;
  const T* grad_;
  const T* velocity_;
  const T* lr_;
  const T mu_;
  const int64_t num_;
  T* param_out_;
  T* velocity_out_;
  RegularizationType regularization_flag_;
  const T regularization_coeff_;

 public:
  DenseMomentumFunctor(const T* param, const T* grad, const T* velocity,
                       const T* learning_rate, const T mu, const int64_t num,
                       RegularizationType regularization_flag,
                       const T regularization_coeff, T* param_out,
                       T* velocity_out)
      : param_(param),
        grad_(grad),
        velocity_(velocity),
        lr_(learning_rate),
        mu_(mu),
        num_(num),
        param_out_(param_out),
        velocity_out_(velocity_out),
        regularization_flag_(regularization_flag),
        regularization_coeff_(regularization_coeff) {}

  inline HOSTDEVICE void operator()(size_t i) const {
    // put memory access in register
    const T param = param_[i];
    T grad = grad_[i];
    const T lr = lr_[0];
    const T velocity = velocity_[i];

    grad = regularization_flag_ == RegularizationType::kL2DECAY
               ? grad + regularization_coeff_ * param
               : grad;

    T velocity_out = velocity * mu_ + grad;
    T param_out = param - lr * velocity_out;
    // write reigster to memory
    velocity_out_[i] = velocity_out;
    param_out_[i] = param_out;
  }
};

template <typename T, typename UpdateMethod>
class SparseMomentumFunctor;

template <typename T>
class SparseMomentumFunctor<T, UseNesterov> {
 private:
  const T* param_;
  const T* grad_;
  const T* velocity_;
  const T* lr_;
  const T mu_;
  const int64_t* rows_;
  const int64_t row_numel_;
  const int64_t row_height_;
  T* param_out_;
  T* velocity_out_;
  RegularizationType regularization_flag_;
  const T regularization_coeff_;

 public:
  SparseMomentumFunctor(const T* param, const T* grad, const T* velocity,
                        const T* lr, const T mu, const int64_t* rows,
                        int64_t row_numel, int64_t row_height,
                        RegularizationType regularization_flag,
                        const T regularization_coeff, T* param_out,
                        T* velocity_out)
      : param_(param),
        grad_(grad),
        velocity_(velocity),
        lr_(lr),
        mu_(mu),
        rows_(rows),
        row_numel_(row_numel),
        row_height_(row_height),
        param_out_(param_out),
        velocity_out_(velocity_out),
        regularization_flag_(regularization_flag),
        regularization_coeff_(regularization_coeff) {}

  inline HOSTDEVICE void operator()(size_t i) {
    auto row_idx =
        math::BinarySearch<int64_t>(rows_, row_height_, i / row_numel_);
    T grad = row_idx >= 0 ? grad_[row_idx * row_numel_ + i % row_numel_]
                          : static_cast<T>(0);
    // put memory access in register
    const T param = param_[i];
    const T lr = lr_[0];
    const T velocity = velocity_[i];

    grad = regularization_flag_ == RegularizationType::kL2DECAY
               ? grad + regularization_coeff_ * param
               : grad;

    T velocity_out = velocity * mu_ + grad;
    T param_out = param - (grad + velocity_out * mu_) * lr;
    // write reigster to memory
    velocity_out_[i] = velocity_out;
    param_out_[i] = param_out;
  }
};

template <typename T>
class SparseMomentumFunctor<T, NoNesterov> {
 private:
  const T* param_;
  const T* grad_;
  const T* velocity_;
  const T* lr_;
  const T mu_;
  const int64_t* rows_;
  const int64_t row_numel_;
  const int64_t row_height_;
  T* param_out_;
  T* velocity_out_;
  RegularizationType regularization_flag_;
  const T regularization_coeff_;

 public:
  SparseMomentumFunctor(const T* param, const T* grad, const T* velocity,
                        const T* lr, const T mu, const int64_t* rows,
                        int64_t row_numel, int64_t row_height,
                        RegularizationType regularization_flag,
                        const T regularization_coeff, T* param_out,
                        T* velocity_out)
      : param_(param),
        grad_(grad),
        velocity_(velocity),
        lr_(lr),
        mu_(mu),
        rows_(rows),
        row_numel_(row_numel),
        row_height_(row_height),
        param_out_(param_out),
        velocity_out_(velocity_out),
        regularization_flag_(regularization_flag),
        regularization_coeff_(regularization_coeff) {}

  inline HOSTDEVICE void operator()(size_t i) {
    auto row_idx =
        math::BinarySearch<int64_t>(rows_, row_height_, i / row_numel_);
    T grad = row_idx >= 0 ? grad_[row_idx * row_numel_ + i % row_numel_]
                          : static_cast<T>(0);
    // put memory access in register
    const T param = param_[i];
    const T lr = lr_[0];
    const T velocity = velocity_[i];

    grad = regularization_flag_ == RegularizationType::kL2DECAY
               ? grad + regularization_coeff_ * param
               : grad;

    T velocity_out = velocity * mu_ + grad;
    T param_out = param - velocity_out * lr;
    // write reigster to memory
    velocity_out_[i] = velocity_out;
    param_out_[i] = param_out;
  }
};

template <typename DeviceContext, typename T>
class MomentumOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    std::string regularization_method =
        ctx.Attr<std::string>("regularization_method");
    if (regularization_method != "" || !regularization_method.empty()) {
      PADDLE_ENFORCE_EQ("l2_decay", regularization_method,
                        platform::errors::InvalidArgument(
                            "if regularization_method is not null, "
                            "it should be l2_decay, but received %s",
                            regularization_method));
    }

    T regularization_coeff =
        static_cast<T>(ctx.Attr<float>("regularization_coeff"));
    RegularizationType regularization_flag{
        RegularizationType::kNONE};  // disable regularization
    if (regularization_method == "l2_decay") {
      regularization_flag = RegularizationType::kL2DECAY;
    }

    T mu = static_cast<T>(ctx.Attr<float>("mu"));
    bool use_nesterov = ctx.Attr<bool>("use_nesterov");

    auto learning_rate = ctx.Input<framework::Tensor>("LearningRate");
    auto param = ctx.Input<framework::Tensor>("Param");
    auto param_out = ctx.Output<framework::Tensor>("ParamOut");
    auto* velocity = ctx.Input<framework::Tensor>("Velocity");
    auto velocity_out = ctx.Output<framework::Tensor>("VelocityOut");

    param_out->mutable_data<T>(ctx.GetPlace());
    velocity_out->mutable_data<T>(ctx.GetPlace());

    auto* grad_var = ctx.InputVar("Grad");
    if (grad_var->IsType<framework::LoDTensor>()) {
      auto grad = ctx.Input<framework::Tensor>("Grad");
      if (platform::is_cpu_place(ctx.GetPlace())) {
        CPUDenseMomentumFunctor<T> functor(
            param, grad, velocity, learning_rate, mu, use_nesterov,
            regularization_flag, regularization_coeff, param_out, velocity_out);
        functor();
      } else if (platform::is_gpu_place(ctx.GetPlace())) {
        platform::ForRange<DeviceContext> for_range(
            static_cast<const DeviceContext&>(ctx.device_context()),
            param->numel());
        if (use_nesterov) {
          DenseMomentumFunctor<T, UseNesterov> functor(
              param->data<T>(), grad->data<T>(), velocity->data<T>(),
              learning_rate->data<T>(), mu, param->numel(), regularization_flag,
              regularization_coeff, param_out->mutable_data<T>(ctx.GetPlace()),
              velocity_out->mutable_data<T>(ctx.GetPlace()));
          for_range(functor);

        } else {
          DenseMomentumFunctor<T, NoNesterov> functor(
              param->data<T>(), grad->data<T>(), velocity->data<T>(),
              learning_rate->data<T>(), mu, param->numel(), regularization_flag,
              regularization_coeff, param_out->mutable_data<T>(ctx.GetPlace()),
              velocity_out->mutable_data<T>(ctx.GetPlace()));
          for_range(functor);
        }
      }

    } else if (grad_var->IsType<framework::SelectedRows>()) {
      // sparse update embedding with selectedrows
      auto grad = ctx.Input<framework::SelectedRows>("Grad");

      // sparse update maybe empty.
      if (grad->rows().size() == 0) {
        VLOG(3) << "Grad SelectedRows contains no data!";
        return;
      }

      framework::SelectedRows tmp_merged_grad;
      framework::SelectedRows* merged_grad = &tmp_merged_grad;
      math::scatter::MergeAdd<DeviceContext, T> merge_func;
      merge_func(ctx.template device_context<DeviceContext>(), *grad,
                 merged_grad);

      const int64_t* rows = merged_grad->rows().Data(ctx.GetPlace());
      int64_t row_numel =
          merged_grad->value().numel() / merged_grad->rows().size();
      platform::ForRange<DeviceContext> for_range(
          static_cast<const DeviceContext&>(ctx.device_context()),
          param->numel());
      if (use_nesterov) {
        SparseMomentumFunctor<T, UseNesterov> functor(
            param->data<T>(), merged_grad->value().data<T>(),
            velocity->data<T>(), learning_rate->data<T>(), mu, rows, row_numel,
            static_cast<int64_t>(merged_grad->rows().size()),
            regularization_flag, regularization_coeff,
            param_out->mutable_data<T>(ctx.GetPlace()),
            velocity_out->mutable_data<T>(ctx.GetPlace()));
        for_range(functor);

      } else {
        SparseMomentumFunctor<T, NoNesterov> functor(
            param->data<T>(), merged_grad->value().data<T>(),
            velocity->data<T>(), learning_rate->data<T>(), mu, rows, row_numel,
            static_cast<int64_t>(merged_grad->rows().size()),
            regularization_flag, regularization_coeff,
            param_out->mutable_data<T>(ctx.GetPlace()),
            velocity_out->mutable_data<T>(ctx.GetPlace()));
        for_range(functor);
      }
    } else {
      PADDLE_ENFORCE_EQ(false, true,
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
