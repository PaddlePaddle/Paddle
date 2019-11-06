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

class MomentumOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override;
};

class MomentumOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Param"),
                   "Input(param) of Momentum should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Grad"),
                   "Input(grad) of Momentum should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Velocity"),
                   "Input(velocity) of Momentum should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("LearningRate"),
                   "Input(LearningRate) of Momentum should not be null.");
    PADDLE_ENFORCE(
        ctx->GetInputsVarType("Param").front() ==
            framework::proto::VarType::LOD_TENSOR,
        "The input var's type should be LoDTensor, but the received is %s",
        ctx->Inputs("Param").front(), ctx->GetInputsVarType("Param").front());

    PADDLE_ENFORCE(ctx->HasOutput("ParamOut"),
                   "Output(ParamOut) of Momentum should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("VelocityOut"),
                   "Output(VelocityOut) of Momentum should not be null.");

    auto lr_dims = ctx->GetInputDim("LearningRate");
    PADDLE_ENFORCE_NE(framework::product(lr_dims), 0,
                      "Maybe the Input variable LearningRate has not "
                      "been initialized. You may need to confirm "
                      "if you put exe.run(startup_program) "
                      "after optimizer.minimize function.");
    PADDLE_ENFORCE_EQ(framework::product(lr_dims), 1,
                      "Learning_rate should be a scalar");

    auto param_dim = ctx->GetInputDim("Param");
    if (ctx->GetInputsVarType("Grad")[0] ==
        framework::proto::VarType::LOD_TENSOR) {
      PADDLE_ENFORCE_EQ(
          param_dim, ctx->GetInputDim("Grad"),
          "Param and Grad input of MomentumOp should have the same dimension.");
      PADDLE_ENFORCE_EQ(
          param_dim, ctx->GetInputDim("Velocity"),
          "Param and Velocity of MomentumOp should have the same dimension.");
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
  const Tensor* param;
  const Tensor* grad;
  const Tensor* velocity;
  const Tensor* learning_rate;
  const T mu;
  const T use_nesterov;
  Tensor* param_out;
  Tensor* velocity_out;

 public:
  CPUDenseMomentumFunctor(const Tensor* param, const Tensor* grad,
                          const Tensor* velocity, const Tensor* learning_rate,
                          const T mu, const bool use_nesterov,
                          Tensor* param_out, Tensor* velocity_out)
      : param(param),
        grad(grad),
        velocity(velocity),
        learning_rate(learning_rate),
        mu(mu),
        use_nesterov(use_nesterov),
        param_out(param_out),
        velocity_out(velocity_out) {}

  inline void operator()() {
    auto p_out = framework::EigenVector<T>::Flatten(*param_out);
    auto v_out = framework::EigenVector<T>::Flatten(*velocity_out);

    auto p = framework::EigenVector<T>::Flatten(*param);
    auto v = framework::EigenVector<T>::Flatten(*velocity);
    auto g = framework::EigenVector<T>::Flatten(*grad);
    auto* lr = learning_rate->data<T>();

    v_out = v * mu + g;
    if (use_nesterov) {
      p_out = p - (g + v_out * mu) * lr[0];
    } else {
      p_out = p - lr[0] * v_out;
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
  const T* p_;
  const T* g_;
  const T* v_;
  const T* lr_;
  const T mu_;
  const int64_t num_;
  T* p_out_;
  T* v_out_;

 public:
  DenseMomentumFunctor(const T* p, const T* g, const T* v,
                       const T* learning_rate, const T mu, const int64_t num,
                       T* p_out, T* v_out)
      : p_(p),
        g_(g),
        v_(v),
        lr_(learning_rate),
        mu_(mu),
        num_(num),
        p_out_(p_out),
        v_out_(v_out) {}
  inline HOSTDEVICE void operator()(size_t i) const {
    // put memory access in register
    const T p = p_[i];
    const T g = g_[i];
    const T lr = lr_[0];
    const T v = v_[i];
    T v_out = v * mu_ + g;
    T p_out = p - (g + v_out * mu_) * lr;
    // write reigster to memory
    v_out_[i] = v_out;
    p_out_[i] = p_out;
  }
};

template <typename T>
class DenseMomentumFunctor<T, NoNesterov> {
 private:
  const T* p_;
  const T* g_;
  const T* v_;
  const T* lr_;
  const T mu_;
  const int64_t num_;
  T* p_out_;
  T* v_out_;

 public:
  DenseMomentumFunctor(const T* p, const T* g, const T* v,
                       const T* learning_rate, const T mu, const int64_t num,
                       T* p_out, T* v_out)
      : p_(p),
        g_(g),
        v_(v),
        lr_(learning_rate),
        mu_(mu),
        num_(num),
        p_out_(p_out),
        v_out_(v_out) {}
  inline HOSTDEVICE void operator()(size_t i) const {
    // put memory access in register
    const T p = p_[i];
    const T g = g_[i];
    const T lr = lr_[0];
    const T v = v_[i];
    T v_out = v * mu_ + g;
    T p_out = p - lr * v_out;
    // write reigster to memory
    v_out_[i] = v_out;
    p_out_[i] = p_out;
  }
};

template <typename T, typename UpdateMethod>
class SparseMomentumFunctor;

template <typename T>
class SparseMomentumFunctor<T, UseNesterov> {
 private:
  const T* p_;
  const T* g_;
  const T* v_;
  const T* lr_;
  const T mu_;
  const int64_t* rows_;
  const int64_t row_numel_;
  const int64_t row_height_;
  T* p_out_;
  T* v_out_;

 public:
  SparseMomentumFunctor(const T* p, const T* g, const T* v, const T* lr,
                        const T mu, const int64_t* rows, int64_t row_numel,
                        int64_t row_height, T* p_out, T* v_out)
      : p_(p),
        g_(g),
        v_(v),
        lr_(lr),
        mu_(mu),
        rows_(rows),
        row_numel_(row_numel),
        row_height_(row_height),
        p_out_(p_out),
        v_out_(v_out) {}

  inline HOSTDEVICE void operator()(size_t i) {
    auto row_idx =
        math::BinarySearch<int64_t>(rows_, row_height_, i / row_numel_);
    T g = row_idx >= 0 ? g_[row_idx * row_numel_ + i % row_numel_]
                       : static_cast<T>(0);
    // put memory access in register
    const T p = p_[i];
    const T lr = lr_[0];
    const T v = v_[i];
    T v_out = v * mu_ + g;
    T p_out = p - (g + v_out * mu_) * lr;
    // write reigster to memory
    v_out_[i] = v_out;
    p_out_[i] = p_out;
  }
};

template <typename T>
class SparseMomentumFunctor<T, NoNesterov> {
 private:
  const T* p_;
  const T* g_;
  const T* v_;
  const T* lr_;
  const T mu_;
  const int64_t* rows_;
  const int64_t row_numel_;
  const int64_t row_height_;
  T* p_out_;
  T* v_out_;

 public:
  SparseMomentumFunctor(const T* p, const T* g, const T* v, const T* lr,
                        const T mu, const int64_t* rows, int64_t row_numel,
                        int64_t row_height, T* p_out, T* v_out)
      : p_(p),
        g_(g),
        v_(v),
        lr_(lr),
        mu_(mu),
        rows_(rows),
        row_numel_(row_numel),
        row_height_(row_height),
        p_out_(p_out),
        v_out_(v_out) {}

  inline HOSTDEVICE void operator()(size_t i) {
    auto row_idx =
        math::BinarySearch<int64_t>(rows_, row_height_, i / row_numel_);
    T g = row_idx >= 0 ? g_[row_idx * row_numel_ + i % row_numel_]
                       : static_cast<T>(0);
    // put memory access in register
    const T p = p_[i];
    const T lr = lr_[0];
    const T v = v_[i];
    T v_out = v * mu_ + g;
    T p_out = p - v_out * lr;
    // write reigster to memory
    v_out_[i] = v_out;
    p_out_[i] = p_out;
  }
};

template <typename DeviceContext, typename T>
class MomentumOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
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
        CPUDenseMomentumFunctor<T> functor(param, grad, velocity, learning_rate,
                                           mu, use_nesterov, param_out,
                                           velocity_out);
        functor();
      } else if (platform::is_gpu_place(ctx.GetPlace())) {
        platform::ForRange<DeviceContext> for_range(
            static_cast<const DeviceContext&>(ctx.device_context()),
            param->numel());
        if (use_nesterov) {
          DenseMomentumFunctor<T, UseNesterov> functor(
              param->data<T>(), grad->data<T>(), velocity->data<T>(),
              learning_rate->data<T>(), mu, param->numel(),
              param_out->mutable_data<T>(ctx.GetPlace()),
              velocity_out->mutable_data<T>(ctx.GetPlace()));
          for_range(functor);

        } else {
          DenseMomentumFunctor<T, NoNesterov> functor(
              param->data<T>(), grad->data<T>(), velocity->data<T>(),
              learning_rate->data<T>(), mu, param->numel(),
              param_out->mutable_data<T>(ctx.GetPlace()),
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
            param_out->mutable_data<T>(ctx.GetPlace()),
            velocity_out->mutable_data<T>(ctx.GetPlace()));
        for_range(functor);

      } else {
        SparseMomentumFunctor<T, NoNesterov> functor(
            param->data<T>(), merged_grad->value().data<T>(),
            velocity->data<T>(), learning_rate->data<T>(), mu, rows, row_numel,
            static_cast<int64_t>(merged_grad->rows().size()),
            param_out->mutable_data<T>(ctx.GetPlace()),
            velocity_out->mutable_data<T>(ctx.GetPlace()));
        for_range(functor);
      }
    } else {
      PADDLE_THROW(
          string::Sprintf("MomentumOp only supports LoDTensor or SelectedRows "
                          "gradient, but the received Variable Type is %s",
                          framework::ToTypeName(grad_var->Type())));
    }
  }
};

}  // namespace operators
}  // namespace paddle
