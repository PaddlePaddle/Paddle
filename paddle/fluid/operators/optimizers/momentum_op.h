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
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

using framework::Tensor;
using framework::SelectedRows;
struct NoNesterov;
struct UseNesterov;

namespace details {

template <typename T>
class MPTypeTrait {
 public:
  using Type = T;
};
template <>
class MPTypeTrait<platform::float16> {
 public:
  using Type = float;
};

template <typename T, typename MT>
struct CPUDenseUpdater {
  template <typename G>
  void operator()(const Tensor& param, const Tensor& velocity, const MT& mu,
                  const MT& lr, const bool use_nesterov, G&& g,
                  Tensor* param_out, Tensor* velocity_out) const {
    auto p_out = framework::EigenVector<MT>::Flatten(*param_out);
    auto v_out = framework::EigenVector<MT>::Flatten(*velocity_out);

    auto p = framework::EigenVector<MT>::Flatten(param);
    auto v = framework::EigenVector<MT>::Flatten(velocity);
    v_out = v * mu + g;
    if (use_nesterov) {
      p_out = p - (g + v_out * mu) * lr;
    } else {
      p_out = p - lr * v_out;
    }
  }
};

}  // namespace details

template <typename T>
using LRType = typename details::MPTypeTrait<T>::Type;

enum class RegularizationFlag {
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

template <typename T, typename MT>
class CPUDenseMomentumFunctor {
 public:
  void operator()(const Tensor* param, const Tensor* grad,
                  const Tensor* velocity, const Tensor* learning_rate,
                  const MT mu, const bool use_nesterov,
                  const RegularizationFlag regularization_flag,
                  const MT regularization_coeff, Tensor* param_out,
                  Tensor* velocity_out) {
    auto g = framework::EigenVector<MT>::Flatten(*grad);
    auto* lr = learning_rate->data<LRType<MT>>();

    details::CPUDenseUpdater<T, MT> updater;
    if (regularization_flag == RegularizationFlag::kL2DECAY) {
      auto p = framework::EigenVector<MT>::Flatten(*param);
      updater(*param, *velocity, mu, static_cast<MT>(lr[0]), use_nesterov,
              p * regularization_coeff + g, param_out, velocity_out);
    } else {
      updater(*param, *velocity, mu, static_cast<MT>(lr[0]), use_nesterov, g,
              param_out, velocity_out);
    }
  }
};

template <typename T, typename MT, typename UpdateMethod>
class DenseMomentumFunctor;

// NOTE(dzh) for performance.
// avoid if/else in inside kernel, implement GPU UseNesterov/NoNesterov as two
// functor.
template <typename T, typename MT>
class DenseMomentumFunctor<T, MT, UseNesterov> {
 private:
  const T* p_;
  const T* g_;
  const MT* v_;
  const LRType<MT>* lr_;
  const MT* mp_;
  const MT mu_;
  const int64_t num_;
  T* p_out_;
  MT* v_out_;
  MT* mp_out_;
  const RegularizationFlag regularization_flag;
  const MT regularization_coeff;

 public:
  DenseMomentumFunctor(const T* p, const T* g, const MT* v,
                       const LRType<MT>* learning_rate, const MT* mp,
                       const MT mu, const int64_t num,
                       const RegularizationFlag regularization_flag,
                       const MT regularization_coeff, T* p_out, MT* v_out,
                       MT* mp_out)
      : p_(p),
        g_(g),
        v_(v),
        lr_(learning_rate),
        mp_(mp),
        mu_(mu),
        num_(num),
        p_out_(p_out),
        v_out_(v_out),
        mp_out_(mp_out),
        regularization_flag(regularization_flag),
        regularization_coeff(regularization_coeff) {}
  inline HOSTDEVICE void operator()(size_t i) const {
    // put memory access in register
    const MT p = mp_ ? mp_[i] : static_cast<MT>(p_[i]);
    MT g = static_cast<MT>(g_[i]);
    const MT lr = static_cast<MT>(lr_[0]);
    const MT v = v_[i];

    g = regularization_flag == RegularizationFlag::kL2DECAY
            ? g + regularization_coeff * p
            : g;

    MT v_out = v * mu_ + g;
    MT p_out = p - (g + v_out * mu_) * lr;
    // write reigster to memory
    v_out_[i] = v_out;
    p_out_[i] = static_cast<T>(p_out);
    if (mp_out_) {
      mp_out_[i] = p_out;
    }
  }
};

template <typename T, typename MT>
class DenseMomentumFunctor<T, MT, NoNesterov> {
 private:
  const T* p_;
  const T* g_;
  const MT* v_;
  const LRType<MT>* lr_;
  const MT* mp_;
  const MT mu_;
  const int64_t num_;
  T* p_out_;
  MT* v_out_;
  MT* mp_out_;
  const RegularizationFlag regularization_flag;
  const MT regularization_coeff;

 public:
  DenseMomentumFunctor(const T* p, const T* g, const MT* v,
                       const LRType<MT>* learning_rate, const MT* mp,
                       const MT mu, const int64_t num,
                       const RegularizationFlag regularization_flag,
                       const MT regularization_coeff, T* p_out, MT* v_out,
                       MT* mp_out)
      : p_(p),
        g_(g),
        v_(v),
        lr_(learning_rate),
        mp_(mp),
        mu_(mu),
        num_(num),
        p_out_(p_out),
        v_out_(v_out),
        mp_out_(mp_out),
        regularization_flag(regularization_flag),
        regularization_coeff(regularization_coeff) {}
  inline HOSTDEVICE void operator()(size_t i) const {
    // put memory access in register
    const MT p = mp_ ? mp_[i] : static_cast<MT>(p_[i]);
    MT g = static_cast<MT>(g_[i]);
    const MT lr = static_cast<MT>(lr_[0]);
    const MT v = v_[i];

    g = regularization_flag == RegularizationFlag::kL2DECAY
            ? g + regularization_coeff * p
            : g;

    MT v_out = v * mu_ + g;
    MT p_out = p - lr * v_out;
    // write reigster to memory
    v_out_[i] = v_out;
    p_out_[i] = static_cast<T>(p_out);
    if (mp_out_) {
      mp_out_[i] = p_out;
    }
  }
};

template <typename T, typename MT, typename UpdateMethod>
class SparseMomentumFunctor;

template <typename T, typename MT>
class SparseMomentumFunctor<T, MT, UseNesterov> {
 private:
  const T* p_;
  const T* g_;
  const MT* v_;
  const LRType<MT>* lr_;
  const MT* mp_;
  const MT mu_;
  const int64_t* rows_;
  const int64_t row_numel_;
  const int64_t row_height_;
  T* p_out_;
  MT* v_out_;
  MT* mp_out_;
  const RegularizationFlag regularization_flag;
  const MT regularization_coeff;

 public:
  SparseMomentumFunctor(const T* p, const T* g, const MT* v,
                        const LRType<MT>* lr, const MT* mp, const MT mu,
                        const int64_t* rows, int64_t row_numel,
                        int64_t row_height,
                        const RegularizationFlag regularization_flag,
                        const MT regularization_coeff, T* p_out, MT* v_out,
                        MT* mp_out)
      : p_(p),
        g_(g),
        v_(v),
        lr_(lr),
        mp_(mp),
        mu_(mu),
        rows_(rows),
        row_numel_(row_numel),
        row_height_(row_height),
        p_out_(p_out),
        v_out_(v_out),
        mp_out_(mp_out),
        regularization_flag(regularization_flag),
        regularization_coeff(regularization_coeff) {}

  inline HOSTDEVICE void operator()(size_t i) {
    auto row_idx =
        math::BinarySearch<int64_t>(rows_, row_height_, i / row_numel_);
    MT g = row_idx >= 0
               ? static_cast<MT>(g_[row_idx * row_numel_ + i % row_numel_])
               : static_cast<MT>(0);
    // put memory access in register
    const MT p = mp_ ? mp_[i] : static_cast<MT>(p_[i]);
    const MT lr = static_cast<MT>(lr_[0]);
    const MT v = v_[i];

    g = regularization_flag == RegularizationFlag::kL2DECAY
            ? g + regularization_coeff * p
            : g;

    MT v_out = v * mu_ + g;
    MT p_out = p - (g + v_out * mu_) * lr;
    // write reigster to memory
    v_out_[i] = v_out;
    p_out_[i] = static_cast<T>(p_out);
    if (mp_out_) {
      mp_out_[i] = p_out;
    }
  }
};

template <typename T, typename MT>
class SparseMomentumFunctor<T, MT, NoNesterov> {
 private:
  const T* p_;
  const T* g_;
  const MT* v_;
  const LRType<MT>* lr_;
  const MT* mp_;
  const MT mu_;
  const int64_t* rows_;
  const int64_t row_numel_;
  const int64_t row_height_;
  T* p_out_;
  MT* v_out_;
  MT* mp_out_;
  const RegularizationFlag regularization_flag;
  const MT regularization_coeff;

 public:
  SparseMomentumFunctor(const T* p, const T* g, const MT* v,
                        const LRType<MT>* lr, const MT* mp, const MT mu,
                        const int64_t* rows, int64_t row_numel,
                        int64_t row_height,
                        const RegularizationFlag regularization_flag,
                        const MT regularization_coeff, T* p_out, MT* v_out,
                        MT* mp_out)
      : p_(p),
        g_(g),
        v_(v),
        lr_(lr),
        mp_(mp),
        mu_(mu),
        rows_(rows),
        row_numel_(row_numel),
        row_height_(row_height),
        p_out_(p_out),
        v_out_(v_out),
        mp_out_(mp_out),
        regularization_flag(regularization_flag),
        regularization_coeff(regularization_coeff) {}

  inline HOSTDEVICE void operator()(size_t i) {
    auto row_idx =
        math::BinarySearch<int64_t>(rows_, row_height_, i / row_numel_);
    MT g = row_idx >= 0
               ? static_cast<MT>(g_[row_idx * row_numel_ + i % row_numel_])
               : static_cast<MT>(0);
    // put memory access in register
    const MT p = mp_ ? mp_[i] : static_cast<MT>(p_[i]);
    const MT lr = static_cast<MT>(lr_[0]);
    const MT v = v_[i];

    g = regularization_flag == RegularizationFlag::kL2DECAY
            ? g + regularization_coeff * p
            : g;

    MT v_out = v * mu_ + g;
    MT p_out = p - v_out * lr;
    // write reigster to memory
    v_out_[i] = v_out;
    p_out_[i] = static_cast<T>(p_out);
    if (mp_out_) {
      mp_out_[i] = p_out;
    }
  }
};

template <typename DeviceContext, typename T>
class MomentumOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using MPDType = typename details::MPTypeTrait<T>::Type;
    const bool multi_precision = ctx.Attr<bool>("multi_precision");
    if (multi_precision) {
      InnerCompute<MPDType>(ctx, multi_precision);
    } else {
      InnerCompute<T>(ctx, multi_precision);
    }
  }

 private:
  template <typename MT>
  void InnerCompute(const framework::ExecutionContext& ctx,
                    const bool multi_precision) const {
    std::string regularization_method =
        ctx.Attr<std::string>("regularization_method");
    MT regularization_coeff =
        static_cast<MT>(ctx.Attr<float>("regularization_coeff"));
    RegularizationFlag regularization_flag{
        RegularizationFlag::kNONE};  // disable regularization
    if (regularization_method == "l2_decay") {
      regularization_flag = RegularizationFlag::kL2DECAY;
    }

    MT mu = static_cast<MT>(ctx.Attr<float>("mu"));
    bool use_nesterov = ctx.Attr<bool>("use_nesterov");

    auto learning_rate = ctx.Input<framework::Tensor>("LearningRate");
    auto param = ctx.Input<framework::Tensor>("Param");
    auto param_out = ctx.Output<framework::Tensor>("ParamOut");
    auto velocity = ctx.Input<framework::Tensor>("Velocity");
    auto velocity_out = ctx.Output<framework::Tensor>("VelocityOut");
    const framework::Tensor* master_param = nullptr;
    framework::Tensor* master_param_out = nullptr;
    if (multi_precision) {
      bool has_master =
          ctx.HasInput("MasterParam") && ctx.HasOutput("MasterParamOut");
      PADDLE_ENFORCE_EQ(has_master, true,
                        platform::errors::InvalidArgument(
                            "The Input(MasterParam) and Output(MasterParamOut) "
                            "should not be null when "
                            "the attr `multi_precision` is true"));
      master_param = ctx.Input<framework::Tensor>("MasterParam");
      master_param_out = ctx.Output<framework::Tensor>("MasterParamOut");
    }

    param_out->mutable_data<T>(ctx.GetPlace());
    velocity_out->mutable_data<MT>(ctx.GetPlace());
    const MT* master_in_data =
        multi_precision ? master_param->data<MT>() : nullptr;
    MT* master_out_data =
        multi_precision ? master_param_out->mutable_data<MT>(ctx.GetPlace())
                        : nullptr;

    auto* grad_var = ctx.InputVar("Grad");
    if (grad_var->IsType<framework::LoDTensor>()) {
      auto grad = ctx.Input<framework::Tensor>("Grad");
      if (platform::is_cpu_place(ctx.GetPlace())) {
        CPUDenseMomentumFunctor<T, MT> functor;
        functor(param, grad, velocity, learning_rate, mu, use_nesterov,
                regularization_flag, regularization_coeff, param_out,
                velocity_out);
      } else if (platform::is_gpu_place(ctx.GetPlace())) {
        platform::ForRange<DeviceContext> for_range(
            static_cast<const DeviceContext&>(ctx.device_context()),
            param->numel());
        if (use_nesterov) {
          DenseMomentumFunctor<T, MT, UseNesterov> functor(
              param->data<T>(), grad->data<T>(), velocity->data<MT>(),
              learning_rate->data<LRType<MT>>(), master_in_data, mu,
              param->numel(), regularization_flag, regularization_coeff,
              param_out->mutable_data<T>(ctx.GetPlace()),
              velocity_out->mutable_data<MT>(ctx.GetPlace()), master_out_data);
          for_range(functor);

        } else {
          DenseMomentumFunctor<T, MT, NoNesterov> functor(
              param->data<T>(), grad->data<T>(), velocity->data<MT>(),
              learning_rate->data<LRType<MT>>(), master_in_data, mu,
              param->numel(), regularization_flag, regularization_coeff,
              param_out->mutable_data<T>(ctx.GetPlace()),
              velocity_out->mutable_data<MT>(ctx.GetPlace()), master_out_data);
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
        SparseMomentumFunctor<T, MT, UseNesterov> functor(
            param->data<T>(), merged_grad->value().data<T>(),
            velocity->data<MT>(), learning_rate->data<LRType<MT>>(),
            master_in_data, mu, rows, row_numel,
            static_cast<int64_t>(merged_grad->rows().size()),
            regularization_flag, regularization_coeff,
            param_out->mutable_data<T>(ctx.GetPlace()),
            velocity_out->mutable_data<MT>(ctx.GetPlace()), master_out_data);
        for_range(functor);

      } else {
        SparseMomentumFunctor<T, MT, NoNesterov> functor(
            param->data<T>(), merged_grad->value().data<T>(),
            velocity->data<MT>(), learning_rate->data<LRType<MT>>(),
            master_in_data, mu, rows, row_numel,
            static_cast<int64_t>(merged_grad->rows().size()),
            regularization_flag, regularization_coeff,
            param_out->mutable_data<T>(ctx.GetPlace()),
            velocity_out->mutable_data<MT>(ctx.GetPlace()), master_out_data);
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
