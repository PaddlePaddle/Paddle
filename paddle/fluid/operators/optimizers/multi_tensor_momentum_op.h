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

#pragma once
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/optimizers/momentum_op.h"
// #include <memory>
// #include <string>
// #include "paddle/fluid/framework/eigen.h"
// #include "paddle/fluid/framework/op_registry.h"
// #include "paddle/fluid/operators/amp/fp16_type_traits.h"
// #include "paddle/fluid/operators/math/algorithm.h"
// #include "paddle/fluid/operators/math/selected_rows_functor.h"
// #include "paddle/fluid/platform/float16.h"
// #include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

// using framework::Tensor;
// using framework::SelectedRows;
// struct NoNesterov;
// struct UseNesterov;

// namespace detail {

// template <typename T>
// struct CPUDenseUpdater1 {
//   template <typename G>
//   void operator()(const Tensor& param, const Tensor& velocity, const T& mu,
//                   const T& lr, const bool use_nesterov, G&& grad,
//                   Tensor* param_out, Tensor* velocity_out) const {
//     auto param_out_vec = framework::EigenVector<T>::Flatten(*param_out);
//     auto velocity_out_vec =
//     framework::EigenVector<T>::Flatten(*velocity_out);

//     auto param_vec = framework::EigenVector<T>::Flatten(param);
//     auto velocity_vec = framework::EigenVector<T>::Flatten(velocity);
//     velocity_out_vec = velocity_vec * mu + grad;
//     if (use_nesterov) {
//       param_out_vec = param_vec - (grad + velocity_out_vec * mu) * lr;
//     } else {
//       param_out_vec = param_vec - lr * velocity_out_vec;
//     }
//   }
// };

// }  // namespace detail

// template <typename T>
// using MultiPrecisionType1 = typename details::MPTypeTrait<T>::Type;

// enum class RegularizationType {
//   kNONE = 0,
//   kL1DECAY = 1,  // do not need support right now
//   kL2DECAY = 2,
// };

// template <typename T>
// class CPUDenseMomentumFunctor1 {
//  public:
//   void operator()(const Tensor* param, const Tensor* grad,
//                   const Tensor* velocity, const Tensor* learning_rate,
//                   const T mu, const bool use_nesterov,
//                   const RegularizationType regularization_flag,
//                   const T regularization_coeff, Tensor* param_out,
//                   Tensor* velocity_out) {
//     auto grad_vec = framework::EigenVector<T>::Flatten(*grad);
//     auto* lr = learning_rate->data<MultiPrecisionType1<T>>();

//     detail::CPUDenseUpdater1<T> updater;
//     if (regularization_flag == RegularizationType::kL2DECAY) {
//       auto param_vec = framework::EigenVector<T>::Flatten(*param);
//       updater(*param, *velocity, mu, static_cast<T>(lr[0]), use_nesterov,
//               param_vec * regularization_coeff + grad_vec, param_out,
//               velocity_out);
//     } else {
//       updater(*param, *velocity, mu, static_cast<T>(lr[0]), use_nesterov,
//               grad_vec, param_out, velocity_out);
//     }
//   }
// };

// template <typename T, typename MT, RegularizationType kRegType,
//           typename UpdateMethod>
// class DenseMomentumFunctor1;

// // NOTE(dzh) for performance.
// // avoid if/else in inside kernel, implement GPU UseNesterov/NoNesterov as
// two
// // functor.
// template <typename T, typename MT, RegularizationType kRegType>
// class DenseMomentumFunctor1<T, MT, kRegType, UseNesterov> {
//  private:
//   const T* param_;
//   const T* grad_;
//   const MT* velocity_;
//   const MultiPrecisionType1<MT>* lr_;
//   const MT* master_param_;
//   const MT mu_;
//   const MT rescale_grad_;
//   const int64_t num_;
//   T* param_out_;
//   MT* velocity_out_;
//   MT* master_param_out_;
//   const MT regularization_coeff_;

//  public:
//   DenseMomentumFunctor1(const T* param, const T* grad, const MT* velocity,
//                        const MultiPrecisionType1<MT>* learning_rate,
//                        const MT* master_param, const MT mu,
//                        const MT rescale_grad, const int64_t num,
//                        const MT regularization_coeff, T* param_out,
//                        MT* velocity_out, MT* master_param_out)
//       : param_(param),
//         grad_(grad),
//         velocity_(velocity),
//         lr_(learning_rate),
//         master_param_(master_param),
//         mu_(mu),
//         rescale_grad_(rescale_grad),
//         num_(num),
//         param_out_(param_out),
//         velocity_out_(velocity_out),
//         master_param_out_(master_param_out),
//         regularization_coeff_(regularization_coeff) {}
//   inline HOSTDEVICE void operator()(size_t i) const {
//     // put memory access in register
//     const MT param =
//         master_param_ ? master_param_[i] : static_cast<MT>(param_[i]);
//     MT grad = static_cast<MT>(grad_[i]) * rescale_grad_;
//     const MT lr = static_cast<MT>(lr_[0]);
//     const MT velocity = velocity_[i];

//     if (kRegType == RegularizationType::kL2DECAY) {
//       grad += regularization_coeff_ * param;
//     }

//     MT velocity_out = velocity * mu_ + grad;
//     MT param_out = param - (grad + velocity_out * mu_) * lr;
//     // write reigster to memory
//     velocity_out_[i] = velocity_out;
//     param_out_[i] = static_cast<T>(param_out);
//     if (master_param_out_) {
//       master_param_out_[i] = param_out;
//     }
//   }
// };

// template <typename T, typename MT, RegularizationType kRegType>
// class DenseMomentumFunctor1<T, MT, kRegType, NoNesterov> {
//  private:
//   const T* param_;
//   const T* grad_;
//   const MT* velocity_;
//   const MultiPrecisionType1<MT>* lr_;
//   const MT* master_param_;
//   const MT mu_;
//   const MT rescale_grad_;
//   const int64_t num_;
//   T* param_out_;
//   MT* velocity_out_;
//   MT* master_param_out_;
//   const MT regularization_coeff_;

//  public:
//   DenseMomentumFunctor1(const T* param, const T* grad, const MT* velocity,
//                        const MultiPrecisionType1<MT>* learning_rate,
//                        const MT* master_param, const MT mu,
//                        const MT rescale_grad, const int64_t num,
//                        const MT regularization_coeff, T* param_out,
//                        MT* velocity_out, MT* master_param_out)
//       : param_(param),
//         grad_(grad),
//         velocity_(velocity),
//         lr_(learning_rate),
//         master_param_(master_param),
//         mu_(mu),
//         rescale_grad_(rescale_grad),
//         num_(num),
//         param_out_(param_out),
//         velocity_out_(velocity_out),
//         master_param_out_(master_param_out),
//         regularization_coeff_(regularization_coeff) {}
//   inline HOSTDEVICE void operator()(size_t i) const {
//     // put memory access in register
//     const MT param =
//         master_param_ ? master_param_[i] : static_cast<MT>(param_[i]);
//     MT grad = static_cast<MT>(grad_[i]) * rescale_grad_;
//     const MT lr = static_cast<MT>(lr_[0]);
//     const MT velocity = velocity_[i];

//     if (kRegType == RegularizationType::kL2DECAY) {
//       grad += regularization_coeff_ * param;
//     }

//     MT velocity_out = velocity * mu_ + grad;
//     MT param_out = param - lr * velocity_out;
//     // write reigster to memory
//     velocity_out_[i] = velocity_out;
//     param_out_[i] = static_cast<T>(param_out);
//     if (master_param_out_) {
//       master_param_out_[i] = param_out;
//     }
//   }
// };

// template <typename T, typename MT, typename UpdateMethod>
// class SparseMomentumFunctor1;

// template <typename T, typename MT>
// class SparseMomentumFunctor1<T, MT, UseNesterov> {
//  private:
//   const T* param_;
//   const T* grad_;
//   const MT* velocity_;
//   const MultiPrecisionType1<MT>* lr_;
//   const MT* master_param_;
//   const MT mu_;
//   const MT rescale_grad_;
//   const int64_t* rows_;
//   const int64_t row_numel_;
//   const int64_t row_height_;
//   T* param_out_;
//   MT* velocity_out_;
//   MT* master_param_out_;
//   const RegularizationType regularization_flag_;
//   const MT regularization_coeff_;

//  public:
//   SparseMomentumFunctor1(const T* param, const T* grad, const MT* velocity,
//                         const MultiPrecisionType1<MT>* lr,
//                         const MT* master_param, const MT mu,
//                         const MT rescale_grad, const int64_t* rows,
//                         int64_t row_numel, int64_t row_height,
//                         const RegularizationType regularization_flag,
//                         const MT regularization_coeff, T* param_out,
//                         MT* velocity_out, MT* master_param_out)
//       : param_(param),
//         grad_(grad),
//         velocity_(velocity),
//         lr_(lr),
//         master_param_(master_param),
//         mu_(mu),
//         rescale_grad_(rescale_grad),
//         rows_(rows),
//         row_numel_(row_numel),
//         row_height_(row_height),
//         param_out_(param_out),
//         velocity_out_(velocity_out),
//         master_param_out_(master_param_out),
//         regularization_flag_(regularization_flag),
//         regularization_coeff_(regularization_coeff) {}

//   inline HOSTDEVICE void operator()(size_t i) {
//     auto row_idx =
//         math::BinarySearch<int64_t>(rows_, row_height_, i / row_numel_);
//     MT grad =
//         row_idx >= 0
//             ? static_cast<MT>(grad_[row_idx * row_numel_ + i % row_numel_]) *
//                   rescale_grad_
//             : static_cast<MT>(0);
//     // put memory access in register
//     const MT param =
//         master_param_ ? master_param_[i] : static_cast<MT>(param_[i]);
//     const MT lr = static_cast<MT>(lr_[0]);
//     const MT velocity = velocity_[i];

//     grad = regularization_flag_ == RegularizationType::kL2DECAY
//                ? grad + regularization_coeff_ * param
//                : grad;

//     MT velocity_out = velocity * mu_ + grad;
//     MT param_out = param - (grad + velocity_out * mu_) * lr;
//     // write reigster to memory
//     velocity_out_[i] = velocity_out;
//     param_out_[i] = static_cast<T>(param_out);
//     if (master_param_out_) {
//       master_param_out_[i] = param_out;
//     }
//   }
// };

// template <typename T, typename MT>
// class SparseMomentumFunctor1<T, MT, NoNesterov> {
//  private:
//   const T* param_;
//   const T* grad_;
//   const MT* velocity_;
//   const MultiPrecisionType1<MT>* lr_;
//   const MT* master_param_;
//   const MT mu_;
//   const MT rescale_grad_;
//   const int64_t* rows_;
//   const int64_t row_numel_;
//   const int64_t row_height_;
//   T* param_out_;
//   MT* velocity_out_;
//   MT* master_param_out_;
//   const RegularizationType regularization_flag_;
//   const MT regularization_coeff_;

//  public:
//   SparseMomentumFunctor1(const T* param, const T* grad, const MT* velocity,
//                         const MultiPrecisionType1<MT>* lr,
//                         const MT* master_param, const MT mu,
//                         const MT rescale_grad, const int64_t* rows,
//                         int64_t row_numel, int64_t row_height,
//                         const RegularizationType regularization_flag,
//                         const MT regularization_coeff, T* param_out,
//                         MT* velocity_out, MT* master_param_out)
//       : param_(param),
//         grad_(grad),
//         velocity_(velocity),
//         lr_(lr),
//         master_param_(master_param),
//         mu_(mu),
//         rescale_grad_(rescale_grad),
//         rows_(rows),
//         row_numel_(row_numel),
//         row_height_(row_height),
//         param_out_(param_out),
//         velocity_out_(velocity_out),
//         master_param_out_(master_param_out),
//         regularization_flag_(regularization_flag),
//         regularization_coeff_(regularization_coeff) {}

//   inline HOSTDEVICE void operator()(size_t i) {
//     auto row_idx =
//         math::BinarySearch<int64_t>(rows_, row_height_, i / row_numel_);
//     MT grad =
//         row_idx >= 0
//             ? static_cast<MT>(grad_[row_idx * row_numel_ + i % row_numel_]) *
//                   rescale_grad_
//             : static_cast<MT>(0);
//     // put memory access in register
//     const MT param =
//         master_param_ ? master_param_[i] : static_cast<MT>(param_[i]);
//     const MT lr = static_cast<MT>(lr_[0]);
//     const MT velocity = velocity_[i];

//     grad = regularization_flag_ == RegularizationType::kL2DECAY
//                ? grad + regularization_coeff_ * param
//                : grad;

//     MT velocity_out = velocity * mu_ + grad;
//     MT param_out = param - velocity_out * lr;
//     // write reigster to memory
//     velocity_out_[i] = velocity_out;
//     param_out_[i] = static_cast<T>(param_out);
//     if (master_param_out_) {
//       master_param_out_[i] = param_out;
//     }
//   }
// };

template <typename DeviceContext, typename T>
class MTMomentumOpKernel : public framework::OpKernel<T> {
  using MPDType = MultiPrecisionType<T>;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
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
    // attr ctx
    std::vector<std::string> regularization_methods =
        ctx.Attr<std::vector<std::string>>("regularization_methods");
    std::vector<float> regularization_coeffs =
        ctx.Attr<std::vector<float>>("regularization_coeffs");
    MT mu = static_cast<MT>(ctx.Attr<float>("mu"));
    MT rescale_grad = static_cast<MT>(ctx.Attr<float>("rescale_grad"));
    bool use_nesterov = ctx.Attr<bool>("use_nesterov");

    VLOG(10) << "get all attr";

    // ipt/opt ctx
    auto params = ctx.MultiInput<framework::Tensor>("Params");
    auto param_outs = ctx.MultiOutput<framework::Tensor>("ParamOuts");
    auto velocitys = ctx.MultiInput<framework::Tensor>("Velocitys");
    auto velocity_outs = ctx.MultiOutput<framework::Tensor>("VelocityOuts");
    auto learning_rates = ctx.MultiInput<framework::Tensor>("LearningRates");
    VLOG(10) << "get master_params";
    std::vector<const framework::Tensor*> master_params(params.size(), nullptr);
    std::vector<framework::Tensor*> master_param_outs(params.size(), nullptr);
    if (multi_precision) {
      bool has_master =
          ctx.HasInput("MasterParams") && ctx.HasOutput("MasterParamOuts");
      PADDLE_ENFORCE_EQ(
          has_master, true,
          platform::errors::InvalidArgument(
              "The Input(MasterParams) and Output(MasterParamOuts) "
              "should not be null when "
              "the attr `multi_precision` is true"));
      master_params = ctx.MultiInput<framework::Tensor>("MasterParams");
      master_param_outs = ctx.MultiOutput<framework::Tensor>("MasterParamOuts");
    }
    VLOG(10) << "get input / output";

    auto grad_vars = ctx.MultiInputVar("Grads");
    if (grad_vars[0]->IsType<framework::LoDTensor>()) {
      auto grads = ctx.MultiInput<framework::Tensor>("Grads");
      // multi_tensor size
      size_t multi_tensor_size = params.size();
      for (size_t idx = 0; idx < multi_tensor_size; idx++) {
        std::string regularization_method = regularization_methods[idx];
        MT regularization_coeff = static_cast<MT>(regularization_coeffs[idx]);
        RegularizationType regularization_flag{
            RegularizationType::kNONE};  // disable regularization
        if (regularization_method == "l2_decay") {
          regularization_flag = RegularizationType::kL2DECAY;
        }
        param_outs[idx]->mutable_data<T>(ctx.GetPlace());
        velocity_outs[idx]->mutable_data<MT>(ctx.GetPlace());
        const MT* master_in_data =
            multi_precision ? master_params[idx]->data<MT>() : nullptr;
        MT* master_out_data =
            multi_precision
                ? master_param_outs[idx]->mutable_data<MT>(ctx.GetPlace())
                : nullptr;
        if (platform::is_cpu_place(ctx.GetPlace())) {
          CPUDenseMomentumFunctor<MT> functor;
          functor(params[idx], grads[idx], velocitys[idx], learning_rates[idx],
                  mu, use_nesterov, regularization_flag, regularization_coeff,
                  param_outs[idx], velocity_outs[idx]);
        } else if (platform::is_gpu_place(ctx.GetPlace())) {
          platform::ForRange<DeviceContext> for_range(
              static_cast<const DeviceContext&>(ctx.device_context()),
              params[idx]->numel());
#define PADDLE_LAUNCH_DENSE_MTMOMENTUM_KERNEL(__nesterov, __reg_type)         \
  DenseMomentumFunctor<T, MT, __reg_type, __nesterov> functor(                \
      params[idx]->data<T>(), grads[idx]->data<T>(),                          \
      velocitys[idx]->data<MT>(), learning_rates[idx]->data<MPDType>(),       \
      master_in_data, mu, rescale_grad, params[idx]->numel(),                 \
      regularization_coeff, param_outs[idx]->mutable_data<T>(ctx.GetPlace()), \
      velocity_outs[idx]->mutable_data<MT>(ctx.GetPlace()), master_out_data); \
  for_range(functor);
          if (use_nesterov) {
            if (regularization_flag == RegularizationType::kL2DECAY) {
              PADDLE_LAUNCH_DENSE_MTMOMENTUM_KERNEL(
                  UseNesterov, RegularizationType::kL2DECAY);
            } else {
              PADDLE_LAUNCH_DENSE_MTMOMENTUM_KERNEL(UseNesterov,
                                                    RegularizationType::kNONE);
            }
          } else {
            if (regularization_flag == RegularizationType::kL2DECAY) {
              PADDLE_LAUNCH_DENSE_MTMOMENTUM_KERNEL(
                  NoNesterov, RegularizationType::kL2DECAY);
            } else {
              PADDLE_LAUNCH_DENSE_MTMOMENTUM_KERNEL(NoNesterov,
                                                    RegularizationType::kNONE);
            }
          }
        }
      }
    } else if (grad_vars[0]->IsType<framework::SelectedRows>()) {
      auto grads = ctx.MultiInput<framework::SelectedRows>("Grads");
      // multi_tensor size
      size_t multi_tensor_size = params.size();
      for (size_t idx = 0; idx < multi_tensor_size; idx++) {
        std::string regularization_method = regularization_methods[idx];
        MT regularization_coeff = static_cast<MT>(regularization_coeffs[idx]);
        RegularizationType regularization_flag{
            RegularizationType::kNONE};  // disable regularization
        if (regularization_method == "l2_decay") {
          regularization_flag = RegularizationType::kL2DECAY;
        }
        param_outs[idx]->mutable_data<T>(ctx.GetPlace());
        velocity_outs[idx]->mutable_data<MT>(ctx.GetPlace());
        const MT* master_in_data =
            multi_precision ? master_params[idx]->data<MT>() : nullptr;
        MT* master_out_data =
            multi_precision
                ? master_param_outs[idx]->mutable_data<MT>(ctx.GetPlace())
                : nullptr;
        // sparse update maybe empty.
        if (grads[idx]->rows().size() == 0) {
          VLOG(3) << "Grad SelectedRows contains no data!";
          return;
        }
        framework::SelectedRows tmp_merged_grad;
        framework::SelectedRows* merged_grad = &tmp_merged_grad;
        math::scatter::MergeAdd<DeviceContext, T> merge_func;
        merge_func(ctx.template device_context<DeviceContext>(), *grads[idx],
                   merged_grad);

        const int64_t* rows = merged_grad->rows().Data(ctx.GetPlace());
        int64_t row_numel =
            merged_grad->value().numel() / merged_grad->rows().size();
        platform::ForRange<DeviceContext> for_range(
            static_cast<const DeviceContext&>(ctx.device_context()),
            params[idx]->numel());
        if (use_nesterov) {
          SparseMomentumFunctor<T, MT, UseNesterov> functor(
              params[idx]->data<T>(), merged_grad->value().data<T>(),
              velocitys[idx]->data<MT>(), learning_rates[idx]->data<MPDType>(),
              master_in_data, mu, rescale_grad, rows, row_numel,
              static_cast<int64_t>(merged_grad->rows().size()),
              regularization_flag, regularization_coeff,
              param_outs[idx]->mutable_data<T>(ctx.GetPlace()),
              velocity_outs[idx]->mutable_data<MT>(ctx.GetPlace()),
              master_out_data);
          for_range(functor);
        } else {
          SparseMomentumFunctor<T, MT, NoNesterov> functor(
              params[idx]->data<T>(), merged_grad->value().data<T>(),
              velocitys[idx]->data<MT>(), learning_rates[idx]->data<MPDType>(),
              master_in_data, mu, rescale_grad, rows, row_numel,
              static_cast<int64_t>(merged_grad->rows().size()),
              regularization_flag, regularization_coeff,
              param_outs[idx]->mutable_data<T>(ctx.GetPlace()),
              velocity_outs[idx]->mutable_data<MT>(ctx.GetPlace()),
              master_out_data);
          for_range(functor);
        }
      }
    } else {
      PADDLE_ENFORCE_EQ(
          false, true,
          platform::errors::PermissionDenied(
              "Unsupported Variable Type of Grads "
              "in MTMomentumOp. Excepted LodTensor "
              "or SelectedRows, But received [%s]",
              paddle::framework::ToTypeName(grad_vars[0]->Type())));
    }
  }
};

}  // namespace operators
}  // namespace paddle
