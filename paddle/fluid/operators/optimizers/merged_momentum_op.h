// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/optimizers/momentum_op.h"
#include "paddle/fluid/platform/for_range.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace operators {

template <typename MT, uint32_t kParamNum, bool kHasMasterParams>
struct MergedMomentumMasterParams {
  MT *PADDLE_RESTRICT master_params[kParamNum];

  HOSTDEVICE MT *MasterParam(size_t idx) const { return master_params[idx]; }
  HOSTDEVICE void SetMasterParam(size_t idx, MT *p) { master_params[idx] = p; }
};

template <typename MT, uint32_t kParamNum>
struct MergedMomentumMasterParams<MT, kParamNum, false> {
  HOSTDEVICE constexpr MT *MasterParam(size_t) const { return nullptr; }
  HOSTDEVICE constexpr void SetMasterParam(size_t, MT *) {}
};

template <typename T, typename MT, bool kHasMasterParams,
          uint32_t kParamNum = kHasMasterParams ? 55 : 110>
struct MergedMomentumKernelParam
    : public MergedMomentumMasterParams<MT, kParamNum, kHasMasterParams> {
  static constexpr auto N = kParamNum;
  size_t sizes[N];
  T *PADDLE_RESTRICT params[N];
  const T *PADDLE_RESTRICT grads[N];
  MT *PADDLE_RESTRICT velocitys[N];
  const MT *PADDLE_RESTRICT lr;
  MT mu;
  MT rescale_grad;
  uint32_t param_num;

  HOSTDEVICE void operator()(size_t i) const {
    const auto lr_val = *lr;
    for (uint32_t idx = 0; idx < param_num; ++idx) {
      auto size = sizes[idx];
      if (i >= size) continue;

      auto param_p = params[idx];
      auto grad_p = grads[idx];
      auto velocity_p = velocitys[idx];
      auto master_param_p = this->MasterParam(idx);

      const MT param =
          master_param_p ? master_param_p[i] : static_cast<MT>(param_p[i]);
      const MT grad = static_cast<MT>(grad_p[i]) * rescale_grad;
      const MT velocity = velocity_p[i];
      const MT velocity_out = velocity * mu + grad;
      const MT param_out = param - lr_val * velocity_out;
      velocity_p[i] = velocity_out;
      param_p[i] = static_cast<T>(param_out);
      if (master_param_p) {
        master_param_p[i] = param_out;
      }
    }
  }
};

template <typename DeviceContext, typename T>
class MergedMomentumOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
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

    auto master_params = ctx.MultiInput<framework::Tensor>("MasterParam");
    auto master_params_out =
        ctx.MultiOutput<framework::Tensor>("MasterParamOut");
    auto multi_precision = ctx.Attr<bool>("multi_precision");
    if (multi_precision) {
      PADDLE_ENFORCE_EQ(
          n, master_params.size(),
          platform::errors::InvalidArgument(
              "The size of Input(MasterParam) must be "
              "equal to Input(Param), but got the size of Input(MasterParam) "
              "is %d, the size of Input(Param) is %d.",
              master_params.size(), n));
      PADDLE_ENFORCE_EQ(
          n, master_params_out.size(),
          platform::errors::InvalidArgument(
              "The size of Output(MasterParamOut) must be equal to "
              "Input(MasterParam), but got the size of Output(MasterParamOut) "
              "is %d, the size of Input(Param) is %d.",
              master_params_out.size(), n));
      for (size_t i = 0; i < n; ++i) {
        PADDLE_ENFORCE_EQ(master_params[i], master_params_out[i],
                          platform::errors::InvalidArgument(
                              "Input(MasterParam) and Output(MasterParamOut) "
                              "must be the same Tensors."));
        PADDLE_ENFORCE_NOT_NULL(master_params[i],
                                platform::errors::InvalidArgument(
                                    "Input(MasterParam) must be provided when "
                                    "multi_precision=True."));
      }
    } else {
      master_params.clear();
      master_params_out.clear();
    }

    auto mu = ctx.Attr<float>("mu");
    auto rescale_grad = ctx.Attr<float>("rescale_grad");
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

    using MPType = typename operators::details::MPTypeTrait<T>::Type;

    auto &dev_ctx = ctx.template device_context<DeviceContext>();

    if (lrs.size() == 1 && use_nesterov == false &&
        regularization_methods.size() == 0) {
#define PADDLE_LAUNCH_MERGED_MOMENTUM_KERNEL(kMultiPrecision)                \
  MergedMomentumKernelParam<T, MPType, kMultiPrecision> kernel_params;       \
  constexpr auto kMaxMergedNum = decltype(kernel_params)::N;                 \
  size_t kernel_num = (n + kMaxMergedNum - 1) / kMaxMergedNum;               \
  kernel_params.mu = static_cast<MPType>(mu);                                \
  kernel_params.rescale_grad = static_cast<MPType>(rescale_grad);            \
  kernel_params.lr = lrs[0]->data<MPType>();                                 \
  for (size_t i = 0; i < kernel_num; ++i) {                                  \
    size_t start = i * kMaxMergedNum;                                        \
    size_t end = std::min((i + 1) * kMaxMergedNum, n);                       \
    kernel_params.param_num = static_cast<uint32_t>(end - start);            \
    size_t max_size = 0;                                                     \
    for (size_t j = 0; j < kernel_params.param_num; ++j) {                   \
      auto size = static_cast<size_t>(params_out[j + start]->numel());       \
      max_size = std::max(max_size, size);                                   \
      kernel_params.sizes[j] = size;                                         \
      kernel_params.params[j] = params_out[j + start]->data<T>();            \
      kernel_params.grads[j] = grads[j + start]->data<T>();                  \
      kernel_params.velocitys[j] = velocitys_out[j + start]->data<MPType>(); \
      kernel_params.SetMasterParam(                                          \
          j, kMultiPrecision ? master_params_out[j + start]->data<MPType>()  \
                             : nullptr);                                     \
    }                                                                        \
    platform::ForRange<DeviceContext> for_range(dev_ctx, max_size);          \
    for_range(kernel_params);                                                \
    VLOG(10) << "Launch MergedMomentum kernel " << i << " "                  \
             << kernel_params.param_num;                                     \
  }
      if (multi_precision) {
        PADDLE_LAUNCH_MERGED_MOMENTUM_KERNEL(true);
      } else {
        PADDLE_LAUNCH_MERGED_MOMENTUM_KERNEL(false);
      }
#undef PADDLE_LAUNCH_MERGED_MOMENTUM_KERNEL
    } else {
      for (size_t idx = 0; idx < n; idx++) {
        RegularizationType regularization_flag =
            regularization_methods.size() > 0 &&
                    regularization_methods[idx] == "l2_decay"
                ? RegularizationType::kL2DECAY
                : RegularizationType::kNONE;

        MPType regularization_coeff = static_cast<MPType>(0.0);
        if (regularization_coeffs.size() != 0) {
          regularization_coeff =
              static_cast<MPType>(regularization_coeffs[idx]);
        }
        auto lr_temp = lrs.size() > 1 ? lrs[idx] : lrs[0];

        const MPType *master_in_data =
            multi_precision ? master_params[idx]->data<MPType>() : nullptr;
        MPType *master_out_data =
            multi_precision ? master_params_out[idx]->data<MPType>() : nullptr;
        if (platform::is_cpu_place(ctx.GetPlace())) {
          CPUDenseMomentumFunctor<MPType> functor;
          functor(params[idx], grads[idx], velocitys[idx], lr_temp, mu,
                  use_nesterov, regularization_flag, regularization_coeff,
                  params_out[idx], velocitys_out[idx]);
          VLOG(10) << "Launch MergedMomentum cpu kernel.";
        } else if (platform::is_gpu_place(ctx.GetPlace())) {
          platform::ForRange<DeviceContext> for_range(
              static_cast<const DeviceContext &>(ctx.device_context()),
              params[idx]->numel());
#define PADDLE_LAUNCH_DENSE_MTMOMENTUM_KERNEL(__nesterov, __reg_type)          \
  DenseMomentumFunctor<T, MPType, __reg_type, __nesterov> functor(             \
      params[idx]->data<T>(), grads[idx]->data<T>(),                           \
      velocitys[idx]->data<MPType>(), lr_temp->data<MPType>(), master_in_data, \
      mu, rescale_grad, params[idx]->numel(), regularization_coeff,            \
      params_out[idx]->data<T>(), velocitys_out[idx]->data<MPType>(),          \
      master_out_data);                                                        \
  for_range(functor);
          if (use_nesterov) {
            if (regularization_flag == RegularizationType::kL2DECAY) {
              PADDLE_LAUNCH_DENSE_MTMOMENTUM_KERNEL(
                  UseNesterov, RegularizationType::kL2DECAY);
              VLOG(10)
                  << "Launch MergedMomentum gpu kernel use_nesterov kL2DECAY.";
            } else {
              PADDLE_LAUNCH_DENSE_MTMOMENTUM_KERNEL(UseNesterov,
                                                    RegularizationType::kNONE);
              VLOG(10)
                  << "Launch MergedMomentum gpu kernel use_nesterov kNONE.";
            }
          } else {
            if (regularization_flag == RegularizationType::kL2DECAY) {
              PADDLE_LAUNCH_DENSE_MTMOMENTUM_KERNEL(
                  NoNesterov, RegularizationType::kL2DECAY);
              VLOG(10)
                  << "Launch MergedMomentum gpu kernel no_nesterov kL2DECAY.";
            } else {
              PADDLE_LAUNCH_DENSE_MTMOMENTUM_KERNEL(NoNesterov,
                                                    RegularizationType::kNONE);
              VLOG(10) << "Launch MergedMomentum gpu kernel no_nesterov kNONE.";
            }
          }
        }
      }
      VLOG(10)
          << "Launch MergedMomentum kernel with multi_lr and regularization.";
    }
  }
};

}  // namespace operators
}  // namespace paddle
