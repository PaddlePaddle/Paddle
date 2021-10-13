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
    PADDLE_ENFORCE_EQ(
        n, params_out.size(),
        platform::errors::InvalidArgument(
            "Output(ParamOut) number must be equal to Input(Param) number."));
    for (size_t i = 0; i < n; ++i) {
      PADDLE_ENFORCE_EQ(
          params[i], params_out[i],
          platform::errors::InvalidArgument(
              "Input(Param) and Output(ParamOut) must be the same Tensors."));
    }

    auto grads = ctx.MultiInput<framework::Tensor>("Grad");
    PADDLE_ENFORCE_EQ(
        n, grads.size(),
        platform::errors::InvalidArgument(
            "Input(Grad) number must be equal to Input(Param) number."));

    auto velocitys = ctx.MultiInput<framework::Tensor>("Velocity");
    PADDLE_ENFORCE_EQ(n, velocitys.size(),
                      platform::errors::InvalidArgument(
                          "Input(Velocity) number and Input(Param) number."));

    auto velocitys_out = ctx.MultiOutput<framework::Tensor>("VelocityOut");
    PADDLE_ENFORCE_EQ(
        n, velocitys_out.size(),
        platform::errors::InvalidArgument("Output(VelocityOut) number must be "
                                          "equal to Input(Param) number."));
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
          platform::errors::InvalidArgument("Input(MasterParam) number must be "
                                            "equal to Input(Param) number."));
      PADDLE_ENFORCE_EQ(n, master_params_out.size(),
                        platform::errors::InvalidArgument(
                            "Output(MasterParamOut) number must be equal to "
                            "Input(MasterParam) number."));
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

    auto lr = ctx.Input<framework::Tensor>("LearningRate");
    auto mu = ctx.Attr<float>("mu");
    auto rescale_grad = ctx.Attr<float>("rescale_grad");
    using MPType = typename operators::details::MPTypeTrait<T>::Type;

    auto &dev_ctx = ctx.template device_context<DeviceContext>();

#define PADDLE_LAUNCH_MERGED_MOMENTUM_KERNEL(kMultiPrecision)                \
  MergedMomentumKernelParam<T, MPType, kMultiPrecision> kernel_params;       \
  constexpr auto kMaxMergedNum = decltype(kernel_params)::N;                 \
  size_t kernel_num = (n + kMaxMergedNum - 1) / kMaxMergedNum;               \
  kernel_params.mu = static_cast<MPType>(mu);                                \
  kernel_params.rescale_grad = static_cast<MPType>(rescale_grad);            \
  kernel_params.lr = lr->data<MPType>();                                     \
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
  }
};

}  // namespace operators
}  // namespace paddle
