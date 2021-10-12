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

namespace paddle {
namespace operators {

template <typename T, typename MT, uint32_t kParamNum = 60>
struct MergedMomentumKernelParam {
  static constexpr auto N = kParamNum;
  uint32_t param_num;
  size_t sizes[N];
  T *params[N];
  const T *grads[N];
  MT *velocitys[N];
  MT *master_params[N];
  const MT *lrs[N];
  MT mu;
  MT rescale_grad;

  HOSTDEVICE void operator()(size_t i) const {
    for (uint32_t idx = 0; idx < param_num; ++idx) {
      auto size = sizes[idx];
      if (i > size) continue;

      auto param_p = params[idx];
      auto grad_p = grads[idx];
      auto velocity_p = velocitys[idx];
      auto master_param_p = master_params[idx];
      auto lr_p = lrs[idx];

      const MT param =
          master_param_p ? master_param_p[i] : static_cast<MT>(param_p[i]);
      MT grad = static_cast<MT>(grad_p[i]) * rescale_grad;
      const MT lr = static_cast<MT>(lr_p[0]);
      const MT velocity = velocity_p[i];
      MT velocity_out = velocity * mu + grad;
      MT param_out = param - (grad + velocity_out * mu) * lr;
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
    PADDLE_ENFORCE_EQ(n, params_out.size());
    for (size_t i = 0; i < n; ++i) {
      PADDLE_ENFORCE_EQ(params[i], params_out[i]);
    }

    auto grads = ctx.MultiInput<framework::Tensor>("Grad");
    PADDLE_ENFORCE_EQ(n, grads.size());

    auto velocitys = ctx.MultiInput<framework::Tensor>("Velocity");
    PADDLE_ENFORCE_EQ(n, velocitys.size());
    auto velocitys_out = ctx.MultiOutput<framework::Tensor>("VelocityOut");
    PADDLE_ENFORCE_EQ(n, velocitys_out.size());
    for (size_t i = 0; i < n; ++i) {
      PADDLE_ENFORCE_EQ(velocitys[i], velocitys_out[i]);
    }

    auto master_params = ctx.MultiInput<framework::Tensor>("MasterParam");
    auto master_params_out =
        ctx.MultiOutput<framework::Tensor>("MasterParamOut");
    auto multi_precision = ctx.Attr<bool>("multi_precision");
    if (multi_precision) {
      PADDLE_ENFORCE_EQ(n, master_params.size());
      PADDLE_ENFORCE_EQ(n, master_params_out.size());
      for (size_t i = 0; i < n; ++i) {
        PADDLE_ENFORCE_EQ(master_params[i], master_params_out[i]);
        PADDLE_ENFORCE_NOT_NULL(master_params[i]);
      }
    } else {
      master_params.clear();
      master_params_out.clear();
    }

    auto lrs = ctx.MultiInput<framework::Tensor>("LearningRate");
    PADDLE_ENFORCE_EQ(n, lrs.size());

    auto mu = ctx.Attr<float>("mu");
    auto rescale_grad = ctx.Attr<float>("rescale_grad");
    using MPType = typename operators::details::MPTypeTrait<T>::Type;

    auto &dev_ctx = ctx.template device_context<DeviceContext>();

    MergedMomentumKernelParam<T, MPType> kernel_params;
    constexpr auto kMaxMergedNum = decltype(kernel_params)::N;
    size_t kernel_num = (n + kMaxMergedNum - 1) / kMaxMergedNum;
    kernel_params.mu = static_cast<MPType>(mu);
    kernel_params.rescale_grad = static_cast<MPType>(rescale_grad);

    for (size_t i = 0; i < kernel_num; ++i) {
      size_t start = i * kMaxMergedNum;
      size_t end = std::min((i + 1) * kMaxMergedNum, n);
      kernel_params.param_num = static_cast<uint32_t>(end - start);

      size_t max_size = 0;
      for (size_t j = 0; j < kernel_params.param_num; ++j) {
        auto size = static_cast<size_t>(params_out[j + start]->numel());
        max_size = std::max(max_size, size);
        kernel_params.sizes[j] = size;
        kernel_params.params[j] = params_out[j + start]->data<T>();
        kernel_params.grads[j] = grads[j + start]->data<T>();
        kernel_params.velocitys[j] = velocitys_out[j + start]->data<MPType>();
        kernel_params.master_params[j] =
            multi_precision ? master_params_out[j + start]->data<MPType>()
                            : nullptr;
        kernel_params.lrs[j] = lrs[j + start]->data<MPType>();
      }
      platform::ForRange<DeviceContext> for_range(dev_ctx, max_size);
      for_range(kernel_params);
      LOG(INFO) << "Launch MergedMomentum kernel " << i;
    }
  }
};

}  // namespace operators
}  // namespace paddle
