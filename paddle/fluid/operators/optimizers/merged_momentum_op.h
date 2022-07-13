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
#include "paddle/phi/kernels/impl/momentum_kernel_impl.h"

namespace paddle {
namespace operators {

template <typename T>
using MultiPrecisionType = typename details::MPTypeTrait<T>::Type;

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

template <typename T,
          typename MT,
          bool kHasMasterParams,
          uint32_t kParamNum = kHasMasterParams ? 55 : 110>
struct MergedMomentumKernelParam
    : public MergedMomentumMasterParams<MT, kParamNum, kHasMasterParams> {
  static constexpr auto N = kParamNum;
  size_t sizes[N];
  T *PADDLE_RESTRICT params[N];
  const T *PADDLE_RESTRICT grads[N];
  MT *PADDLE_RESTRICT velocitys[N];
  const MultiPrecisionType<MT> *PADDLE_RESTRICT lr;
  MT mu;
  MT rescale_grad;
  uint32_t param_num;

  HOSTDEVICE void operator()(size_t i) const {
    const MT lr_val = static_cast<MT>(*lr);
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
  using MPType = typename operators::details::MPTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const bool multi_precision = ctx.Attr<bool>("multi_precision");
    if (multi_precision) {
      InnerCompute<MPType>(ctx, multi_precision);
    } else {
      InnerCompute<T>(ctx, multi_precision);
    }
  }

 private:
  template <typename MT>
  void InnerCompute(const framework::ExecutionContext &ctx,
                    const bool multi_precision) const {}
};

}  // namespace operators
}  // namespace paddle
