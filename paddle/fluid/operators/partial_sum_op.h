/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;

template <typename DeviceContext, typename T>
class PartialSumKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto ins = ctx.MultiInput<phi::DenseTensor>("X");
    phi::DenseTensor* out = ctx.Output<phi::DenseTensor>("Out");
    PADDLE_ENFORCE_EQ(
        ins[0] != nullptr,
        true,
        platform::errors::InvalidArgument("The input should not be null."));

    auto place = ctx.GetPlace();  // CPUPlace only now

    auto* out_t = out->mutable_data<T>(place);
    auto start_index = ctx.Attr<int>("start_index");
    auto length = ctx.Attr<int>("length");
    auto batch_size = ins[0]->dims()[0];
    if (length == -1) {
      length = ins[0]->dims()[1] - start_index;
    }

    memset(out_t, 0, sizeof(T) * batch_size * length);

    for (size_t i = 0; i < ins.size(); ++i) {
      auto* in_t = ins[i]->data<T>();
      auto total_len = ins[i]->dims()[1];
      for (auto bs_id = 0; bs_id < batch_size; ++bs_id) {
        for (auto k = 0; k < length; ++k) {
          out_t[bs_id * length + k] +=
              in_t[bs_id * total_len + start_index + k];
        }
      }
    }
  }
};

template <typename T>
class PartialSumGradientOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* out_grad = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto ins = ctx.MultiInput<framework::LoDTensor>("X");
    auto outs =
        ctx.MultiOutput<framework::LoDTensor>(framework::GradVarName("X"));

    PADDLE_ENFORCE_EQ(
        ins[0] != nullptr,
        true,
        platform::errors::InvalidArgument("The input should not be null."));
    auto start_index = ctx.Attr<int>("start_index");
    auto length = ctx.Attr<int>("length");
    auto batch_size = ins[0]->dims()[0];
    if (length == -1) {
      length = ins[0]->dims()[1] - start_index;
    }

    // initialize
    auto& place =
        *ctx.template device_context<phi::CPUContext>().eigen_device();
    for (size_t i = 0; i < outs.size(); ++i) {
      outs[i]->mutable_data<T>(ctx.GetPlace());
      auto dxt = framework::EigenVector<T>::Flatten(*outs[i]);
      dxt.device(place) = dxt.constant(static_cast<T>(0));
    }

    auto* out_grad_t = out_grad->data<T>();
    for (size_t i = 0; i < outs.size(); ++i) {
      auto* out_t = outs[i]->data<T>();
      auto total_len = ins[i]->dims()[1];
      for (auto bs_id = 0; bs_id < batch_size; ++bs_id) {
        for (int len = 0; len < length; ++len) {
          out_t[start_index + bs_id * total_len + len] =
              out_grad_t[bs_id * length + len] * static_cast<T>(1);
        }
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
