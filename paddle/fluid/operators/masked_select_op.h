// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include <vector>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using DDim = framework::DDim;

template <typename DeviceContext, typename T>
class MaskedSelectKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto input = context.Input<framework::Tensor>("X");
    auto mask = context.Input<framework::Tensor>("Mask");
    auto out = context.Output<framework::Tensor>("Y");
    auto* mask_data = mask->data<bool>();
    auto input_data = input->data<T>();

    auto mask_size = mask->numel();

    auto input_dim = input->dims();
    auto mask_dim = mask->dims();
    PADDLE_ENFORCE_EQ(
        input_dim, mask_dim,
        platform::errors::InvalidArgument(
            "The dim size of input and mask in OP(masked_selected) "
            "must be equal, but got input dim:(%ld), mask dim: "
            "(%ld). Please check input "
            "value.",
            input_dim, mask_dim));

    int out_size = 0;
    for (int i = 0; i < mask_size; i++) {
      if (mask_data[i]) out_size++;
    }

    framework::DDim out_dim{out_size};
    out->Resize(out_dim);
    auto out_data = out->mutable_data<T>(context.GetPlace());

    int index = 0;
    for (int i = 0; i < mask_size; i++) {
      if (mask_data[i]) {
        out_data[index] = input_data[i];
        index++;
      }
    }
  }
};

template <typename DeviceContext, typename T>
class MaskedSelectGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto out = context.Output<framework::Tensor>(framework::GradVarName("X"));
    auto mask = context.Input<framework::Tensor>("Mask");
    auto input = context.Input<framework::Tensor>(framework::GradVarName("Y"));

    auto* mask_data = mask->data<bool>();
    auto* input_data = input->data<T>();
    auto* out_data = out->mutable_data<T>(context.GetPlace());
    int mask_size = mask->numel();

    int index = 0;
    for (int i = 0; i < mask_size; i++) {
      if (mask_data[i]) {
        out_data[i] = input_data[index];
        index++;
      } else {
        out_data[i] = 0;
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
