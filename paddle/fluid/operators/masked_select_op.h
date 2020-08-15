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
    *auto* input = context.InputVar("X");
    auto* mask = context.InputVar("Mask");
    auto* out = context.OutputVar("Y");

    auto input_size = inputs->numel();
    auto mask_size = mask->numel();
    int out_size = 0;

    for (int i = 0; i < mask_size; i++) {
      if (mask[i]) out_size++;
    }

    framework::DDim out_dim{out_size};
    out->Resize(out_dim);

    int index = 0;
    for (int i = 0; i < mask_size; i++) {
      if (mask[i]) {
        out[index] = input[i] index++;
      }
    }
  }
};

template <typename DeviceContext, typename T>
class MaskedSelectGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* mask = context.InputVar("Mask");
    auto* out = context.OutputVar(framework::GradVarName("X"));
    auto* input = context.InputVar(framework::GradVarName("Out"));
  }
};

}  // namespace operators
}  // namespace paddle
