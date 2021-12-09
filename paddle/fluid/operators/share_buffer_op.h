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

namespace paddle {
namespace operators {

template <typename T>
class ShareBufferOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto inputs = ctx.MultiInput<framework::Tensor>("X");
    auto outputs = ctx.MultiOutput<framework::Tensor>("Out");
    size_t n = inputs.size();
    PADDLE_ENFORCE_EQ(n, outputs.size(), platform::errors::PermissionDenied(
                                             "Variable number not match."));
    const auto &share_dims_and_dtype =
        ctx.Attr<std::vector<bool>>("share_dims_and_dtype");
    if (!share_dims_and_dtype.empty()) {
      PADDLE_ENFORCE_EQ(n, share_dims_and_dtype.size(),
                        platform::errors::PermissionDenied(
                            "Attribute share_dims_and_dtype number not match "
                            "input variable number."));
    }

    const std::vector<std::string> *input_args = nullptr,
                                   *output_args = nullptr;
    if (VLOG_IS_ON(10)) {
      input_args = &ctx.GetOp().Inputs("X");
      output_args = &ctx.GetOp().Outputs("Out");
    }
    for (size_t i = 0; i < n; ++i) {
      if (inputs[i] == nullptr || outputs[i] == nullptr) {
        continue;
      }
      outputs[i]->ShareBufferWith(*inputs[i]);
      VLOG(10) << "Share tensor buffer " << (*input_args)[i] << " -> "
               << (*output_args)[i];
      if (!share_dims_and_dtype.empty() && share_dims_and_dtype[i]) {
        outputs[i]->Resize(inputs[i]->dims());
        outputs[i]->ShareDataTypeWith(*inputs[i]);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
