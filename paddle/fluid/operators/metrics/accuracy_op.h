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
#include <algorithm>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class AccuracyKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* inference = ctx.Input<Tensor>("Out");
    auto* indices = ctx.Input<Tensor>("Indices");
    auto* label = ctx.Input<Tensor>("Label");
    auto* accuracy = ctx.Output<Tensor>("Accuracy");
    auto* correct = ctx.Output<Tensor>("Correct");
    auto* total = ctx.Output<Tensor>("Total");

    int* correct_data = correct->mutable_data<int>(ctx.GetPlace());
    int* total_data = total->mutable_data<int>(ctx.GetPlace());
    float* accuracy_data = accuracy->mutable_data<float>(ctx.GetPlace());

    const int64_t* indices_data = indices->data<int64_t>();
    const int64_t* label_data = label->data<int64_t>();

    size_t num_samples = inference->dims()[0];
    size_t class_dim = inference->dims()[1];
    *accuracy_data = 0.0f;

    if (num_samples == 0) {
      return;
    }

    int num_correct = 0;
    // assume inference is already the topk of the output
    for (size_t i = 0; i < num_samples; ++i) {
      PADDLE_ENFORCE_GE(label_data[i], 0, "label must >= 0");
      for (size_t j = 0; j < class_dim; ++j) {
        if (indices_data[i * class_dim + j] == label_data[i]) {
          ++num_correct;
          break;
        }
      }
    }

    *correct_data = num_correct;
    *total_data = num_samples;
    *accuracy_data =
        static_cast<float>(num_correct) / static_cast<float>(num_samples);
  }
};

}  // namespace operators
}  // namespace paddle
