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
#include <functional>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

template <typename T>
class CPULinspaceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    T start = context.Input<framework::Tensor>("Start")->data<T>()[0];
    T stop = context.Input<framework::Tensor>("Stop")->data<T>()[0];
    int32_t num = context.Input<framework::Tensor>("Num")->data<int32_t>()[0];
    auto* out = context.Output<framework::Tensor>("Out");
    PADDLE_ENFORCE(num > 0, "The num of linspace op should be larger than 0.");

    out->Resize(framework::make_ddim({num}));

    T* out_data = out->mutable_data<T>(context.GetPlace());

    if (num > 1) {
      T step = (stop - start) / (num - 1);
      T value = start;
      for (int i = 0; i < num; ++i) {
        out_data[i] = value;
        value += step;
      }
    } else {
      out_data[0] = start;
    }
  }
};

}  // namespace operators
}  // namespace paddle
