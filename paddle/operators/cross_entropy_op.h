/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include "paddle/operators/type_alias.h"

namespace paddle {
namespace operators {

template <typename Place, typename T>
class OnehotCrossEntropyOpKernel : public OpKernel {
public:
  constexpr T LOG_THRESHOLD() const { return static_cast<T>(1e-20); }

  void Compute(const KernelContext& context) const override {
    auto X = context.Input(0)->Get<Tensor>();
    const T* X_data = X.data<T>();
    const int* label_data = context.Input(1)->Get<Tensor>().data<int>();
    auto* Y = context.Output(0)->GetMutable<Tensor>();

    Y->mutable_data<T>(context.GetPlace());

    T* Y_data = Y->data<T>();

    int batch_size = X.dims()[0];
    int class_num = X.dims()[1];

    // Y[i] = -log(X[i][j])
    for (int i = 0; i < batch_size; ++i) {
      Y_data[i] = -std::log(
          std::max(X_data[i * class_num + label_data[i]], LOG_THRESHOLD()));
    }
  }
};

}  // namespace operators
}  // namespace paddle
