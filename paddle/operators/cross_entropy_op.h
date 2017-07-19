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
#include "glog/logging.h"
#include "paddle/framework/operator.h"

namespace paddle {
namespace operators {

template <typename Place, typename T>
class CrossEntropyOpKernel : public framework::OpKernel {
public:
  void Compute(const framework::KernelContext& context) const override {
    auto X = context.Input(0)->Get<framework::Tensor>();
    const float* X_data = X.data<float>();
    const float* label_data =
        context.Input(1)->Get<framework::Tensor>().data<float>();
    float* Y_data =
        context.Output(0)->GetMutable<framework::Tensor>()->raw_data<float>();

    int input_rank = (int)X.dims().size();
    int batch_size, class_num;
    if (input_rank == 1) {
      batch_size = 1;
      class_num = X.dims()[0];
    } else {
      batch_size = X.dims()[0];
      class_num = X.dims()[1];
    }

    // Y[i] = sum_j (label[i][j] * log(X[i][j]))
    for (int i = 0; i < batch_size; ++i) {
      Y_data[i] = 0;
      for (int j = 0; j < class_num; ++j) {
        Y_data[i] +=
            label_data[i * class_num + j] * std::log(X_data[i * class_num + j]);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
