/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/math/sum.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {
namespace math {

/*
 * All inputs' dimension should be the same and the values of
 * each dimension must be the same.
 */
template <typename T>
class SumLoDTensorFunctor<platform::CPUDeviceContext, T> {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const std::vector<const framework::Tensor*>& inputs,
                  framework::Tensor* output) {
    bool in_place = (output == inputs[0]) ? true : false;

    std::vector<const framework::Tensor*> actual_inputs;
    int64_t length = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (inputs[i]->numel() > 0) {
        actual_inputs.push_back(inputs[i]);
        if (length == 0) {
          length = inputs[i]->numel();
        } else {
          PADDLE_ENFORCE_EQ(length, inputs[i]->numel());
        }
      }
    }

    size_t in_num = actual_inputs.size();
    if (in_num == 1) {
      // Copy actual_inputs[0] -> output
      framework::TensorCopy(*actual_inputs[0], context.GetPlace(), context,
                            output);
      return;
    }

    auto result = framework::EigenVector<T>::Flatten(*output);
    auto& place = *context.eigen_device();

    int start = in_place ? 1 : 0;
    if (!in_place) {
      auto in_0_e = framework::EigenVector<T>::Flatten(*actual_inputs[0]);
      auto in_1_e = framework::EigenVector<T>::Flatten(*actual_inputs[1]);
      result.device(place) = in_0_e + in_1_e;
      start = 2;
    }

    // If in_place, just skip the first tensor
    for (size_t i = start; i < in_num; i++) {
      auto in = framework::EigenVector<T>::Flatten(*actual_inputs[i]);
      result.device(place) = result + in;
    }
  }
};

template class SumLoDTensorFunctor<platform::CPUDeviceContext, float>;
template class SumLoDTensorFunctor<platform::CPUDeviceContext, double>;
template class SumLoDTensorFunctor<platform::CPUDeviceContext, int>;
template class SumLoDTensorFunctor<platform::CPUDeviceContext, int64_t>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
