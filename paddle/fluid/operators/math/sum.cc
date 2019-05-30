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
                  const std::vector<framework::Tensor*>& inputs,
                  framework::Tensor* output) {
    size_t in_num = inputs.size();
    PADDLE_ENFORCE_LE(in_num, 2UL,
                      "The number of inputs should be not less than 2.");

    bool in_place = (output == inputs[0]) ? true : false;
    int start = in_place ? 1 : 0;

    auto result = framework::EigenVector<T>::Flatten(*output);
    auto& place = *context.eigen_device();

    if (!in_place) {
      if (inputs[0]->numel() && inputs[1]->numel()) {
        auto in_0_e = framework::EigenVector<T>::Flatten(*inputs[0]);
        auto in_1_e = framework::EigenVector<T>::Flatten(*inputs[1]);
        result.device(place) = in_0_e + in_1_e;
        start = 2;
      }
      if (start != 2) {
        math::SetConstant<platform::CPUDeviceContext, T> constant_functor;
        constant_functor(context, output, static_cast<T>(0));
      }
    }

    // If in_place, just skip the first tensor
    for (size_t i = start; i < in_num; i++) {
      if (inputs[i]->numel() == 0) {
        continue;
      }
      auto in = framework::EigenVector<T>::Flatten(*inputs[i]);
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
