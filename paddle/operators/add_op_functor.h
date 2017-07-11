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

#include "paddle/framework/tensor_types.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace paddle {
namespace operators {
namespace functor {

template <typename Device, typename T>
struct Add {
  void Operator()(const Device& d,
                  typename TTypes<T>::ConstTensor input1,
                  typename TTypes<T>::ConstTensor input2,
                  typename TTypes<T>::Tensor output) {
    output.device(d) = input1 + input2;
  }
};
}  // namespace functor
}  // namespace operators
}  // namespace paddle
