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
#include <vector>
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/cudnn_helper.h"

namespace paddle {
namespace operators {
namespace math {

#ifdef PADDLE_WITH_CUDA
template <typename T>
class PreluChannelWiseDirectCUDAFunctor {
 public:
  void operator()(cudaStream_t stream, const T *input, const T *alpha,
                  T *output, std::vector<int> input_shape);
};

template <typename T>
class PreluElementWiseDirectCUDAFunctor {
 public:
  void operator()(cudaStream_t stream, const T *input, const T *alpha,
                  T *output, std::vector<int> input_shape);
};

template <typename T>
class PreluScalarDirectCUDAFunctor {
 public:
  void operator()(cudaStream_t stream, const T *input, const T *alpha,
                  T *output, std::vector<int> input_shape);
};
#endif

}  // namespace math
}  // namespace operators
}  // namespace paddle
