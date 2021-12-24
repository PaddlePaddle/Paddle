//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#ifdef PADDLE_WITH_MKLML
#include <mkl.h>
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/complex.h"

namespace paddle {
namespace operators {
namespace math {

template <>
template <typename T>
void Sparse<platform::CPUDeviceContext>::DenseToSparseCsr(
    const int M, const int N, const T* dense, int64_t* crows, int64_t* cols,
    T* values) const {}

template <>
template <>
void Sparse<platform::CPUDeviceContext>::DenseToSparseCsr(
    const int M, const int N, const float* dense, int64_t* crows, int64_t* cols,
    float* values) const {}

template <>
template <>
void Sparse<platform::CPUDeviceContext>::DenseToSparseCsr(
    const int M, const int N, const double* dense, int64_t* crows,
    int64_t* cols, double* values) const {}

}  // namespace math
}  // namespace operators
}  // namespace paddle
