/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/platform/complex.h"
#include "paddle/phi/core/utils/array.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"

namespace paddle {
namespace operators {

// Define the binary functors used in elementwise ops.
// Note: InverseXxxFunctor is needed when calling ElementwiseComputeEx on CPU.

// Add
template <typename T>
using AddFunctor = phi::funcs::AddFunctor<T>;

template <typename T>
using InverseAddFunctor = phi::funcs::InverseAddFunctor<T>;

// Subtract
template <typename T>
using SubFunctor = phi::funcs::SubtractFunctor<T>;

template <typename T>
using InverseSubFunctor = phi::funcs::InverseSubtractFunctor<T>;

// Multiply
template <typename T>
using MulFunctor = phi::funcs::MultiplyFunctor<T>;

template <typename T>
using InverseMulFunctor = phi::funcs::InverseMultiplyFunctor<T>;

// Divide
template <typename T>
using DivFunctor = phi::funcs::DivideFunctor<T>;

template <typename T>
using InverseDivFunctor = phi::funcs::InverseDivideFunctor<T>;

#undef DIV_ERROR_INFO

// Maximum
template <typename T>
using MaxFunctor = phi::funcs::MaximumFunctor<T>;

// Minmum
template <typename T>
using MinFunctor = phi::funcs::MinimumFunctor<T>;

template <typename T>
using Complex = paddle::platform::complex<T>;

// Ternary compare
template <typename T>
using MaxGradXFunctor = phi::funcs::MaxGradXFunctor<T>;
template <typename T>
using MaxGradYFunctor = phi::funcs::MaxGradYFunctor<T>;
template <typename InT, typename OutT>
using MaxGradXYFunctor = phi::funcs::MaxGradXYFunctor<InT, OutT>;

template <typename T>
using MinGradXFunctor = phi::funcs::MinGradXFunctor<T>;
template <typename T>
using MinGradYFunctor = phi::funcs::MinGradYFunctor<T>;
template <typename InT, typename OutT>
using MinGradXYFunctor = phi::funcs::MinGradXYFunctor<InT, OutT>;

}  // namespace operators
}  // namespace paddle
