// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/primitive/backend/backend.h"

namespace paddle {
namespace primitive {

using Tensor = paddle::Tensor;
using IntArray = paddle::experimental::IntArray;

template <typename T>
Tensor full(const IntArray& shape,
            const Scalar& value,
            DataType dtype = DataType::FLOAT32,
            Place place = Place()) {
  return backend::full<T>(shape, value, dtype, place);
}

template <typename T>
Tensor arange(const Scalar& start,
              const Scalar& end,
              const Scalar& step,
              DataType dtype = DataType::FLOAT64,
              Place place = CPUPlace()) {
  return backend::arange<T>(start, end, step, dtype, place);
}

}  // namespace primitive
}  // namespace paddle
