// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <vector>
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/utils/optional.h"

namespace paddle {
namespace prim {

using Tensor = paddle::Tensor;
using Scalar = paddle::experimental::Scalar;
using IntArray = paddle::experimental::IntArray;
using DataType = phi::DataType;

template <typename T>
Tensor full(const IntArray& shape,
            const Scalar& value,
            DataType dtype = DataType::FLOAT32,
            const Place& place = CPUPlace());

template <typename T>
Tensor cast(const Tensor& x, DataType dtype);

template <typename T>
Tensor slice(const Tensor& input,
             const std::vector<int64_t>& axes,
             const IntArray& starts,
             const IntArray& ends,
             const std::vector<int64_t>& infer_flags,
             const std::vector<int64_t>& decrease_axis);

}  // namespace prim
}  // namespace paddle
