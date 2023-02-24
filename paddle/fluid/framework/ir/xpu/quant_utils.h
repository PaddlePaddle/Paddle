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
#include "paddle/phi/core/dense_tensor.h"

namespace paddle {
namespace framework {
namespace ir {

template <typename T>
void Transpose2D(const phi::DenseTensor& in, phi::DenseTensor* out);

// 1. Quant weight from fp32 to int16/int31
// 2. Weight data is in-place update.
// 3. Generate weight max tensor
template <typename T>
void QuantWeight(phi::DenseTensor* weight,
                 phi::DenseTensor* weight_max,
                 bool transpose);

}  // namespace ir
}  // namespace framework
}  // namespace paddle
