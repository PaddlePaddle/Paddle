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

void Assign(const phi::DenseTensor& in, phi::DenseTensor* out);

void Transpose2D(phi::DenseTensor* in, phi::DenseTensor* out = nullptr);

void CastToFp32(phi::DenseTensor* in, phi::DenseTensor* out = nullptr);

void CastToInt32(phi::DenseTensor* in, phi::DenseTensor* out = nullptr);

template <typename T>
void ConvertWithoutQuant(phi::DenseTensor* weight,
                         phi::DenseTensor* weight_max,
                         bool transpose,
                         const std::vector<float>& weight_scales);

template <typename Tcpu,
          typename Txpu,
          typename std::enable_if<std::is_same<Tcpu, float>::value, Tcpu>::type*
              ptr = nullptr>
void ConvertWithQuant(phi::DenseTensor* weight,
                      phi::DenseTensor* weight_max,
                      bool transpose,
                      const std::vector<float>& weight_scales);

template <typename Tcpu,
          typename Txpu,
          typename std::enable_if<!std::is_same<Tcpu, float>::value,
                                  Tcpu>::type* ptr = nullptr>
void ConvertWithQuant(phi::DenseTensor* weight,
                      phi::DenseTensor* weight_max,
                      bool transpose,
                      const std::vector<float>& weight_scales);

// 1. Quant weight from fp32 to int16/int31
// 2. Weight data is in-place update.
// 3. Generate weight max tensor
template <typename T>
void PrepareWeight(phi::DenseTensor* weight,
                   phi::DenseTensor* weight_max,
                   bool transpose);

bool IsPerTensorQuant(const std::vector<float>& weight_max);

}  // namespace ir
}  // namespace framework
}  // namespace paddle
