// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

namespace phi {

template <typename T, typename Context>
void MoeKernel(const Context& ctx,
               const DenseTensor& x,
               const DenseTensor& gate,
               const DenseTensor& bmm0,
               const DenseTensor& bias0,
               const DenseTensor& bmm1,
               const DenseTensor& bias1,
               const std::string& act_type,
               DenseTensor* output);

}  // namespace phi
