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

namespace phi {

template <typename Context>
void ViewShapeKernel(const Context& dev_ctx,
                     const DenseTensor& input,
                     const std::vector<int64_t>& dims,
                     DenseTensor* out);

template <typename Context>
void ViewDtypeKernel(const Context& dev_ctx,
                     const DenseTensor& input,
                     DataType dtype,
                     DenseTensor* out);
}  // namespace phi
