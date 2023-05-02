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

#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {

template <typename T, typename Context>
void DropoutRawKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const paddle::optional<DenseTensor>& seed_tensor,
                      const Scalar& p,
                      bool is_test,
                      const std::string& mode,
                      int seed,
                      bool fix_seed,
                      DenseTensor* out,
                      DenseTensor* mask);

template <typename T, typename Context>
void DropoutNdKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const paddle::optional<DenseTensor>& seed_tensor,
                     const Scalar& p,
                     bool is_test,
                     const std::string& mode,
                     int seed,
                     bool fix_seed,
                     const std::vector<int>& axis,
                     DenseTensor* out,
                     DenseTensor* mask);

}  // namespace phi
