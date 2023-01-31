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
#include "paddle/phi/core/selected_rows.h"

namespace phi {
namespace sr {

template <typename T, typename Context>
void UniformRawKernel(const Context& dev_ctx,
                      const IntArray& shape,
                      DataType dtype,
                      const Scalar& min,
                      const Scalar& max,
                      int seed,
                      int diag_num,
                      int diag_step,
                      float diag_val,
                      SelectedRows* out);

template <typename T, typename Context>
void UniformKernel(const Context& dev_ctx,
                   const IntArray& shape,
                   DataType dtype,
                   const Scalar& min,
                   const Scalar& max,
                   int seed,
                   SelectedRows* out);

}  // namespace sr
}  // namespace phi
