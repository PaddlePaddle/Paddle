
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
#include "paddle/phi/core/dense_tensor.h"

namespace phi {

template <typename T, typename Context>
void SqueezeKernel(const Context& dev_ctx,
                   const DenseTensor& x,
<<<<<<< HEAD
                   const std::vector<int>& axes,
=======
                   const IntArray& axes,
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
                   DenseTensor* out);

template <typename T, typename Context>
void SqueezeWithXShapeKernel(const Context& dev_ctx,
                             const DenseTensor& x,
<<<<<<< HEAD
                             const std::vector<int>& axes,
=======
                             const IntArray& axes,
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
                             DenseTensor* out,
                             DenseTensor* xshape);

}  // namespace phi
