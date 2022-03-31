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
void StridedSliceRawGradKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const DenseTensor& out_grad,
                               const std::vector<int>& axes,
                               const IntArray& starts,
                               const IntArray& ends,
                               const IntArray& strides,
                               const std::vector<int>& infer_flags,
                               const std::vector<int>& decrease_axis,
                               DenseTensor* x_grad);

template <typename T, typename Context>
void StridedSliceGradKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& out_grad,
                            const std::vector<int>& axes,
                            const IntArray& starts,
                            const IntArray& ends,
                            const IntArray& strides,
                            DenseTensor* x_grad);

template <typename T, typename Context>
void StridedSliceArrayGradKernel(
    const Context& dev_ctx,
    const std::vector<const DenseTensor*>& x,
    const std::vector<const DenseTensor*>& out_grad,
    const std::vector<int>& axes,
    const IntArray& starts,
    const IntArray& ends,
    const IntArray& strides,
    const std::vector<int>& infer_flags,
    const std::vector<int>& decrease_axis,
    std::vector<DenseTensor*> x_grad);
}  // namespace phi
