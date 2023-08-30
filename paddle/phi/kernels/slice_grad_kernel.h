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
#include "paddle/phi/core/tensor_array.h"

namespace phi {

template <typename T, typename Context>
void SliceGradKernel(const Context& ctx,
                     const DenseTensor& input,
                     const DenseTensor& out_grad,
                     const std::vector<int64_t>& axes,
                     const IntArray& starts,
                     const IntArray& ends,
                     const std::vector<int64_t>& infer_flags,
                     const std::vector<int64_t>& decrease_axis,
                     DenseTensor* input_grad);

template <typename Context>
void SliceGradStridedKernel(const Context& ctx,
                            const DenseTensor& input,
                            const DenseTensor& out_grad,
                            const std::vector<int64_t>& axes,
                            const IntArray& starts,
                            const IntArray& ends,
                            const std::vector<int64_t>& infer_flags,
                            const std::vector<int64_t>& decrease_axis,
                            DenseTensor* input_grad);

template <typename T, typename Context>
void SliceArrayGradKernel(const Context& dev_ctx,
                          const TensorArray& input,
                          const TensorArray& out_grad,
                          const IntArray& starts,
                          const IntArray& ends,
                          TensorArray* input_grad);

template <typename T, typename Context>
void SliceArrayDenseGradKernel(const Context& dev_ctx,
                               const TensorArray& input,
                               const DenseTensor& out_grad,
                               const IntArray& starts,
                               TensorArray* input_grad);

}  // namespace phi
