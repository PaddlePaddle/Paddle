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
#include "paddle/phi/kernels/strided_slice_grad_kernel.h"

#include "paddle/phi/kernels/funcs/strided_slice.h"

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
                               DenseTensor* x_grad) {
  int rank = x.dims().size();
#define SLICE_CASE(Rank)                                            \
  case Rank:                                                        \
    funcs::StridedSliceGradCompute<Context, T, Rank>(dev_ctx,       \
                                                     x,             \
                                                     out_grad,      \
                                                     axes,          \
                                                     starts,        \
                                                     ends,          \
                                                     strides,       \
                                                     infer_flags,   \
                                                     decrease_axis, \
                                                     x_grad);       \
    break;

  switch (rank) {
    SLICE_CASE(1)
    SLICE_CASE(2)
    SLICE_CASE(3)
    SLICE_CASE(4)
    SLICE_CASE(5)
    SLICE_CASE(6)
  }
#undef SLICE_CASE
}

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
    std::vector<DenseTensor*> x_grad) {
  funcs::StridedSliceGradCompute<Context, T, 1>(dev_ctx,
                                                x,
                                                out_grad,
                                                axes,
                                                starts,
                                                ends,
                                                strides,
                                                infer_flags,
                                                decrease_axis,
                                                x_grad);
}

}  // namespace phi
