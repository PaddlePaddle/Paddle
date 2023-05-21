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
#include <glog/logging.h>
#include "gflags/gflags.h"
#include "paddle/phi/kernels/funcs/strided_slice.h"
#include "paddle/phi/kernels/strided_slice_kernel.h"
DECLARE_string(throw_strided_error_op);

namespace phi {

template <typename T, typename Context>
void StridedSliceRawKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           const std::vector<int>& axes,
                           const IntArray& starts,
                           const IntArray& ends,
                           const IntArray& strides,
                           const std::vector<int>& infer_flags,
                           const std::vector<int>& decrease_axis,
                           DenseTensor* out) {
  DenseTensor& xx = const_cast<DenseTensor&>(x);
  if (!xx.IsSharedBufferWith(*out)) {
    if (xx.can_not_uses != out->can_not_uses) {
      out->can_not_uses = xx.can_not_uses;
      if (*out->canNotUse == false) {
        *out->canNotUse = *xx.canNotUse;
      }
      xx.can_not_uses->insert(xx.canNotUse);
      xx.can_not_uses->insert(out->canNotUse);
      VLOG(1) << "stride api call log: StridedSliceRawKernel";

      if (FLAGS_throw_strided_error_op == "StridedSliceRawKernel") {
        PADDLE_THROW(phi::errors::PermissionDenied("wanghuan"));
      }
    }
  }
  int rank = x.dims().size();
#define SLICE_CASE(Rank)                                        \
  case Rank:                                                    \
    funcs::StridedSliceCompute<Context, T, Rank>(dev_ctx,       \
                                                 x,             \
                                                 axes,          \
                                                 starts,        \
                                                 ends,          \
                                                 strides,       \
                                                 infer_flags,   \
                                                 decrease_axis, \
                                                 out);          \
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
void StridedSliceArrayKernel(const Context& dev_ctx,
                             const TensorArray& x,
                             const std::vector<int>& axes,
                             const IntArray& starts,
                             const IntArray& ends,
                             const IntArray& strides,
                             const std::vector<int>& infer_flags,
                             const std::vector<int>& decrease_axis,
                             TensorArray* out) {
  funcs::StridedSliceCompute<Context, T, 1>(
      dev_ctx, x, axes, starts, ends, strides, infer_flags, decrease_axis, out);
}

}  // namespace phi
