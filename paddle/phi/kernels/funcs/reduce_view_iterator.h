// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <vector>

#include "paddle/phi/kernels/funcs/dense_tensor_iterator.h"

namespace phi {

/**
 * ReduceViewIterator: reduction iterator over one output and one input whose
 * shape and strides (in elements) are given as vectors instead of read from
 * the tensors' DDim, so the view may have more dims than DDim::kMaxRank.
 *
 * `out_shape` has 1 at the reduced dims. The result equals what
 * DenseTensorIteratorConfig::is_reduction(true).build() gives for tensors of
 * these shapes, and converts to DenseTensorIterator for GPUReduceScheduler.
 */
struct ReduceViewIterator : public DenseTensorIteratorBase {
  ReduceViewIterator(const DenseTensor& out,
                     const std::vector<int64_t>& out_shape,
                     const std::vector<int64_t>& out_strides,
                     const DenseTensor& in,
                     const std::vector<int64_t>& in_shape,
                     const std::vector<int64_t>& in_strides) {
    is_reduction_ = true;
    num_outputs_ = 1;
    shape_ = in_shape;
    AddOperand(out, out_shape, out_strides, /*is_output=*/true);
    AddOperand(in, in_shape, in_strides, /*is_output=*/false);
    reorder_dimensions();
    coalesce_dimensions();
    view_offsets_.assign(std::max(ndim(), 1), 0);
  }

 private:
  void AddOperand(const DenseTensor& t,
                  const std::vector<int64_t>& shape,
                  const std::vector<int64_t>& strides,
                  bool is_output) {
    auto& op = operands_.emplace_back(const_cast<DenseTensor*>(&t));
    op.is_output = is_output;
    op.data = const_cast<void*>(t.data());
    const int64_t elem_size = static_cast<int64_t>(phi::SizeOf(t.dtype()));
    op.stride_bytes.resize(ndim());
    for (int i = 0; i < ndim(); ++i) {
      // Broadcast dims read the same element, as in compute_strides.
      op.stride_bytes[i] =
          (shape[i] == 1 && shape_[i] != 1) ? 0 : strides[i] * elem_size;
    }
  }
};

}  // namespace phi
