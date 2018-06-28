/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <algorithm>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace operators {
namespace math {

inline static size_t MaximumSequenceLength(const framework::LoD& lod,
                                           const size_t level) {
  const size_t num_sequences = lod[level].size() - 1;
  size_t max_sequence_length = 0;
  framework::LoD abs_offset_lod = framework::ToAbsOffset(lod);
  for (size_t i = 0; i < num_sequences; ++i) {
    max_sequence_length =
        std::max(max_sequence_length,
                 abs_offset_lod[level][i + 1] - abs_offset_lod[level][i]);
  }
  return max_sequence_length;
}

/*
 * \brief   Padding/Unpadding LoDTensor to/from normal Tensor of the shape
 *          [max_sequence_length, num_sequences, sequence_width].
 *
 *  Padding sequence:
 *        padding[i] = seq[lod[level][i]]
 *  Unpadding sequence:
 *        seq[lod[level][i]] = padding[i]
 *
 *  All sequences will be padded to the same length and stored in a transposed
 * shape.
 *  Example:
 *    seq     (s0, s0, s0, s0; s1, s1; s2, s2, s2; s3)
 *    padding (s0, s1, s2, s3; s0, s1, s2, 0; s0, 0, s2, 0; s0, 0, 0, 0)
 *
 * \param context       device context of this functor.
 * \param seq           LoDTensor which is stored in sequence format, the shape
 *                      is [total_sequence_length, sequence_width] where
 *                      total_sequence_length is the sum of all sequences'
 *                      length.
 * \param padding       Tensor which is padded to the same length, the shape is
 *                      [max_sequence_length, num_sequences, sequence_width].
 * \param norm_by_times whether dividing sequence's length.
 *
 * \note  transposition is also done in this functor.
 */
template <typename DeviceContext, typename T>
class PaddingLoDTensorFunctor {
 public:
  void operator()(const DeviceContext& context, const framework::LoDTensor& seq,
                  framework::Tensor* padding, bool norm_by_times);
};

template <typename DeviceContext, typename T>
class UnpaddingLoDTensorFunctor {
 public:
  void operator()(const DeviceContext& context, framework::LoDTensor* seq,
                  const framework::Tensor& padding, bool norm_by_times);
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
