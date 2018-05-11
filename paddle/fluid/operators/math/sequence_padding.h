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

enum PaddingLayout { BATCH_LENGTH_WIDTH, LENGTH_BATCH_WIDTH };

inline static size_t MaximumSequenceLength(const framework::LoD& lod,
                                           const size_t level) {
  const size_t seq_num = lod[level].size() - 1;
  size_t max_seq_len = 0;
  auto abs_offset = framework::ToAbsOffset(lod)[level];
  for (size_t i = 0; i < seq_num; ++i) {
    max_seq_len = std::max(max_seq_len, abs_offset[i + 1] - abs_offset[i]);
  }
  return max_seq_len;
}

inline static void ValidateLoD(const framework::LoDTensor& seq_tensor,
                               const size_t& lod_level) {
  PADDLE_ENFORCE(lod_level < seq_tensor.lod().size(),
                 "Invalid `lod_level` which should be at least 0 and less "
                 "than maximum lod level of `seq_tensor`.");
}

inline static void ValidateShape(const framework::DDim& seq_tensor_dims,
                                 const size_t& abs_offset_back_value,
                                 const framework::DDim& padding_tensor_dims,
                                 const int64_t& max_seq_len,
                                 const int64_t& seq_num,
                                 const int64_t& seq_width,
                                 const PaddingLayout& padding_layout) {
  PADDLE_ENFORCE_EQ(static_cast<size_t>(seq_tensor_dims[0]),
                    abs_offset_back_value,
                    "The 1st dimension of `seq_tensor` should be equal to "
                    "sum of lengths of all sequences.");

  PADDLE_ENFORCE_EQ(padding_tensor_dims.size(), 3UL,
                    "`padding_tensor` should be a 3-D tensor.");

  if (padding_layout == BATCH_LENGTH_WIDTH) {
    PADDLE_ENFORCE_EQ(padding_tensor_dims,
                      framework::make_ddim({seq_num, max_seq_len, seq_width}));
  } else if (padding_layout == LENGTH_BATCH_WIDTH) {
    PADDLE_ENFORCE_EQ(padding_tensor_dims,
                      framework::make_ddim({max_seq_len, seq_num, seq_width}));
  } else {
    PADDLE_THROW("Unsupported padding layout.");
  }
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
template <typename DeviceContext, typename T, PaddingLayout padding_layout>
class PaddingLoDTensorFunctor {
 public:
  void operator()(const DeviceContext& context,
                  const framework::LoDTensor& seq_tensor,
                  framework::Tensor* padding_tensor,
                  T padding_value = static_cast<T>(0),
                  bool norm_by_times = false, size_t lod_level = 0);
};

template <typename DeviceContext, typename T, PaddingLayout padding_layout>
class UnpaddingLoDTensorFunctor {
 public:
  void operator()(const DeviceContext& context,
                  framework::LoDTensor* seq_tensor,
                  const framework::Tensor& padding_tensor,
                  bool norm_by_times = false, size_t lod_level = 0);
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
