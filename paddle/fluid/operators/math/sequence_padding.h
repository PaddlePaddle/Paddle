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

enum OutputLayout { kBatchLengthWidth = 0, kLengthBatchWidth };

inline static size_t MaximumSequenceLength(
    const framework::Vector<size_t>& seq_offset) {
  size_t seq_num = seq_offset.size() - 1;
  size_t max_seq_len = 0;
  for (size_t i = 0; i < seq_num; ++i) {
    max_seq_len = std::max(max_seq_len, seq_offset[i + 1] - seq_offset[i]);
  }
  return max_seq_len;
}

inline static void CheckLoD(const framework::LoDTensor& seq_tensor,
                            const size_t& lod_level) {
  PADDLE_ENFORCE(lod_level < seq_tensor.lod().size(),
                 "Invalid lod level which should be at least 0 and less "
                 "than maximum lod level of sequence tensor.");
}

inline static void CheckDims(const framework::DDim& seq_tensor_dims,
                             const size_t& last_offset,
                             const framework::DDim& pad_tensor_dims,
                             const int64_t& max_seq_len, const int64_t& seq_num,
                             const int64_t& seq_width,
                             const OutputLayout& output_layout) {
  PADDLE_ENFORCE_EQ(static_cast<size_t>(seq_tensor_dims[0]), last_offset,
                    "Value of 1st dimension of the sequence tensor should be "
                    "equal to sum of lengths of all sequences.");

  PADDLE_ENFORCE_EQ(pad_tensor_dims.size(), 3UL,
                    "Padded tensor should be a 3-D tensor.");

  if (output_layout == kBatchLengthWidth) {
    PADDLE_ENFORCE_EQ(pad_tensor_dims,
                      framework::make_ddim({seq_num, max_seq_len, seq_width}));
  } else if (output_layout == kLengthBatchWidth) {
    PADDLE_ENFORCE_EQ(pad_tensor_dims,
                      framework::make_ddim({max_seq_len, seq_num, seq_width}));
  } else {
    PADDLE_THROW("Unsupported output layout.");
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
template <typename DeviceContext, typename T>
class PaddingLoDTensorFunctor {
 public:
  void operator()(const DeviceContext& context,
                  const framework::LoDTensor& seq_tensor,
                  framework::Tensor* pad_tensor,
                  T pad_value = static_cast<T>(0), bool norm_by_times = false,
                  size_t lod_level = 0,
                  OutputLayout output_layout = kBatchLengthWidth);
};

template <typename DeviceContext, typename T>
class UnpaddingLoDTensorFunctor {
 public:
  void operator()(const DeviceContext& context,
                  framework::LoDTensor* seq_tensor,
                  const framework::Tensor& pad_tensor,
                  bool norm_by_times = false, size_t lod_level = 0,
                  OutputLayout output_layout = kBatchLengthWidth);
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
