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
#include <vector>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace operators {
namespace math {

enum PadLayout { kBatchLengthWidth = 0, kLengthBatchWidth };

enum CopyType { kSeqToPad, kPadToSeq };

inline static size_t MaximumSequenceLength(
    const framework::Vector<size_t>& seq_offset) {
  size_t seq_num = seq_offset.size() - 1;
  size_t max_seq_len = 0;
  for (size_t i = 0; i < seq_num; ++i) {
    max_seq_len = std::max(max_seq_len, seq_offset[i + 1] - seq_offset[i]);
  }
  return max_seq_len;
}

inline static size_t TotalSequenceLength(
    const framework::Vector<size_t>& seq_offset) {
  size_t seq_num = seq_offset.size() - 1;
  size_t total_seq_len = 0;
  for (size_t i = 0; i < seq_num; ++i) {
    total_seq_len += seq_offset[i + 1] - seq_offset[i];
  }
  return total_seq_len;
}

inline static void CheckDims(const framework::DDim& seq_tensor_dims,
                             const framework::DDim& pad_tensor_dims,
                             const framework::Vector<size_t>& seq_offset,
                             int64_t padded_seq_len, int64_t step_width,
                             const PadLayout& layout) {
  PADDLE_ENFORCE_EQ(
      static_cast<size_t>(seq_tensor_dims[0]), seq_offset.back(),
      platform::errors::InvalidArgument(
          "Value of 1st dimension of the sequence tensor should be "
          "equal to sum of lengths of all sequences. Expected %ld == %ld, but "
          "got %ld != %ld. Please check the input value.",
          static_cast<size_t>(seq_tensor_dims[0]), seq_offset.back(),
          static_cast<size_t>(seq_tensor_dims[0]), seq_offset.back()));

  PADDLE_ENFORCE_EQ(
      seq_tensor_dims.size() + 1 == pad_tensor_dims.size() ||
          seq_tensor_dims.size() == pad_tensor_dims.size(),
      true, platform::errors::InvalidArgument(
                "pad_tensor's rank should be 1 greater than seq_tensor's "
                "rank, or be equal with it. The pad_tensor's rank is %ld, "
                "expected the seq_tensor's rank is %ld or %ld, but got %ld. "
                "Please check the input value.",
                pad_tensor_dims.size(), pad_tensor_dims.size(),
                pad_tensor_dims.size() - 1, seq_tensor_dims.size()));
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
                  framework::LoDTensor* pad_tensor,
                  const framework::LoDTensor& pad_value, int pad_seq_len = -1,
                  int lod_level = 0, bool norm_by_times = false,
                  bool norm_by_batchsize = false,
                  bool norm_by_total_logits_len = false,
                  const PadLayout layout = kBatchLengthWidth);
};

template <typename DeviceContext, typename T>
class UnpaddingLoDTensorFunctor {
 public:
  void operator()(const DeviceContext& context,
                  const framework::LoDTensor& pad_tensor,
                  framework::LoDTensor* seq_tensor, int pad_seq_len = -1,
                  int lod_level = 0, bool norm_by_times = false,
                  bool norm_by_batchsize = false,
                  bool norm_by_total_logits_len = false,
                  const PadLayout layout = kBatchLengthWidth);
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
