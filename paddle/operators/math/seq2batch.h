/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include "paddle/framework/lod_tensor.h"
#include "paddle/platform/device_context.h"

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
 * \brief   Memory copy from sequence/batch to batch/sequence
 *
 *  Copy from sequence to batch:
 *        batch[i] = seq[lod[level][i]]
 *  Copy from batch to seq:
 *        seq[lod[level][i]] = batch[i]
 *
 *  When Padding is true, all sequences will be padded to the same length.
 *  Example:
 *    seq   (s0, s0, s0, s0; s1, s1; s2, s2, s2; s3)
 *    batch (s0, s1, s2, s3; s0, s1, s2, 0; s0, 0, s2, 0; s0, 0, 0, 0)
 *
 * \param context       device context of this functor.
 * \param seq           LoDTensor which is stored in sequence format.
 * \param batch         Tensor which is stored in batch format.
 * \param norm_by_times whether dividing sequence's length.
 */
template <bool Padding, typename Place, typename T>
class Seq2BatchFunctor {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::LoDTensor& seq, framework::Tensor& batch,
                  bool norm_by_times);
};

template <bool Padding, typename Place, typename T>
class Batch2SeqFunctor {
 public:
  void operator()(const platform::DeviceContext& context,
                  framework::LoDTensor& seq, const framework::Tensor& batch,
                  bool norm_by_times);
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
