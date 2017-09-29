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
  for (size_t i = 0; i < num_sequences; ++i) {
    max_sequence_length =
        std::max(max_sequence_length, lod[level][i + 1] - lod[level][i]);
  }
  return max_sequence_length;
}

/*
 * \brief
 *
 * \param
 * \param
 *
 * \note
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
