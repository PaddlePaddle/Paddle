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

#include "paddle/operators/math/seq2batch.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T>
class Seq2BatchFunctor<true, platform::CPUPlace, T> {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::LoDTensor& seq, framework::Tensor& batch,
                  bool norm_by_times) {
    auto lod = seq.lod();
    PADDLE_ENFORCE_GT(lod.size(), 0UL,
                      "The lod of LoDTensor seq should not be null.");

    const size_t level = 0;
    framework::LoD abs_offset_lod = framework::ToAbsOffset(lod);
    const size_t num_sequences = abs_offset_lod[level].size() - 1;

    // Compute maximum sequence length
    const size_t max_sequence_length = MaximumSequenceLength(lod, level);

    auto seq_dims = seq.dims();
    PADDLE_ENFORCE_EQ(seq_dims[0], abs_offset_lod[level].back(),
                      "The first dimension of LoDTensor seq should be "
                      "equal to the sum of all sequences's length.");

    const size_t sequence_width = seq.numel() / seq_dims[0];
    auto batch_dims =
        framework::make_ddim({static_cast<int64_t>(max_sequence_length),
                              static_cast<int64_t>(num_sequences),
                              static_cast<int64_t>(sequence_width)});

    const T* seq_data = seq.data<T>();
    T* batch_data = batch.mutable_data<T>(batch_dims, context.GetPlace());
    for (size_t i = 0; i < max_sequence_length; ++i) {
      for (size_t j = 0; j < num_sequences; ++j) {
        size_t start_pos = abs_offset_lod[level][j];
        size_t sequence_length = abs_offset_lod[level][j + 1] - start_pos;
        if (i < sequence_length) {
          // i > 0 => sequence_length > 0
          T scale =
              norm_by_times ? (1.0f / static_cast<T>(sequence_length)) : 1.0f;
          for (size_t k = 0; k < sequence_width; ++k) {
            batch_data[(i * num_sequences + j) * sequence_width + k] =
                seq_data[(start_pos + i) * sequence_width + k] * scale;
          }
        } else {
          memset(batch_data + (i * num_sequences + j) * sequence_width, 0,
                 sequence_width * sizeof(T));
        }
      }
    }
  }
};

template <typename T>
class Batch2SeqFunctor<true, platform::CPUPlace, T> {
 public:
  void operator()(const platform::DeviceContext& context,
                  framework::LoDTensor& seq, const framework::Tensor& batch,
                  bool norm_by_times) {
    auto lod = seq.lod();
    PADDLE_ENFORCE_GT(lod.size(), 0UL,
                      "The lod of LoDTensor seq should not be null.");

    const size_t level = 0;
    framework::LoD abs_offset_lod = framework::ToAbsOffset(lod);
    const size_t num_sequences = abs_offset_lod[level].size() - 1;

    // Compute maximum sequence length
    const size_t max_sequence_length = MaximumSequenceLength(lod, level);

    auto batch_dims = batch.dims();
    PADDLE_ENFORCE_EQ(batch_dims.size(), 3UL,
                      "The input batch should be a 3-D Tensor.");
    PADDLE_ENFORCE_EQ(batch_dims[0], max_sequence_length,
                      "The first dimension of Tensor batch should be "
                      "equal to the maximum sequence's length.");
    PADDLE_ENFORCE_EQ(batch_dims[1], num_sequences,
                      "The second dimension of Tensor batch should be "
                      "equal to the number of sequences.");

    const size_t sequence_width = batch_dims[2];
    auto seq_dims = framework::make_ddim(
        {static_cast<int64_t>(abs_offset_lod[level].back()),
         static_cast<int64_t>(sequence_width)});

    const T* batch_data = batch.data<T>();
    T* seq_data = seq.mutable_data<T>(seq_dims, context.GetPlace());
    for (size_t i = 0; i < num_sequences; ++i) {
      size_t start_pos = abs_offset_lod[level][i];
      size_t sequence_length = abs_offset_lod[level][i + 1] - start_pos;
      for (size_t j = 0; j < sequence_length; ++j) {
        // sequence_width > j > 0
        T scale =
            norm_by_times ? (1.0f / static_cast<T>(sequence_length)) : 1.0f;
        for (size_t k = 0; k < sequence_width; ++k) {
          seq_data[(start_pos + j) * sequence_width + k] =
              batch_data[(j * num_sequences + i) * sequence_width + k] * scale;
        }
      }
    }
  }
};

template class Seq2BatchFunctor<true, platform::CPUPlace, float>;
template class Batch2SeqFunctor<true, platform::CPUPlace, float>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
