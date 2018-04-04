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

#include "paddle/fluid/operators/math/sequence_padding.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T>
class PaddingLoDTensorFunctor<platform::CPUDeviceContext, T> {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::LoDTensor& seq, framework::Tensor* padding,
                  bool norm_by_times) {
    auto lod = seq.lod();
    PADDLE_ENFORCE_GT(lod.size(), 0UL,
                      "The LoD of LoDTensor seq should not be null.");

    const size_t level = 0;
    framework::LoD abs_offset_lod = framework::ToAbsOffset(lod);

    auto seq_dims = seq.dims();
    PADDLE_ENFORCE_EQ(seq_dims[0],
                      static_cast<int64_t>(abs_offset_lod[level].back()),
                      "The first dimension of LoDTensor seq should be "
                      "equal to the sum of all sequences's length.");

    auto padding_dims = padding->dims();
    PADDLE_ENFORCE_EQ(padding_dims.size(), 3UL,
                      "The input padding should be a 3-D Tensor of shape "
                      "[max_sequence_length, num_sequences, sequence_width].");

    const int64_t max_sequence_length = MaximumSequenceLength(lod, level);
    PADDLE_ENFORCE_EQ(padding_dims[0], max_sequence_length,
                      "The first dimension of Tensor padding should be the "
                      "maximum length of all sequences in LoDTensor seq.");

    const int64_t num_sequences = abs_offset_lod[level].size() - 1;
    PADDLE_ENFORCE_EQ(padding_dims[1], num_sequences,
                      "The second dimension of Tensor padding should be the "
                      "number of sequences in LoDTensor seq.");

    const int64_t sequence_width = seq.numel() / seq_dims[0];
    PADDLE_ENFORCE_EQ(padding_dims[2], sequence_width,
                      "The third dimension of Tensor padding should be the "
                      "width of sequence in LoDTensor seq.");

    const T* seq_data = seq.data<T>();
    T* padding_data = padding->data<T>();
    for (int64_t i = 0; i < max_sequence_length; ++i) {
      for (int64_t j = 0; j < num_sequences; ++j) {
        int64_t start_pos = abs_offset_lod[level][j];
        int64_t sequence_length = abs_offset_lod[level][j + 1] - start_pos;
        if (i < sequence_length) {
          // i > 0 => sequence_length > 0
          T scale =
              norm_by_times ? (1.0f / static_cast<T>(sequence_length)) : 1.0f;
          for (int64_t k = 0; k < sequence_width; ++k) {
            padding_data[(i * num_sequences + j) * sequence_width + k] =
                seq_data[(start_pos + i) * sequence_width + k] * scale;
          }
        } else {
          memset(padding_data + (i * num_sequences + j) * sequence_width, 0,
                 sequence_width * sizeof(T));
        }
      }
    }
  }
};

template <typename T>
class UnpaddingLoDTensorFunctor<platform::CPUDeviceContext, T> {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  framework::LoDTensor* seq, const framework::Tensor& padding,
                  bool norm_by_times) {
    auto lod = seq->lod();
    PADDLE_ENFORCE_GT(lod.size(), 0UL,
                      "The LoD of LoDTensor seq should not be null.");

    const size_t level = 0;
    framework::LoD abs_offset_lod = framework::ToAbsOffset(lod);

    auto seq_dims = seq->dims();
    PADDLE_ENFORCE_EQ(seq_dims[0],
                      static_cast<int64_t>(abs_offset_lod[level].back()),
                      "The first dimension of LoDTensor seq should be "
                      "equal to the sum of all sequences's length.");

    auto padding_dims = padding.dims();
    PADDLE_ENFORCE_EQ(padding_dims.size(), 3UL,
                      "The input padding should be a 3-D Tensor of shape "
                      "[max_sequnece_length, num_sequences, sequence_width].");

    const int64_t max_sequence_length = MaximumSequenceLength(lod, level);
    PADDLE_ENFORCE_EQ(padding_dims[0], max_sequence_length,
                      "The first dimension of Tensor padding should be "
                      "the maximum length of all sequences in LoDTensor seq.");

    const int64_t num_sequences = abs_offset_lod[level].size() - 1;
    PADDLE_ENFORCE_EQ(padding_dims[1], num_sequences,
                      "The second dimension of Tensor padding should be "
                      "the number of sequences in LoDTensor seq.");

    const int64_t sequence_width = seq->numel() / seq_dims[0];
    PADDLE_ENFORCE_EQ(padding_dims[2], sequence_width,
                      "The third dimension of Tensor padding should be the "
                      "width of sequence in LoDTensor seq.");

    const T* padding_data = padding.data<T>();
    T* seq_data = seq->data<T>();
    for (int64_t i = 0; i < num_sequences; ++i) {
      int64_t start_pos = abs_offset_lod[level][i];
      int64_t sequence_length = abs_offset_lod[level][i + 1] - start_pos;
      for (int64_t j = 0; j < sequence_length; ++j) {
        // sequence_width > j > 0
        T scale =
            norm_by_times ? (1.0f / static_cast<T>(sequence_length)) : 1.0f;
        for (int64_t k = 0; k < sequence_width; ++k) {
          seq_data[(start_pos + j) * sequence_width + k] =
              padding_data[(j * num_sequences + i) * sequence_width + k] *
              scale;
        }
      }
    }
  }
};

template class PaddingLoDTensorFunctor<platform::CPUDeviceContext, float>;
template class UnpaddingLoDTensorFunctor<platform::CPUDeviceContext, float>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
