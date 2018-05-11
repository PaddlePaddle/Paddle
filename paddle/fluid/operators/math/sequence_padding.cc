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

template <typename T, PaddingLayout padding_layout>
void CopyDataCPU(framework::LoDTensor* seq_tensor,
                 framework::Tensor* padding_tensor,
                 const framework::Vector<size_t>& abs_offset,
                 const int64_t& max_seq_len, const int64_t& seq_width,
                 bool seq_to_padding, bool norm_by_len) {
  T* seq_data = seq_tensor->data<T>();
  T* padding_data = padding_tensor->data<T>();

  int64_t seq_num = abs_offset.size() - 1;

  for (int64_t i = 0; i < seq_num; ++i) {
    int64_t seq_start = abs_offset[i];
    int64_t seq_len = abs_offset[i + 1] - seq_start;

    T scale = norm_by_len ? (1.0f / static_cast<T>(seq_len)) : 1.0f;

    for (int64_t j = 0; j < seq_len; ++j) {
      for (int64_t k = 0; k < seq_width; ++k) {
        size_t padding_offset = 0;
        if (padding_layout == BATCH_LENGTH_WIDTH) {
          padding_offset = (i * max_seq_len * seq_width) + j * seq_width + k;
        } else {
          padding_offset = (j * seq_num * seq_width) + i * seq_width + k;
        }
        if (seq_to_padding) {
          padding_data[padding_offset] =
              seq_data[(seq_start + j) * seq_width + k] * scale;
        } else {
          seq_data[(seq_start + j) * seq_width + k] =
              padding_data[padding_offset] * scale;
        }
      }
    }
  }
}

template <typename T, PaddingLayout padding_layout>
class PaddingLoDTensorFunctor<platform::CPUDeviceContext, T, padding_layout> {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::LoDTensor& seq_tensor,
                  framework::Tensor* padding_tensor,
                  T padding_value = static_cast<T>(0),
                  bool norm_by_times = false, size_t lod_level = 0) {
    ValidateLoD(seq_tensor, lod_level);

    auto& lod = seq_tensor.lod();
    auto& abs_offset = framework::ToAbsOffset(lod)[lod_level];

    auto seq_dims = seq_tensor.dims();
    auto padding_dims = padding_tensor->dims();
    int64_t max_seq_len = MaximumSequenceLength(lod, lod_level);
    int64_t seq_num = abs_offset.size() - 1;
    int64_t seq_width = seq_tensor.numel() / seq_dims[0];
    int64_t numel = max_seq_len * seq_num * seq_width;

    ValidateShape(seq_dims, abs_offset.back(), padding_dims, max_seq_len,
                  seq_num, seq_width, padding_layout);

    T* padding_data = padding_tensor->data<T>();

    memset(padding_data, padding_value, numel * sizeof(T));

    CopyDataCPU<T, padding_layout>(
        const_cast<framework::LoDTensor*>(&seq_tensor), padding_tensor,
        abs_offset, max_seq_len, seq_width, true /* seq_to_padding */,
        norm_by_times);
  }
};

template <typename T, PaddingLayout padding_layout>
class UnpaddingLoDTensorFunctor<platform::CPUDeviceContext, T, padding_layout> {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  framework::LoDTensor* seq_tensor,
                  const framework::Tensor& padding_tensor,
                  bool norm_by_times = false, size_t lod_level = 0) {
    ValidateLoD(*seq_tensor, lod_level);

    auto& lod = seq_tensor->lod();
    auto& abs_offset = framework::ToAbsOffset(lod)[lod_level];

    auto& seq_dims = seq_tensor->dims();
    auto& padding_dims = padding_tensor.dims();
    int64_t max_seq_len = MaximumSequenceLength(lod, lod_level);
    int64_t seq_num = abs_offset.size() - 1;
    int64_t seq_width = seq_tensor->numel() / seq_dims[0];

    ValidateShape(seq_dims, abs_offset.back(), padding_dims, max_seq_len,
                  seq_num, seq_width, padding_layout);

    T* seq_data = seq_tensor->data<T>();
    memset(seq_data, static_cast<T>(0), seq_tensor->numel() * sizeof(T));

    CopyDataCPU<T, padding_layout>(
        seq_tensor, const_cast<framework::Tensor*>(&padding_tensor), abs_offset,
        max_seq_len, seq_width, false /* seq_to_padding */, norm_by_times);
  }
};

template class PaddingLoDTensorFunctor<platform::CPUDeviceContext, float,
                                       LENGTH_BATCH_WIDTH>;
template class UnpaddingLoDTensorFunctor<platform::CPUDeviceContext, float,
                                         LENGTH_BATCH_WIDTH>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
