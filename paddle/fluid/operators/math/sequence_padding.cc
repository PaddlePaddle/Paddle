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
void CopyDataCPU(framework::LoDTensor* seq_tensor,
                 framework::Tensor* pad_tensor,
                 const framework::Vector<size_t>& seq_offset,
                 const int64_t& max_seq_len, const int64_t& seq_width,
                 bool seq_to_pad, bool norm_by_len,
                 OutputLayout output_layout) {
  T* seq_data = seq_tensor->data<T>();
  T* pad_data = pad_tensor->data<T>();

  int64_t seq_num = seq_offset.size() - 1;

  for (int64_t i = 0; i < seq_num; ++i) {
    int64_t seq_start = seq_offset[i];
    int64_t seq_len = seq_offset[i + 1] - seq_start;
    T scale = norm_by_len ? (1.0f / static_cast<T>(seq_len)) : 1.0f;
    for (int64_t j = 0; j < seq_len; ++j) {
      for (int64_t k = 0; k < seq_width; ++k) {
        size_t pad_data_idx = 0;
        size_t seq_data_idx = (seq_start + j) * seq_width + k;
        if (output_layout == kBatchLengthWidth) {
          pad_data_idx = (i * max_seq_len + j) * seq_width + k;
        } else {
          pad_data_idx = (j * seq_num + i) * seq_width + k;
        }
        if (seq_to_pad) {
          pad_data[pad_data_idx] = seq_data[seq_data_idx] * scale;
        } else {
          seq_data[seq_data_idx] = pad_data[pad_data_idx] * scale;
        }
      }
    }
  }
}

template <typename T>
class PaddingLoDTensorFunctor<platform::CPUDeviceContext, T> {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::LoDTensor& seq_tensor,
                  framework::Tensor* pad_tensor,
                  T pad_value = static_cast<T>(0), bool norm_by_times = false,
                  size_t lod_level = 0,
                  OutputLayout output_layout = kBatchLengthWidth) {
    CheckLoD(seq_tensor, lod_level);

    auto& lod = seq_tensor.lod();
    auto& seq_offset = framework::ToAbsOffset(lod)[lod_level];

    auto seq_tensor_dims = seq_tensor.dims();
    auto pad_tensor_dims = pad_tensor->dims();
    int64_t max_seq_len = MaximumSequenceLength(seq_offset);
    int64_t seq_num = seq_offset.size() - 1;
    int64_t seq_width = seq_tensor.numel() / seq_tensor_dims[0];

    CheckDims(seq_tensor_dims, seq_offset.back(), pad_tensor_dims, max_seq_len,
              seq_num, seq_width, output_layout);

    T* pad_data = pad_tensor->data<T>();

    memset(pad_data, pad_value, max_seq_len * seq_num * seq_width * sizeof(T));

    CopyDataCPU<T>(const_cast<framework::LoDTensor*>(&seq_tensor), pad_tensor,
                   seq_offset, max_seq_len, seq_width, true /* seq_to_pad */,
                   norm_by_times, output_layout);
  }
};

template <typename T>
class UnpaddingLoDTensorFunctor<platform::CPUDeviceContext, T> {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  framework::LoDTensor* seq_tensor,
                  const framework::Tensor& pad_tensor,
                  bool norm_by_times = false, size_t lod_level = 0,
                  OutputLayout output_layout = kBatchLengthWidth) {
    CheckLoD(*seq_tensor, lod_level);

    auto& lod = seq_tensor->lod();
    auto& seq_offset = framework::ToAbsOffset(lod)[lod_level];

    auto& seq_tensor_dims = seq_tensor->dims();
    auto& pad_tensor_dims = pad_tensor.dims();
    int64_t max_seq_len = MaximumSequenceLength(seq_offset);
    int64_t seq_num = seq_offset.size() - 1;
    int64_t seq_width = seq_tensor->numel() / seq_tensor_dims[0];

    CheckDims(seq_tensor_dims, seq_offset.back(), pad_tensor_dims, max_seq_len,
              seq_num, seq_width, output_layout);

    T* seq_data = seq_tensor->data<T>();
    memset(seq_data, static_cast<T>(0), seq_tensor->numel() * sizeof(T));

    CopyDataCPU<T>(seq_tensor, const_cast<framework::Tensor*>(&pad_tensor),
                   seq_offset, max_seq_len, seq_width, false /* seq_to_pad */,
                   norm_by_times, output_layout);
  }
};

template class PaddingLoDTensorFunctor<platform::CPUDeviceContext, int>;
template class PaddingLoDTensorFunctor<platform::CPUDeviceContext, int64_t>;
template class PaddingLoDTensorFunctor<platform::CPUDeviceContext, float>;
template class PaddingLoDTensorFunctor<platform::CPUDeviceContext, double>;

template class UnpaddingLoDTensorFunctor<platform::CPUDeviceContext, int>;
template class UnpaddingLoDTensorFunctor<platform::CPUDeviceContext, int64_t>;
template class UnpaddingLoDTensorFunctor<platform::CPUDeviceContext, float>;
template class UnpaddingLoDTensorFunctor<platform::CPUDeviceContext, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
