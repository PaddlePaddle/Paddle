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

namespace pten {
class DenseTensor;
}  // namespace pten

namespace paddle {
namespace framework {}  // namespace framework
namespace platform {
class CPUDeviceContext;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {
namespace math {

template <typename T>
void CopyValidData(framework::Tensor* dst_tensor,
                   const framework::Tensor* src_tensor,
                   const framework::Vector<size_t>& seq_offsets,
                   int pad_seq_len, int step_width, bool norm_by_len,
                   CopyType type, PadLayout layout) {
  int seq_num = seq_offsets.size() - 1;
  const T* src_data = src_tensor->data<T>();
  T* dst_data = dst_tensor->data<T>();

  int seq_cpy_gap = step_width;
  int pad_cpy_gap =
      layout == kBatchLengthWidth ? step_width : seq_num * step_width;
  for (int seq_idx = 0; seq_idx < seq_num; ++seq_idx) {
    int valid_seq_len = seq_offsets[seq_idx + 1] - seq_offsets[seq_idx];
    PADDLE_ENFORCE_GE(
        pad_seq_len, valid_seq_len,
        platform::errors::InvalidArgument(
            "The padded sequence length can not "
            "be less than its original length. Expected %ld >= %ld, but got "
            "%ld < %ld. Please check input value.",
            pad_seq_len, valid_seq_len, pad_seq_len, valid_seq_len));
    int seq_data_offset = seq_offsets[seq_idx] * step_width;
    int pad_data_offset = layout == kBatchLengthWidth
                              ? seq_idx * pad_seq_len * step_width
                              : seq_idx * step_width;
    float scale = 1.0f / static_cast<float>(valid_seq_len);

    for (int step_idx = 0; step_idx < valid_seq_len; ++step_idx) {
      const T* src =
          src_data + (type == kSeqToPad ? seq_data_offset : pad_data_offset);
      T* dst =
          dst_data + (type == kSeqToPad ? pad_data_offset : seq_data_offset);
      memcpy(dst, src, step_width * sizeof(T));
      if (norm_by_len) {
        for (int i = 0; i < step_width; ++i) {
          *(dst + i) *= scale;
        }
      }
      seq_data_offset += seq_cpy_gap;
      pad_data_offset += pad_cpy_gap;
    }
  }
}

template <typename T>
static void fast_mem_init(void* dest, size_t dest_size, const T* src,
                          size_t num_bytes) {
  if (dest == nullptr || dest_size == 0 || src == nullptr) return;

  memcpy(dest, src, num_bytes);

  dest_size *= num_bytes;
  while (dest_size > num_bytes) {
    size_t remaining = dest_size - num_bytes;
    size_t count = (remaining > num_bytes) ? num_bytes : remaining;
    memcpy((unsigned char*)dest + num_bytes, dest, count);
    num_bytes += count;
  }
}

template <typename T>
class PaddingLoDTensorFunctor<platform::CPUDeviceContext, T> {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::LoDTensor& seq_tensor,
                  framework::LoDTensor* pad_tensor,
                  const framework::LoDTensor& pad_value, int pad_seq_len = -1,
                  int lod_level = 0, bool norm_by_times = false,
                  const PadLayout layout = kBatchLengthWidth) {
    auto seq_lod = seq_tensor.lod();
    const auto seq_offsets = framework::ToAbsOffset(seq_lod)[lod_level];
    const auto& seq_tensor_dims = seq_tensor.dims();
    const auto& pad_tensor_dims = pad_tensor->dims();
    if (pad_seq_len == -1) {
      pad_seq_len = MaximumSequenceLength(seq_offsets);
    }
    int step_width = seq_tensor.numel() / seq_tensor_dims[0];

    CheckDims(seq_tensor_dims, pad_tensor_dims, seq_offsets, pad_seq_len,
              step_width, layout);

    PADDLE_ENFORCE_EQ(
        pad_value.numel() == 1 || pad_value.numel() == step_width, true,
        platform::errors::InvalidArgument(
            "The numel of 'pad_value' can only be 1 or be equal to the "
            "'step_width', but got %ld != 1 and %ld. Please check the input "
            "value.",
            pad_value.numel(), step_width));

    // fill padding value
    T* pad_data = pad_tensor->data<T>();
    const T* pad_value_data = pad_value.data<T>();
    if (pad_value.numel() == 1) {
      fast_mem_init<T>(pad_data, pad_tensor->numel(), pad_value_data,
                       sizeof(T));
    } else {
      for (int i = 0; i < pad_tensor->numel(); i += step_width) {
        memcpy(pad_data + i, pad_value_data, step_width * sizeof(T));
      }
    }

    CopyValidData<T>(pad_tensor, &seq_tensor, seq_offsets, pad_seq_len,
                     step_width, norm_by_times, kSeqToPad, layout);
  }
};

template <typename T>
class UnpaddingLoDTensorFunctor<platform::CPUDeviceContext, T> {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::LoDTensor& pad_tensor,
                  framework::LoDTensor* seq_tensor, int pad_seq_len = -1,
                  int lod_level = 0, bool norm_by_times = false,
                  const PadLayout layout = kBatchLengthWidth) {
    auto seq_offsets = framework::ToAbsOffset(seq_tensor->lod())[lod_level];
    const auto& seq_tensor_dims = seq_tensor->dims();
    const auto& pad_tensor_dims = pad_tensor.dims();
    if (pad_seq_len == -1) {
      pad_seq_len = MaximumSequenceLength(seq_offsets);
    }
    int step_width = seq_tensor->numel() / seq_tensor_dims[0];

    CheckDims(seq_tensor_dims, pad_tensor_dims, seq_offsets, pad_seq_len,
              step_width, layout);

    CopyValidData<T>(seq_tensor, &pad_tensor, seq_offsets, pad_seq_len,
                     step_width, norm_by_times, kPadToSeq, layout);
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
