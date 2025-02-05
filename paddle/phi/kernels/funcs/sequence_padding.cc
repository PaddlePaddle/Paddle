/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/funcs/sequence_padding.h"

#include "paddle/phi/backends/cpu/cpu_context.h"

#ifdef PADDLE_WITH_XPU
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#endif

namespace phi::funcs {

template <typename T>
void CopyValidData(phi::DenseTensor* dst_tensor,
                   const phi::DenseTensor* src_tensor,
                   const phi::Vector<size_t>& seq_offsets,
                   int pad_seq_len,
                   int step_width,
                   bool norm_by_len,
                   CopyType type,
                   PadLayout layout) {
  int seq_num = static_cast<int>(seq_offsets.size() - 1);
  const T* src_data = src_tensor->data<T>();
  T* dst_data = dst_tensor->data<T>();

  int seq_cpy_gap = step_width;
  int pad_cpy_gap =
      layout == kBatchLengthWidth ? step_width : seq_num * step_width;
  for (int seq_idx = 0; seq_idx < seq_num; ++seq_idx) {
    int valid_seq_len =
        static_cast<int>(seq_offsets[seq_idx + 1] - seq_offsets[seq_idx]);
    PADDLE_ENFORCE_GE(
        pad_seq_len,
        valid_seq_len,
        common::errors::InvalidArgument(
            "The padded sequence length can not "
            "be less than its original length. Expected %ld >= %ld, but got "
            "%ld < %ld. Please check input value.",
            pad_seq_len,
            valid_seq_len,
            pad_seq_len,
            valid_seq_len));
    int seq_data_offset = static_cast<int>(seq_offsets[seq_idx] * step_width);
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
static void fast_mem_init(void* dest,
                          size_t dest_size,
                          const T* src,
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
class PaddingDenseTensorFunctor<phi::CPUContext, T> {
 public:
  void operator()(const phi::CPUContext& context UNUSED,
                  const phi::DenseTensor& seq_tensor,
                  phi::DenseTensor* pad_tensor,
                  const phi::DenseTensor& pad_value,
                  int pad_seq_len = -1,
                  int lod_level = 0,
                  bool norm_by_times = false,
                  const PadLayout layout = kBatchLengthWidth) {
    auto seq_lod = seq_tensor.lod();
    const auto seq_offsets = phi::ToAbsOffset(seq_lod)[lod_level];
    const auto& seq_tensor_dims = seq_tensor.dims();
    const auto& pad_tensor_dims = pad_tensor->dims();
    if (pad_seq_len == -1) {
      pad_seq_len = static_cast<int>(MaximumSequenceLength(seq_offsets));
    }
    int step_width = static_cast<int>(seq_tensor.numel() / seq_tensor_dims[0]);

    CheckDims(seq_tensor_dims,
              pad_tensor_dims,
              seq_offsets,
              pad_seq_len,
              step_width,
              layout);

    PADDLE_ENFORCE_EQ(
        pad_value.numel() == 1 || pad_value.numel() == step_width,
        true,
        common::errors::InvalidArgument(
            "The numel of 'pad_value' can only be 1 or be equal to the "
            "'step_width', but got %ld != 1 and %ld. Please check the input "
            "value.",
            pad_value.numel(),
            step_width));

    // fill padding value
    T* pad_data = pad_tensor->data<T>();
    const T* pad_value_data = pad_value.data<T>();
    if (pad_value.numel() == 1) {
      fast_mem_init<T>(
          pad_data, pad_tensor->numel(), pad_value_data, sizeof(T));
    } else {
      for (int i = 0; i < pad_tensor->numel(); i += step_width) {
        memcpy(pad_data + i, pad_value_data, step_width * sizeof(T));
      }
    }

    CopyValidData<T>(pad_tensor,
                     &seq_tensor,
                     seq_offsets,
                     pad_seq_len,
                     step_width,
                     norm_by_times,
                     kSeqToPad,
                     layout);
  }
};

template <typename T>
class UnpaddingDenseTensorFunctor<phi::CPUContext, T> {
 public:
  void operator()(const phi::CPUContext& context UNUSED,
                  const phi::DenseTensor& pad_tensor,
                  phi::DenseTensor* seq_tensor,
                  int pad_seq_len = -1,
                  int lod_level = 0,
                  bool norm_by_times = false,
                  const PadLayout layout = kBatchLengthWidth) {
    auto seq_offsets = phi::ToAbsOffset(seq_tensor->lod())[lod_level];
    const auto& seq_tensor_dims = seq_tensor->dims();
    const auto& pad_tensor_dims = pad_tensor.dims();
    if (pad_seq_len == -1) {
      pad_seq_len = static_cast<int>(MaximumSequenceLength(seq_offsets));
    }
    int step_width = static_cast<int>(seq_tensor->numel() / seq_tensor_dims[0]);

    CheckDims(seq_tensor_dims,
              pad_tensor_dims,
              seq_offsets,
              pad_seq_len,
              step_width,
              layout);

    CopyValidData<T>(seq_tensor,
                     &pad_tensor,
                     seq_offsets,
                     pad_seq_len,
                     step_width,
                     norm_by_times,
                     kPadToSeq,
                     layout);
  }
};

#ifdef PADDLE_WITH_XPU
template <typename T>
class UnpaddingDenseTensorFunctor<phi::XPUContext, T> {
 public:
  void operator()(const phi::XPUContext& context,
                  const phi::DenseTensor& pad_tensor,
                  phi::DenseTensor* seq_tensor,
                  int pad_seq_len = -1,
                  int lod_level = 0,
                  bool norm_by_times = false,
                  const PadLayout layout = kBatchLengthWidth) {
    auto seq_offsets = phi::ToAbsOffset(seq_tensor->lod())[lod_level];
    const auto& seq_tensor_dims = seq_tensor->dims();
    const auto& pad_tensor_dims = pad_tensor.dims();
    if (pad_seq_len == -1) {
      pad_seq_len = MaximumSequenceLength(seq_offsets);
    }
    int step_width = seq_tensor->numel() / seq_tensor_dims[0];

    CheckDims(seq_tensor_dims,
              pad_tensor_dims,
              seq_offsets,
              pad_seq_len,
              step_width,
              layout);

    const T* pad_data = pad_tensor.data<T>();  // padding tensor x
    T* seq_data = seq_tensor->data<T>();       // unpadding tensor y

    xpu::VectorParam<int64_t> seq_offsets_param{
        reinterpret_cast<int64_t*>(seq_offsets.data()),
        static_cast<int>(seq_offsets.size()),
        nullptr};
    int r = xpu::sequence_unpad<T, int64_t>(context.x_context(),
                                            pad_data,
                                            seq_data,
                                            seq_offsets_param,
                                            pad_seq_len /*max_seqlen*/,
                                            step_width /*dim*/);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "sequence_unpad");
  }
};
#endif

template class PaddingDenseTensorFunctor<phi::CPUContext, int>;
template class PaddingDenseTensorFunctor<phi::CPUContext, int64_t>;
template class PaddingDenseTensorFunctor<phi::CPUContext, float>;
template class PaddingDenseTensorFunctor<phi::CPUContext, double>;

template class UnpaddingDenseTensorFunctor<phi::CPUContext, int>;
template class UnpaddingDenseTensorFunctor<phi::CPUContext, int64_t>;
template class UnpaddingDenseTensorFunctor<phi::CPUContext, float>;
template class UnpaddingDenseTensorFunctor<phi::CPUContext, double>;

#ifdef PADDLE_WITH_XPU
template class UnpaddingDenseTensorFunctor<phi::XPUContext, float>;
#endif

}  // namespace phi::funcs
