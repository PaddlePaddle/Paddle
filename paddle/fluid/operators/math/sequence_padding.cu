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

#include <algorithm>
#include "paddle/fluid/operators/math/sequence_padding.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T, CopyType Type>
__global__ void SequencePaddingKernel(
    T* dst, const T* src, const T* pad_value, bool is_constant_pad,
    const size_t* seq_offsets, const size_t seq_num, const size_t pad_seq_len,
    const size_t step_width, bool norm_by_len, bool norm_by_batchsize,
    bool norm_by_total_logits_len, int total_logits_len,
    const PadLayout layout) {
  size_t seq_idx = blockIdx.y;
  size_t seq_len = seq_offsets[seq_idx + 1] - seq_offsets[seq_idx];

  size_t step_idx = blockIdx.x * blockDim.y + threadIdx.y;
  size_t seq_data_offset = (seq_offsets[seq_idx] + step_idx) * step_width;
  size_t pad_data_offset = layout == kBatchLengthWidth
                               ? (seq_idx * pad_seq_len + step_idx) * step_width
                               : (step_idx * seq_num + seq_idx) * step_width;

  T* dst_data = dst + (Type == kSeqToPad ? pad_data_offset : seq_data_offset);
  const T* src_data =
      src + (Type == kSeqToPad ? seq_data_offset : pad_data_offset);

  if (step_idx < seq_len) {
    float scale = 1.0f;
    if (norm_by_total_logits_len) {
      scale = 1.0f / static_cast<float>(total_logits_len);
    } else if (norm_by_batchsize) {
      scale = 1.0f / static_cast<float>(seq_num);
    } else if (norm_by_len) {
      scale = norm_by_len ? (1.0f / static_cast<float>(seq_len)) : 1.0f;
    }

    for (size_t i = threadIdx.x; i < step_width; i += blockDim.x) {
      dst_data[i] = scale * src_data[i];
    }
  } else if (step_idx < pad_seq_len && Type == kSeqToPad) {
    for (size_t i = threadIdx.x; i < step_width; i += blockDim.x) {
      dst_data[i] = is_constant_pad ? pad_value[0] : pad_value[i];
    }
  }
}

template <typename T>
class PaddingLoDTensorFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::LoDTensor& seq_tensor,
                  framework::LoDTensor* pad_tensor,
                  const framework::LoDTensor& pad_value, int pad_seq_len = -1,
                  int lod_level = 0, bool norm_by_times = false,
                  bool norm_by_batchsize = false,
                  bool norm_by_total_logits_len = false,
                  const PadLayout layout = kBatchLengthWidth) {
    auto seq_lod = seq_tensor.lod();
    const auto seq_offsets = framework::ToAbsOffset(seq_lod)[lod_level];
    const auto& seq_tensor_dims = seq_tensor.dims();
    const auto& pad_tensor_dims = pad_tensor->dims();
    int max_seq_len = MaximumSequenceLength(seq_offsets);
    if (pad_seq_len == -1) {
      pad_seq_len = max_seq_len;
    }
    PADDLE_ENFORCE_GE(
        pad_seq_len, max_seq_len,
        platform::errors::InvalidArgument(
            "The pad_seq_len must be equal to or greater than the "
            "original max sequence length. Expected %ld >= %ld, but got %ld < "
            "%ld. Please check the input value.",
            pad_seq_len, max_seq_len, pad_seq_len, max_seq_len));
    int step_width = seq_tensor.numel() / seq_tensor_dims[0];
    int seq_num = seq_offsets.size() - 1;

    CheckDims(seq_tensor_dims, pad_tensor_dims, seq_offsets, pad_seq_len,
              step_width, layout);
    PADDLE_ENFORCE_EQ(
        pad_value.numel() == 1 || pad_value.numel() == step_width, true,
        platform::errors::InvalidArgument(
            "The numel of 'pad_value' can only be 1 or be equal to "
            "the 'step_width', but got %ld != 1 and %ld. Please check the "
            "input value.",
            pad_value.numel(), step_width));

    const int kBlockSize = 512;

    /* At least use 32 threads to copy sequence_width elements,
     * and at least 8 elements for each thread.
     */
    size_t block_dim_x =
        std::min(((((step_width + 7) >> 3) + 31) >> 5) << 5, kBlockSize);
    size_t block_dim_y = kBlockSize / block_dim_x;
    dim3 threads(block_dim_x, block_dim_y);

    size_t grid_dim_x = (pad_seq_len + block_dim_y - 1) / block_dim_y;
    size_t grid_dim_y = seq_num;
    dim3 grid(grid_dim_x, grid_dim_y);

    const T* seq_data = seq_tensor.data<T>();
    T* pad_data = pad_tensor->data<T>();
    const T* pad_value_data = pad_value.data<T>();

    SequencePaddingKernel<T, kSeqToPad><<<grid, threads, 0, context.stream()>>>(
        pad_data, seq_data, pad_value_data, pad_value.numel() == 1,
        seq_offsets.CUDAData(context.GetPlace()), seq_num, pad_seq_len,
        step_width, norm_by_times, false, false, 0, layout);
  }
};

template <typename T>
class UnpaddingLoDTensorFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::LoDTensor& pad_tensor,
                  framework::LoDTensor* seq_tensor, int pad_seq_len = -1,
                  int lod_level = 0, bool norm_by_times = false,
                  bool norm_by_batchsize = false,
                  bool norm_by_total_logits_len = false,
                  const PadLayout layout = kBatchLengthWidth) {
    auto seq_offsets = framework::ToAbsOffset(seq_tensor->lod())[lod_level];
    const auto& seq_tensor_dims = seq_tensor->dims();
    const auto& pad_tensor_dims = pad_tensor.dims();
    int max_seq_len = MaximumSequenceLength(seq_offsets);
    if (pad_seq_len == -1) {
      pad_seq_len = max_seq_len;
    }
    int total_logits_len = TotalSequenceLength(seq_offsets);
    int step_width = seq_tensor->numel() / seq_tensor_dims[0];
    int seq_num = seq_offsets.size() - 1;

    CheckDims(seq_tensor_dims, pad_tensor_dims, seq_offsets, pad_seq_len,
              step_width, layout);
    /*
    if (!norm_by_times && seq_num == 1UL && pad_seq_len == max_seq_len) {
      TensorCopy(pad_tensor, context.GetPlace(), context, seq_tensor);
      seq_tensor->Resize(seq_tensor_dims);
      return;
    }
    */

    const int kBlockSize = 512;

    /* At least use 32 threads to copy sequence_width elements,
     * and at least 8 elements for each thread.
     */
    size_t block_dim_x =
        std::min(((((step_width + 7) >> 3) + 31) >> 5) << 5, kBlockSize);
    size_t block_dim_y = kBlockSize / block_dim_x;
    dim3 threads(block_dim_x, block_dim_y);

    size_t grid_dim_x = (pad_seq_len + block_dim_y - 1) / block_dim_y;
    size_t grid_dim_y = seq_num;
    dim3 grid(grid_dim_x, grid_dim_y);

    const T* pad_data = pad_tensor.data<T>();
    T* seq_data = seq_tensor->data<T>();

    SequencePaddingKernel<T, kPadToSeq><<<grid, threads, 0, context.stream()>>>(
        seq_data, pad_data, nullptr, false,
        seq_offsets.CUDAData(context.GetPlace()), seq_num, pad_seq_len,
        step_width, norm_by_times, norm_by_batchsize, norm_by_total_logits_len,
        total_logits_len, layout);
  }
};

template class PaddingLoDTensorFunctor<platform::CUDADeviceContext, int>;
template class PaddingLoDTensorFunctor<platform::CUDADeviceContext, int64_t>;
template class PaddingLoDTensorFunctor<platform::CUDADeviceContext, float>;
template class PaddingLoDTensorFunctor<platform::CUDADeviceContext, double>;

template class UnpaddingLoDTensorFunctor<platform::CUDADeviceContext, int>;
template class UnpaddingLoDTensorFunctor<platform::CUDADeviceContext, int64_t>;
template class UnpaddingLoDTensorFunctor<platform::CUDADeviceContext, float>;
template class UnpaddingLoDTensorFunctor<platform::CUDADeviceContext, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
