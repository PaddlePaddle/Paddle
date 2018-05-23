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

template <typename T, bool Padding>
__global__ void SequencePaddingKernel(
    T* pad_data, T* seq_data, const size_t* seq_offset, const size_t& seq_num,
    const size_t& max_seq_len, const size_t& seq_width, bool norm_by_times,
    const T& pad_value, const OutputLayout& output_layout) {
  size_t seq_idx = blockIdx.y;
  size_t seq_start = seq_offset[seq_idx];
  size_t seq_len = seq_offset[seq_idx + 1] - seq_start;

  size_t seq_step_idx = blockIdx.x * blockDim.y + threadIdx.y;

  size_t seq_data_offset = (seq_start + seq_step_idx) * seq_width;

  size_t pad_data_offset = 0;

  if (output_layout == kLengthBatchWidth) {
    pad_data_offset = (seq_step_idx * seq_num + seq_idx) * seq_width;
  } else {
    pad_data_offset = (seq_idx * max_seq_len + seq_step_idx) * seq_width;
  }

  if (seq_step_idx < seq_len) {
    T scale = norm_by_times ? (1.0f / static_cast<T>(seq_len)) : 1.0f;
    if (Padding) {
      /* seq -> pad */
      for (size_t i = threadIdx.x; i < seq_width; i += blockDim.x) {
        pad_data[pad_data_offset + i] = scale * seq_data[seq_data_offset + i];
      }
    } else {
      /* pad -> seq */
      for (size_t i = threadIdx.x; i < seq_width; i += blockDim.x) {
        seq_data[seq_data_offset + i] = scale * pad_data[pad_data_offset + i];
      }
    }
  } else if (seq_step_idx < max_seq_len) {
    if (Padding) {
      /* seq -> pad */
      for (size_t i = threadIdx.x; i < seq_width; i += blockDim.x) {
        pad_data[pad_data_offset + i] = pad_value;
      }
    }
  }
}

template <typename T>
class PaddingLoDTensorFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
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

    if (!norm_by_times && seq_num == 1UL) {
      TensorCopy(seq_tensor, context.GetPlace(), context, pad_tensor);
      pad_tensor->Resize(pad_tensor_dims);
      return;
    }

    const int64_t kBlockSize = 512;

    /* At least use 32 threads to copy sequence_width elements,
     * and at least 8 elements for each thread.
     */
    size_t block_dim_x =
        std::min(((((seq_width + 7) >> 3) + 31) >> 5) << 5, kBlockSize);
    size_t block_dim_y = kBlockSize / block_dim_x;
    dim3 threads(block_dim_x, block_dim_y);

    size_t grid_dim_x = (max_seq_len + block_dim_y - 1) / block_dim_y;
    size_t grid_dim_y = seq_num;
    dim3 grid(grid_dim_x, grid_dim_y);

    const T* seq_data = seq_tensor.data<T>();
    T* pad_data = pad_tensor->data<T>();

    SequencePaddingKernel<T, 1><<<grid, threads, 0, context.stream()>>>(
        pad_data, const_cast<T*>(seq_data),
        seq_offset.CUDAData(context.GetPlace()), seq_num, max_seq_len,
        seq_width, norm_by_times, pad_value, output_layout);
  }
};

template <typename T>
class UnpaddingLoDTensorFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  framework::LoDTensor* seq_tensor,
                  const framework::Tensor& pad_tensor,
                  bool norm_by_times = false, size_t lod_level = 0,
                  OutputLayout output_layout = kBatchLengthWidth) {
    CheckLoD(*seq_tensor, lod_level);

    auto& lod = seq_tensor->lod();
    auto& seq_offset = framework::ToAbsOffset(lod)[lod_level];

    auto seq_tensor_dims = seq_tensor->dims();
    auto pad_tensor_dims = pad_tensor.dims();
    int64_t max_seq_len = MaximumSequenceLength(seq_offset);
    int64_t seq_num = seq_offset.size() - 1;
    int64_t seq_width = seq_tensor->numel() / seq_tensor_dims[0];

    CheckDims(seq_tensor_dims, seq_offset.back(), pad_tensor_dims, max_seq_len,
              seq_num, seq_width, output_layout);

    if (!norm_by_times && seq_num == 1UL) {
      TensorCopy(pad_tensor, context.GetPlace(), context, seq_tensor);
      seq_tensor->Resize(seq_tensor_dims);
      return;
    }

    const int64_t kBlockSize = 512;

    /* At least use 32 threads to copy sequence_width elements,
     * and at least 8 elements for each thread.
     */
    size_t block_dim_x =
        std::min(((((seq_width + 7) >> 3) + 31) >> 5) << 5, kBlockSize);
    size_t block_dim_y = kBlockSize / block_dim_x;
    dim3 threads(block_dim_x, block_dim_y);

    size_t grid_dim_x = (max_seq_len + block_dim_y - 1) / block_dim_y;
    size_t grid_dim_y = seq_num;
    dim3 grid(grid_dim_x, grid_dim_y);

    const T* pad_data = pad_tensor.data<T>();
    T* seq_data = seq_tensor->data<T>();

    SequencePaddingKernel<T, 0><<<grid, threads, 0, context.stream()>>>(
        const_cast<T*>(pad_data), seq_data,
        seq_offset.CUDAData(context.GetPlace()), seq_num, max_seq_len,
        seq_width, norm_by_times, static_cast<T>(0), output_layout);
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
