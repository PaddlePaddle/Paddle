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
    T* padding_data, T* seq_data, const size_t* abs_offset,
    const size_t& seq_num, const size_t& max_seq_len, const size_t& seq_width,
    const PaddingLayout& padding_layout, bool norm_by_times = false,
    const T& padding_value = 0) {
  size_t padding_idx = blockIdx.y;
  size_t seq_start = abs_offset[padding_idx];
  size_t seq_len = abs_offset[padding_idx + 1] - seq_start;

  size_t seq_idx = blockIdx.x * blockDim.y + threadIdx.y;

  size_t seq_offset = (seq_start + seq_idx) * seq_width;

  size_t padding_offset = 0;

  if (padding_layout == LENGTH_BATCH_WIDTH) {
    padding_offset = (seq_idx * seq_num + padding_idx) * seq_width;
  } else {
    padding_offset = (padding_idx * max_seq_len + seq_idx) * seq_width;
  }

  if (seq_idx < seq_len) {
    T scale = norm_by_times ? (1.0f / static_cast<T>(seq_len)) : 1.0f;
    if (Padding) {
      /* sequence -> padding */
      for (size_t i = threadIdx.x; i < seq_width; i += blockDim.x) {
        padding_data[padding_offset + i] = scale * seq_data[seq_offset + i];
      }
    } else {
      /* padding -> sequence */
      for (size_t i = threadIdx.x; i < seq_width; i += blockDim.x) {
        seq_data[seq_offset + i] = scale * padding_data[padding_offset + i];
      }
    }
  } else if (seq_idx < max_seq_len) {
    if (Padding) {
      /* sequence -> padding */
      for (size_t i = threadIdx.x; i < seq_width; i += blockDim.x) {
        padding_data[padding_offset + i] = padding_value;
      }
    }
  }
}

template <typename T, PaddingLayout padding_layout>
class PaddingLoDTensorFunctor<platform::CUDADeviceContext, T, padding_layout> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
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
    const int64_t seq_num = abs_offset.size() - 1;
    const int64_t seq_width = seq_tensor.numel() / seq_dims[0];

    ValidateShape(seq_dims, abs_offset.back(), padding_dims, max_seq_len,
                  seq_num, seq_width, padding_layout);

    if (!norm_by_times && seq_num == 1UL) {
      TensorCopy(seq_tensor, context.GetPlace(), context, padding_tensor);
      padding_tensor->Resize(padding_dims);
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
    T* padding_data = padding_tensor->data<T>();

    SequencePaddingKernel<T, 1><<<grid, threads, 0, context.stream()>>>(
        padding_data, const_cast<T*>(seq_data),
        abs_offset.CUDAData(context.GetPlace()), seq_num, max_seq_len,
        seq_width, padding_layout, norm_by_times, padding_value);
  }
};

template <typename T, PaddingLayout padding_layout>
class UnpaddingLoDTensorFunctor<platform::CUDADeviceContext, T,
                                padding_layout> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  framework::LoDTensor* seq_tensor,
                  const framework::Tensor& padding_tensor,
                  bool norm_by_times = false, size_t lod_level = 0) {
    ValidateLoD(*seq_tensor, lod_level);

    auto& lod = seq_tensor->lod();
    auto& abs_offset = framework::ToAbsOffset(lod)[lod_level];

    auto seq_dims = seq_tensor->dims();
    auto padding_dims = padding_tensor.dims();
    int64_t max_seq_len = MaximumSequenceLength(lod, lod_level);
    int64_t seq_num = abs_offset.size() - 1;
    int64_t seq_width = seq_tensor->numel() / seq_dims[0];

    if (!norm_by_times && seq_num == 1UL) {
      TensorCopy(padding_tensor, context.GetPlace(), context, seq_tensor);
      seq_tensor->Resize(seq_dims);
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

    const T* padding_data = padding_tensor.data<T>();
    T* seq_data = seq_tensor->data<T>();

    SequencePaddingKernel<T, 1><<<grid, threads, 0, context.stream()>>>(
        const_cast<T*>(padding_data), seq_data,
        abs_offset.CUDAData(context.GetPlace()), seq_num, max_seq_len,
        seq_width, padding_layout, norm_by_times);
  }
};

template class PaddingLoDTensorFunctor<platform::CUDADeviceContext, float,
                                       LENGTH_BATCH_WIDTH>;
template class UnpaddingLoDTensorFunctor<platform::CUDADeviceContext, float,
                                         LENGTH_BATCH_WIDTH>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
