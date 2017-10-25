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

#include "paddle/operators/math/sequence_padding.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T, bool NormByTimes, bool Padding>
__global__ void SequencePaddingKernel(T* padding, T* sequence,
                                      const size_t* sequence_start_positions,
                                      const size_t sequence_width,
                                      const size_t max_sequence_length,
                                      const size_t num_sequences) {
  size_t padding_idx = blockIdx.y;
  size_t start_pos = sequence_start_positions[padding_idx];
  size_t sequence_length =
      sequence_start_positions[padding_idx + 1] - start_pos;

  size_t sequence_idx = blockIdx.x * blockDim.y + threadIdx.y;
  size_t padding_base_idx =
      (sequence_idx * num_sequences + padding_idx) * sequence_width;
  size_t sequence_base_idx = (start_pos + sequence_idx) * sequence_width;

  if (sequence_idx < sequence_length) {
    T scale = NormByTimes ? (1.0f / static_cast<T>(sequence_length)) : 1.0f;
    if (Padding) {
      /* sequence -> padding */
      for (size_t i = threadIdx.x; i < sequence_width; i += blockDim.x) {
        padding[padding_base_idx + i] = scale * sequence[sequence_base_idx + i];
      }
    } else {
      /* padding -> sequence */
      for (size_t i = threadIdx.x; i < sequence_width; i += blockDim.x) {
        sequence[sequence_base_idx + i] = scale * padding[padding_base_idx + i];
      }
    }
  } else if (sequence_idx < max_sequence_length) {
    if (Padding) {
      /* sequence -> padding */
      for (size_t i = threadIdx.x; i < sequence_width; i += blockDim.x) {
        padding[padding_base_idx + i] = 0;
      }
    }
  }
}

template <typename T>
class PaddingSequenceFunctor<platform::GPUPlace, T> {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::LoDTensor& seq, framework::Tensor& padding,
                  bool norm_by_times) {
    auto lod = seq.lod();
    PADDLE_ENFORCE_GT(lod.size(), 0UL,
                      "The lod of LoDTensor seq should not be null.");

    const size_t level = 0;
    framework::LoD abs_offset_lod = framework::ToAbsOffset(lod);
    const size_t num_sequences = abs_offset_lod[level].size() - 1;

    // Compute maximum sequence length
    size_t max_sequence_length = MaximumSequenceLength(lod, level);

    auto seq_dims = seq.dims();
    PADDLE_ENFORCE_EQ(seq_dims[0], abs_offset_lod[level].back(),
                      "The first dimension of LoDTensor seq should be "
                      "equal to the sum of all sequences's length.");

    const size_t sequence_width = seq.numel() / seq_dims[0];
    auto padding_dims =
        framework::make_ddim({static_cast<int64_t>(max_sequence_length),
                              static_cast<int64_t>(num_sequences),
                              static_cast<int64_t>(sequence_width)});

    if (!norm_by_times && num_sequences == 1UL) {
      padding.CopyFrom(seq, context.GetPlace(), context);
      padding.Resize(padding_dims);
      return;
    }

    const size_t kBlockSize = 512;

    /* At least use 32 threads to copy sequence_width elements,
     * and at least 8 elements for each thread.
     */
    size_t block_dim_x =
        std::min(((((sequence_width + 7) >> 3) + 31) >> 5) << 5, kBlockSize);
    size_t block_dim_y = kBlockSize / block_dim_x;
    dim3 threads(block_dim_x, block_dim_y);

    size_t grid_dim_x = (max_sequence_length + block_dim_y - 1) / block_dim_y;
    size_t grid_dim_y = num_sequences;
    dim3 grid(grid_dim_x, grid_dim_y);

    const T* seq_data = seq.data<T>();
    T* padding_data = padding.mutable_data<T>(padding_dims, context.GetPlace());
    if (norm_by_times) {
      SequencePaddingKernel<
          T, 1,
          1><<<grid, threads, 0,
               reinterpret_cast<const platform::CUDADeviceContext&>(context)
                   .stream()>>>(padding_data, const_cast<T*>(seq_data),
                                abs_offset_lod[level].data(), sequence_width,
                                max_sequence_length, num_sequences);
    } else {
      SequencePaddingKernel<
          T, 0,
          1><<<grid, threads, 0,
               reinterpret_cast<const platform::CUDADeviceContext&>(context)
                   .stream()>>>(padding_data, const_cast<T*>(seq_data),
                                abs_offset_lod[level].data(), sequence_width,
                                max_sequence_length, num_sequences);
    }
  }
};

template <typename T>
class UnpaddingSequenceFunctor<platform::GPUPlace, T> {
 public:
  void operator()(const platform::DeviceContext& context,
                  framework::LoDTensor& seq, const framework::Tensor& padding,
                  bool norm_by_times) {
    auto lod = seq.lod();
    PADDLE_ENFORCE_GT(lod.size(), 0UL,
                      "The lod of LoDTensor seq should not be null.");

    const size_t level = 0;
    framework::LoD abs_offset_lod = framework::ToAbsOffset(lod);
    const size_t num_sequences = abs_offset_lod[level].size() - 1;

    // Compute maximum sequence length
    size_t max_sequence_length = MaximumSequenceLength(lod, level);

    auto padding_dims = padding.dims();
    PADDLE_ENFORCE_EQ(padding_dims.size(), 3UL,
                      "The input padding should be a 3-D Tensor.");
    PADDLE_ENFORCE_EQ(padding_dims[0], max_sequence_length,
                      "The first dimension of Tensor padding should be "
                      "equal to the maximum sequence's length.");
    PADDLE_ENFORCE_EQ(padding_dims[1], num_sequences,
                      "The second dimension of Tensor padding should be "
                      "equal to the number of sequences.");

    const size_t sequence_width = padding_dims[2];
    auto seq_dims = framework::make_ddim(
        {static_cast<int64_t>(abs_offset_lod[level].back()),
         static_cast<int64_t>(sequence_width)});

    if (!norm_by_times && num_sequences == 1UL) {
      seq.CopyFrom(padding, context.GetPlace(), context);
      seq.Resize(seq_dims);
      return;
    }

    const size_t kBlockSize = 512;

    /* At least use 32 threads to copy sequence_width elements,
     * and at least 8 elements for each thread.
     */
    size_t block_dim_x =
        std::min(((((sequence_width + 7) >> 3) + 31) >> 5) << 5, kBlockSize);
    size_t block_dim_y = kBlockSize / block_dim_x;
    dim3 threads(block_dim_x, block_dim_y);

    size_t grid_dim_x = (max_sequence_length + block_dim_y - 1) / block_dim_y;
    size_t grid_dim_y = num_sequences;
    dim3 grid(grid_dim_x, grid_dim_y);

    const T* padding_data = padding.data<T>();
    T* seq_data = seq.mutable_data<T>(seq_dims, context.GetPlace());
    if (norm_by_times) {
      SequencePaddingKernel<
          T, 1,
          0><<<grid, threads, 0,
               reinterpret_cast<const platform::CUDADeviceContext&>(context)
                   .stream()>>>(const_cast<T*>(padding_data), seq_data,
                                abs_offset_lod[level].data(), sequence_width,
                                max_sequence_length, num_sequences);
    } else {
      SequencePaddingKernel<
          T, 0,
          0><<<grid, threads, 0,
               reinterpret_cast<const platform::CUDADeviceContext&>(context)
                   .stream()>>>(const_cast<T*>(padding_data), seq_data,
                                abs_offset_lod[level].data(), sequence_width,
                                max_sequence_length, num_sequences);
    }
  }
};

template class PaddingSequenceFunctor<platform::GPUPlace, float>;
template class UnpaddingSequenceFunctor<platform::GPUPlace, float>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
