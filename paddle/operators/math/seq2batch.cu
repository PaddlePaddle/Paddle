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

template <typename T, bool NormByTimes, bool Seq2Batch>
__global__ void Seq2BatchPaddingKernel(T* batch, T* sequence,
                                       const size_t* sequence_start_positions,
                                       const size_t sequence_width,
                                       const size_t max_sequence_length,
                                       const size_t num_sequences) {
  size_t batch_idx = blockIdx.y;
  size_t start_pos = sequence_start_positions[batch_idx];
  size_t sequence_length = sequence_start_positions[batch_idx + 1] - start_pos;

  size_t sequence_idx = blockIdx.x * blockDim.y + threadIdx.y;
  size_t batch_base_idx =
      (sequence_idx * num_sequences + batch_idx) * sequence_width;
  size_t sequence_base_idx = (start_pos + sequence_idx) * sequence_width;

  T scale = NormByTimes ? (1.0 / static_cast<T>(sequence_length)) : 1.0;

  if (sequence_idx < sequence_length) {
    if (Seq2Batch) {
      /* sequence -> batch */
      for (size_t i = threadIdx.x; i < sequence_width; i += blockDim.x) {
        batch[batch_base_idx + i] = scale * sequence[sequence_base_idx + i];
      }
    } else {
      /* batch -> sequence */
      for (size_t i = threadIdx.x; i < sequence_width; i += blockDim.x) {
        sequence[sequence_base_idx + i] = scale * batch[batch_base_idx + i];
      }
    }
  } else if (sequence_idx < max_sequence_length) {
    if (Seq2Batch) {
      /* sequence -> batch */
      for (size_t i = threadIdx.x; i < sequence_width; i += blockDim.x) {
        batch[batch_base_idx + i] = 0;
      }
    }
  }
}

template <typename T>
class Seq2BatchFunctor<true, platform::GPUPlace, T> {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::LoDTensor& seq, framework::Tensor& batch,
                  bool norm_by_times) {
    auto lod = seq.lod();
    PADDLE_ENFORCE_GT(lod.size(), 0UL,
                      "The lod of LoDTensor seq should not be null.");

    const size_t level = 0;
    const size_t num_sequences = lod[level].size() - 1;

    // Compute maximum sequence length
    size_t max_sequence_length = MaximumSequenceLength(lod, level);

    auto seq_dims = seq.dims();
    PADDLE_ENFORCE_EQ(seq_dims[0], lod[level].back(),
                      "The first dimension of LoDTensor seq should be "
                      "equal to the sum of all sequences's length.");

    const size_t sequence_width = seq.numel() / seq_dims[0];
    auto batch_dims =
        framework::make_ddim({static_cast<int64_t>(max_sequence_length),
                              static_cast<int64_t>(num_sequences),
                              static_cast<int64_t>(sequence_width)});

    if (!norm_by_times && num_sequences == 1) {
      batch.CopyFrom<T>(seq, context.GetPlace());
      batch.Resize(batch_dims);
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
    T* batch_data = batch.mutable_data<T>(batch_dims, context.GetPlace());
    if (norm_by_times) {
      Seq2BatchPaddingKernel<
          T, 1,
          1><<<grid, threads, 0,
               reinterpret_cast<const platform::CUDADeviceContext&>(context)
                   .stream()>>>(batch_data, const_cast<T*>(seq_data),
                                lod[level].data(), sequence_width,
                                max_sequence_length, num_sequences);
    } else {
      Seq2BatchPaddingKernel<
          T, 0,
          1><<<grid, threads, 0,
               reinterpret_cast<const platform::CUDADeviceContext&>(context)
                   .stream()>>>(batch_data, const_cast<T*>(seq_data),
                                lod[level].data(), sequence_width,
                                max_sequence_length, num_sequences);
    }
  }
};

template <typename T>
class Batch2SeqFunctor<true, platform::GPUPlace, T> {
 public:
  void operator()(const platform::DeviceContext& context,
                  framework::LoDTensor& seq, const framework::Tensor& batch,
                  bool norm_by_times) {
    auto lod = seq.lod();
    PADDLE_ENFORCE_GT(lod.size(), 0UL,
                      "The lod of LoDTensor seq should not be null.");

    const size_t level = 0;
    const size_t num_sequences = lod[level].size() - 1;

    // Compute maximum sequence length
    size_t max_sequence_length = MaximumSequenceLength(lod, level);

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
    auto seq_dims =
        framework::make_ddim({static_cast<int64_t>(lod[level].back()),
                              static_cast<int64_t>(sequence_width)});

    if (!norm_by_times && num_sequences == 1) {
      seq.CopyFrom<T>(batch, context.GetPlace());
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

    const T* batch_data = batch.data<T>();
    T* seq_data = seq.mutable_data<T>(seq_dims, context.GetPlace());
    if (norm_by_times) {
      Seq2BatchPaddingKernel<
          T, 1,
          0><<<grid, threads, 0,
               reinterpret_cast<const platform::CUDADeviceContext&>(context)
                   .stream()>>>(const_cast<T*>(batch_data), seq_data,
                                lod[level].data(), sequence_width,
                                max_sequence_length, num_sequences);
    } else {
      Seq2BatchPaddingKernel<
          T, 0,
          0><<<grid, threads, 0,
               reinterpret_cast<const platform::CUDADeviceContext&>(context)
                   .stream()>>>(const_cast<T*>(batch_data), seq_data,
                                lod[level].data(), sequence_width,
                                max_sequence_length, num_sequences);
    }
  }
};

template class Seq2BatchFunctor<true, platform::GPUPlace, float>;
template class Batch2SeqFunctor<true, platform::GPUPlace, float>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
