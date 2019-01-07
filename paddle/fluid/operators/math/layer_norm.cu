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

#include <cub/cub.cuh>
#include "paddle/fluid/operators/math/layer_norm.h"

namespace paddle {
namespace operators {
namespace math {

inline static int GetDesiredBlockDim(int block_dim) {
  const int kMaxBlockDim = 512;
  return block_dim >= kMaxBlockDim
             ? kMaxBlockDim
             : (1 << (static_cast<int>(std::log2f(block_dim))));
}

#define FIXED_BLOCK_DIM_CASE_BASE(log2_block_dim, ...)  \
  case (1 << (log2_block_dim)): {                       \
    constexpr auto kBlockDim = (1 << (log2_block_dim)); \
    __VA_ARGS__;                                        \
  } break

#define FIXED_BLOCK_DIM_CASE(...)              \
  FIXED_BLOCK_DIM_CASE_BASE(9, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_CASE_BASE(8, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_CASE_BASE(7, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_CASE_BASE(6, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_CASE_BASE(5, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_CASE_BASE(4, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_CASE_BASE(3, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_CASE_BASE(2, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_CASE_BASE(1, ##__VA_ARGS__)

static __device__ __forceinline__ float real_sqrt(float x) { return sqrtf(x); }
static __device__ __forceinline__ double real_sqrt(double x) { return sqrt(x); }

template <typename T>
struct PairForLayerNorm {
  __device__ __forceinline__ PairForLayerNorm() {}
  __device__ __forceinline__ PairForLayerNorm(const T &first, const T &second)
      : first_(first), second_(second) {}

  T first_;
  T second_;
};

template <typename T>
struct PairForLayerNormAddFunctor {
  __device__ __forceinline__ PairForLayerNorm<T> operator()(
      const PairForLayerNorm<T> &p1, const PairForLayerNorm<T> &p2) {
    return PairForLayerNorm<T>(p1.first_ + p2.first_, p1.second_ + p2.second_);
  }
};

template <typename T, int BlockDim>
__global__ void LayerNormForward(const T *x, const T *scale, const T *bias,
                                 T *y, T *mean, T *var, float epsilon,
                                 int feature_size, int batch_size) {
  using BlockReduce = cub::BlockReduce<PairForLayerNorm<double>, BlockDim>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int beg_idx = blockIdx.x * feature_size + threadIdx.x;
  int end_idx = (blockIdx.x + 1) * feature_size;

  // Step 1: Reduce to calculate mean and var
  double mean_val = 0;
  double var_val = 0;
  for (int i = beg_idx; i < end_idx; i += BlockDim) {
    T tmp = x[i];
    mean_val += tmp;
    var_val += (tmp * tmp);
  }
  auto pair = BlockReduce(temp_storage)
                  .Reduce(PairForLayerNorm<double>(mean_val, var_val),
                          PairForLayerNormAddFunctor<double>());

  /*
  double tmp_mean = 0;
  double tmp_val = 0;
  if (threadIdx.x == 0) {
    auto tmp = static_cast<T>(pair.first_ / feature_size);
    tmp_mean = static_cast<T>(tmp);
    tmp_val = static_cast<T>(pair.second_ / feature_size - tmp * tmp);
  }
  __syncthreads();

  mean_val = static_cast<T>(pair.first_ / feature_size);
  var_val = static_cast<T>(real_sqrt((static_cast<T>(tmp_val)) + epsilon));
  */
  if (threadIdx.x == 0) {
    auto tmp = pair.first_ / feature_size;
    mean[blockIdx.x] = static_cast<T>(tmp);
    var[blockIdx.x] = static_cast<T>(pair.second_ / feature_size - tmp * tmp);
  }
  __syncthreads();
  mean_val = mean[blockIdx.x];
  var_val = static_cast<T>(real_sqrt(var[blockIdx.x] + epsilon));

  // Step 2: Calculate y
  if (scale != nullptr) {
    if (bias != nullptr) {
      for (int i = beg_idx, j = threadIdx.x; i < end_idx;
           i += BlockDim, j += BlockDim) {
        y[i] = scale[j] * (x[i] - mean_val) / var_val + bias[j];
      }
    } else {
      for (int i = beg_idx, j = threadIdx.x; i < end_idx;
           i += BlockDim, j += BlockDim) {
        y[i] = scale[j] * (x[i] - mean_val) / var_val;
      }
    }
  } else {  // scale == nullptr
    if (bias != nullptr) {
      for (int i = beg_idx, j = threadIdx.x; i < end_idx;
           i += BlockDim, j += BlockDim) {
        y[i] = (x[i] - mean_val) / var_val + bias[j];
      }
    } else {
      for (int i = beg_idx, j = threadIdx.x; i < end_idx;
           i += BlockDim, j += BlockDim) {
        y[i] = (x[i] - mean_val) / var_val;
      }
    }
  }
}

template <typename T>
void LayerNormDirectCUDAFunctor<T>::operator()(
    cudaStream_t stream, const T *x_data, const T *scale_data,
    const T *bias_data, T *y_data, T *mean_data, T *var_data, float epsilon,
    int feature_size, int left_num) {
  switch (GetDesiredBlockDim(feature_size)) {
    FIXED_BLOCK_DIM_CASE(
        LayerNormForward<T, kBlockDim><<<left_num, kBlockDim, 0, stream>>>(
            x_data, scale_data, bias_data, y_data, mean_data, var_data, epsilon,
            feature_size, left_num));
    default:
      PADDLE_THROW("Product from begin_norm_axis to end must be larger than 1");
      break;
  }
}

#undef FIXED_BLOCK_DIM_CASE_BASE
#undef FIXED_BLOCK_DIM_CASE
template class LayerNormDirectCUDAFunctor<float>;
template class LayerNormDirectCUDAFunctor<double>;
}  // namespace math
}  // namespace operators
}  // namespace paddle
