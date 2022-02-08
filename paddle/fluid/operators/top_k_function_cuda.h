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

#pragma once
#include <stdio.h>
#include <cstdio>
#include <vector>
#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
#endif
#include "paddle/fluid/operators/eigen/eigen_function.h"
#include "paddle/fluid/operators/top_k_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_device_function.h"
#include "paddle/fluid/platform/float16.h"

#ifdef __HIPCC__
namespace rocprim {
namespace detail {
template <>
struct radix_key_codec_base<paddle::platform::float16>
    : radix_key_codec_integral<paddle::platform::float16, uint16_t> {};
}  // namespace detail
}  // namespace rocprim
namespace cub = hipcub;
#else
// set cub base traits in order to handle float16
namespace cub {
template <>
struct NumericTraits<paddle::platform::float16>
    : BaseTraits<FLOATING_POINT, true, false, uint16_t,
                 paddle::platform::float16> {};
}  // namespace cub
#endif

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

struct SegmentOffsetIter {
  EIGEN_DEVICE_FUNC
  explicit SegmentOffsetIter(int num_cols) : num_cols_(num_cols) {}

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int operator()(int idx) const {
    return idx * num_cols_;
  }

  int num_cols_;
};

// Iter using into a column
struct ColumnIndexIter {
  explicit ColumnIndexIter(int num_cols) : num_cols_(num_cols) {}

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int operator()(
      const Eigen::array<int, 1>& ix) const {
    return ix[0] % num_cols_;
  }

  int num_cols_;
};

inline static int GetDesiredBlockDim(int dim) {
  if (dim > 128) {
    return 256;
  } else if (dim > 64) {
    return 128;
  } else if (dim > 32) {
    return 64;
  } else {
    return 32;
  }
}

template <typename T>
__global__ void InitIndex(T* indices, T num_rows, T num_cols) {
  int col_id = threadIdx.x;
  int row_id = blockIdx.x;

  for (int64_t j = row_id; j < num_rows; j += gridDim.x) {
    for (int64_t i = col_id; i < num_cols; i += blockDim.x) {
      indices[j * num_cols + i] = i;
    }
  }
}

template <typename T>
struct Pair {
  __device__ __forceinline__ Pair() {}
  __device__ __forceinline__ Pair(T value, int64_t id) : v(value), id(id) {}

  __device__ __forceinline__ void set(T value, int64_t id) {
    v = value;
    id = id;
  }

  __device__ __forceinline__ void operator=(const Pair<T>& in) {
    v = in.v;
    id = in.id;
  }

  __device__ __forceinline__ bool operator<(const T value) const {
    return (v < value);
  }

  __device__ __forceinline__ bool operator>(const T value) const {
    return (v > value);
  }
  __device__ __forceinline__ bool operator<(const Pair<T>& in) const {
    return (v < in.v) || ((v == in.v) && (id > in.id));
  }

  __device__ __forceinline__ bool operator>(const Pair<T>& in) const {
    return (v > in.v) || ((v == in.v) && (id < in.id));
  }

  T v;
  int64_t id;
};

template <typename T>
__device__ __forceinline__ void AddTo(Pair<T> topk[], const Pair<T>& p,
                                      int beam_size, const bool& largest) {
  for (int k = beam_size - 2; k >= 0; k--) {
    if (largest) {
      if (topk[k] < p) {
        topk[k + 1] = topk[k];
      } else {
        topk[k + 1] = p;
        return;
      }
    } else {
      if (topk[k] > p) {
        topk[k + 1] = topk[k];
      } else {
        topk[k + 1] = p;
        return;
      }
    }
  }
  topk[0] = p;
}

template <typename T, int BlockSize>
__device__ __forceinline__ void GetTopK(Pair<T> topk[], const T* src, int idx,
                                        int dim, int beam_size,
                                        const bool& largest) {
  while (idx < dim) {
    if (largest) {
      if (topk[beam_size - 1] < src[idx]) {
        Pair<T> tmp(src[idx], idx);
        AddTo<T>(topk, tmp, beam_size, largest);
      }
    } else {
      if (topk[beam_size - 1] > src[idx]) {
        Pair<T> tmp(src[idx], idx);
        AddTo<T>(topk, tmp, beam_size, largest);
      }
    }
    idx += BlockSize;
  }
}

template <typename T, int BlockSize>
__device__ __forceinline__ void GetTopK(Pair<T> topk[], const T* src, int idx,
                                        int dim, const Pair<T>& max,
                                        int beam_size, const bool& largest) {
  while (idx < dim) {
    if (largest) {
      if (topk[beam_size - 1] < src[idx]) {
        Pair<T> tmp(src[idx], idx);
        if (tmp < max) {
          AddTo<T>(topk, tmp, beam_size, largest);
        }
      }
    } else {
      if (topk[beam_size - 1] > src[idx]) {
        Pair<T> tmp(src[idx], idx);
        if (tmp > max) {
          AddTo<T>(topk, tmp, beam_size, largest);
        }
      }
    }
    idx += BlockSize;
  }
}

template <typename T, int MaxLength, int BlockSize>
__device__ __forceinline__ void ThreadGetTopK(Pair<T> topk[], int* beam,
                                              int beam_size, const T* src,
                                              bool* firstStep, bool* is_empty,
                                              Pair<T>* max, int dim,
                                              const int tid, bool largest) {
  if (*beam > 0) {
    int length = (*beam) < beam_size ? *beam : beam_size;
    if (*firstStep) {
      *firstStep = false;
      GetTopK<T, BlockSize>(topk, src, tid, dim, length, largest);
    } else {
      for (int k = 0; k < MaxLength; k++) {
        if (k < MaxLength - (*beam)) {
          topk[k] = topk[k + *beam];
        } else {
          topk[k].set(-static_cast<T>(INFINITY), -1);
        }
      }
      if (!(*is_empty)) {
        GetTopK<T, BlockSize>(topk + MaxLength - *beam, src, tid, dim, *max,
                              length, largest);
      }
    }

    *max = topk[MaxLength - 1];
    if ((*max).v == -static_cast<T>(1)) *is_empty = true;
    *beam = 0;
  }
}

template <typename T, int MaxLength, int BlockSize>
__device__ __forceinline__ void BlockReduce(Pair<T>* sh_topk, int* maxid,
                                            Pair<T> topk[], T** topVal,
                                            int64_t** topIds, int* beam, int* k,
                                            const int tid, const int warp,
                                            const bool& largest) {
  while (true) {
    __syncthreads();
    if (tid < BlockSize / 2) {
      if (largest) {
        if (sh_topk[tid] < sh_topk[tid + BlockSize / 2]) {
          maxid[tid] = tid + BlockSize / 2;
        } else {
          maxid[tid] = tid;
        }
      } else {
        if (sh_topk[tid] > sh_topk[tid + BlockSize / 2]) {
          maxid[tid] = tid + BlockSize / 2;
        } else {
          maxid[tid] = tid;
        }
      }
    }
    __syncthreads();
    for (int stride = BlockSize / 4; stride > 0; stride = stride / 2) {
      if (tid < stride) {
        if (largest) {
          if (sh_topk[maxid[tid]] < sh_topk[maxid[tid + stride]]) {
            maxid[tid] = maxid[tid + stride];
          }
        } else {
          if (sh_topk[maxid[tid]] > sh_topk[maxid[tid + stride]]) {
            maxid[tid] = maxid[tid + stride];
          }
        }
      }
      __syncthreads();
    }
    __syncthreads();

    if (tid == 0) {
      **topVal = sh_topk[maxid[0]].v;
      **topIds = sh_topk[maxid[0]].id;
      (*topVal)++;
      (*topIds)++;
    }
    if (tid == maxid[0]) (*beam)++;
    if (--(*k) == 0) break;
    __syncthreads();

    if (tid == maxid[0]) {
      if (*beam < MaxLength) {
        sh_topk[tid] = topk[*beam];
      }
    }
    // NOTE(zcd): temporary solution
    unsigned mask = 0u;
    CREATE_SHFL_MASK(mask, true);

    if (maxid[0] / 32 == warp) {
      if (platform::CudaShuffleSync(mask, *beam, (maxid[0]) % 32, 32) ==
          MaxLength)
        break;
    }
  }
}

/**
 * Each block compute one sample.
 * In a block:
 * 1. every thread get top MaxLength value;
 * 2. merge to sh_topk, block reduce and get max value;
 * 3. go to the second setp, until one thread's topk value is null;
 * 4. go to the first setp, until get the topk value.
 */

template <typename T, int MaxLength, int BlockSize>
__global__ void KeMatrixTopK(T* output, int output_stride, int64_t* indices,
                             const T* src, int lds, int dim, int k,
                             int grid_dim, int num, bool largest = true) {
  __shared__ Pair<T> sh_topk[BlockSize];
  const int tid = threadIdx.x;
  const int warp = threadIdx.x / 32;

  const int bid = blockIdx.x;
  for (int i = bid; i < num; i += grid_dim) {
    int top_num = k;
    __shared__ int maxid[BlockSize / 2];
    T* out = output + i * output_stride;
    int64_t* inds = indices + i * k;
    Pair<T> topk[MaxLength];
    int beam = MaxLength;
    Pair<T> max;
    bool is_empty = false;
    bool firststep = true;

    for (int j = 0; j < MaxLength; j++) {
      if (largest) {
        topk[j].set(-static_cast<T>(INFINITY), -1);
      } else {
        topk[j].set(static_cast<T>(INFINITY), -1);
      }
    }
    while (top_num) {
      ThreadGetTopK<T, MaxLength, BlockSize>(topk, &beam, k, src + i * lds,
                                             &firststep, &is_empty, &max, dim,
                                             tid, largest);

      sh_topk[tid] = topk[0];
      BlockReduce<T, MaxLength, BlockSize>(sh_topk, maxid, topk, &out, &inds,
                                           &beam, &top_num, tid, warp, largest);
    }
  }
}

template <typename T, int MaxLength, int BlockSize>
__global__ void AssignGrad(T* x_grad, const int64_t* indices, const T* out_grad,
                           size_t rows, size_t cols, size_t k) {
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      x_grad[i * cols + j] = 0;
    }
    __syncthreads();
    for (size_t j = 0; j < k; ++j) {
      size_t idx = indices[i * k + j];
      x_grad[i * cols + idx] = out_grad[i * k + j];
    }
  }
}

// the grad assign with the axis
template <typename T>
__global__ void AssignGradWithAxis(const T* grad_out, const int64_t* indices,
                                   T* grad_in, int pre, int post,
                                   int raw_height, int k) {
  // raw_height is the length of topk axis
  for (int i = blockIdx.x; i < pre; i += gridDim.x) {
    int base_index = i * post * k;
    int base_grad = i * post * raw_height;
    for (int j = threadIdx.x; j < raw_height * post; j += blockDim.x) {
      grad_in[base_grad + j] = static_cast<T>(0);
    }
    __syncthreads();
    for (int j = threadIdx.x; j < k * post; j += blockDim.x) {
      int64_t idx_ij = indices[base_index + j];
      int64_t in_ij = base_grad + (idx_ij * post) + (j % post);
      grad_in[in_ij] = grad_out[base_index + j];
    }
  }
}
// use the radix sort for the topk
template <typename T>
bool SortTopk(const platform::CUDADeviceContext& ctx,
              const framework::Tensor* input_tensor, const int64_t num_cols,
              const int64_t num_rows, const int k,
              framework::Tensor* out_tensor, framework::Tensor* indices_tensor,
              bool largest = true) {
  auto cu_stream = ctx.stream();

  Tensor input_indices;
  const std::vector<int64_t> dims = {num_rows, num_cols};
  auto dim = framework::make_ddim(dims);
  input_indices.Resize(dim);
  // input_indices.Resize(num_rows*num_cols);
  input_indices.mutable_data<int64_t>(ctx.GetPlace());
  size_t temp_storage_bytes = -1;

  auto ComputeBlockSize = [](int col) {
    if (col > 512)
      return 1024;
    else if (col > 256 && col <= 512)
      return 512;
    else if (col > 128 && col <= 256)
      return 256;
    else if (col > 64 && col <= 128)
      return 128;
    else
      return 64;
  };
  int block_size = ComputeBlockSize(num_cols);

  unsigned int maxGridDimX = ctx.GetCUDAMaxGridDimSize()[0];
  // actually, int num_rows < max_grid_size
  unsigned int grid_size = num_rows < maxGridDimX
                               ? static_cast<unsigned int>(num_rows)
                               : maxGridDimX;
  // Init a index array
  InitIndex<int64_t><<<grid_size, block_size, 0, cu_stream>>>(
      input_indices.data<int64_t>(), num_rows, num_cols);

  // create iter for counting input
  cub::CountingInputIterator<int64_t> counting_iter(0);
  // segment_offset is used for move to next row
  cub::TransformInputIterator<int64_t, SegmentOffsetIter,
                              cub::CountingInputIterator<int64_t>>
      segment_offsets_t(counting_iter, SegmentOffsetIter(num_cols));

  T* sorted_values_ptr;
  int64_t* sorted_indices_ptr;

  Tensor temp_values;
  Tensor temp_indices;

  const T* input = input_tensor->data<T>();
  T* values = out_tensor->data<T>();
  int64_t* indices = indices_tensor->mutable_data<int64_t>(ctx.GetPlace());

  if (k == num_cols) {
    // Doing a full sort.
    sorted_values_ptr = values;
    sorted_indices_ptr = indices;
  } else {
    temp_values.Resize(dim);
    temp_indices.Resize(dim);
    sorted_values_ptr = temp_values.mutable_data<T>(ctx.GetPlace());
    sorted_indices_ptr = temp_indices.mutable_data<int64_t>(ctx.GetPlace());
  }

  // Get temp storage buffer size, maybe can allocate a fixed buffer to save
  // time.
  if (largest) {
    auto err = cub::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr, temp_storage_bytes, input, sorted_values_ptr,
        input_indices.data<int64_t>(), sorted_indices_ptr, num_cols * num_rows,
        num_rows, segment_offsets_t, segment_offsets_t + 1, 0, sizeof(T) * 8,
        cu_stream);
#ifdef __HIPCC__
    if (err != hipSuccess) {
      LOG(ERROR) << "TopKOP failed as could not launch "
                    "hipcub::DeviceSegmentedRadixSort::SortPairsDescending to "
                    "calculate "
                    "temp_storage_bytes, status: "
                 << hipGetErrorString(err);
      return false;
    }
#else
    if (err != cudaSuccess) {
      LOG(ERROR)
          << "TopKOP failed as could not launch "
             "cub::DeviceSegmentedRadixSort::SortPairsDescending to calculate "
             "temp_storage_bytes, status: "
          << cudaGetErrorString(err);
      return false;
    }
#endif
  } else {
    auto err = cub::DeviceSegmentedRadixSort::SortPairs(
        nullptr, temp_storage_bytes, input, sorted_values_ptr,
        input_indices.data<int64_t>(), sorted_indices_ptr, num_cols * num_rows,
        num_rows, segment_offsets_t, segment_offsets_t + 1, 0, sizeof(T) * 8,
        cu_stream);
#ifdef __HIPCC__
    if (err != hipSuccess) {
      LOG(ERROR) << "TopKOP failed as could not launch "
                    "hipcub::DeviceSegmentedRadixSort::SortPairs to calculate "
                    "temp_storage_bytes, status: "
                 << hipGetErrorString(err);
      return false;
    }
#else
    if (err != cudaSuccess) {
      LOG(ERROR) << "TopKOP failed as could not launch "
                    "cub::DeviceSegmentedRadixSort::SortPairs to calculate "
                    "temp_storage_bytes, status: "
                 << cudaGetErrorString(err);
      return false;
    }
#endif
  }
  Tensor temp_storage;
  temp_storage.mutable_data<uint8_t>(ctx.GetPlace(), temp_storage_bytes);

  if (largest) {
    auto err = cub::DeviceSegmentedRadixSort::SortPairsDescending(
        temp_storage.data<uint8_t>(), temp_storage_bytes, input,
        sorted_values_ptr, input_indices.data<int64_t>(), sorted_indices_ptr,
        num_cols * num_rows, num_rows, segment_offsets_t, segment_offsets_t + 1,
        0, sizeof(T) * 8, cu_stream);
#ifdef __HIPCC__
    if (err != hipSuccess) {
      LOG(ERROR) << "TopKOP failed as could not launch "
                    "hipcub::DeviceSegmentedRadixSort::SortPairsDescending to "
                    "sort input, "
                    "temp_storage_bytes: "
                 << temp_storage_bytes
                 << ", status: " << hipGetErrorString(err);
      return false;
    }
#else
    if (err != cudaSuccess) {
      LOG(ERROR) << "TopKOP failed as could not launch "
                    "cub::DeviceSegmentedRadixSort::SortPairsDescending to "
                    "sort input, "
                    "temp_storage_bytes: "
                 << temp_storage_bytes
                 << ", status: " << cudaGetErrorString(err);
      return false;
    }
#endif
  } else {
    auto err = cub::DeviceSegmentedRadixSort::SortPairs(
        temp_storage.data<uint8_t>(), temp_storage_bytes, input,
        sorted_values_ptr, input_indices.data<int64_t>(), sorted_indices_ptr,
        num_cols * num_rows, num_rows, segment_offsets_t, segment_offsets_t + 1,
        0, sizeof(T) * 8, cu_stream);
#ifdef __HIPCC__
    if (err != hipSuccess) {
      LOG(ERROR) << "TopKOP failed as could not launch "
                    "hipcub::DeviceSegmentedRadixSort::SortPairs to "
                    "sort input, "
                    "temp_storage_bytes: "
                 << temp_storage_bytes
                 << ", status: " << hipGetErrorString(err);
      return false;
    }
#else
    if (err != cudaSuccess) {
      LOG(ERROR) << "TopKOP failed as could not launch "
                    "cub::DeviceSegmentedRadixSort::SortPairs to "
                    "sort input, "
                    "temp_storage_bytes: "
                 << temp_storage_bytes
                 << ", status: " << cudaGetErrorString(err);
      return false;
    }
#endif
  }
  auto& dev = *ctx.eigen_device();
  if (k < num_cols) {
    // copy sliced data to output.
    const Eigen::DSizes<Eigen::DenseIndex, 2> slice_indices{0, 0};
    const Eigen::DSizes<Eigen::DenseIndex, 2> slice_sizes{num_rows, k};
    auto e_indices =
        framework::EigenMatrix<int64_t>::From(*indices_tensor, dim);
    auto e_tmp_indices = framework::EigenMatrix<int64_t>::From(
        static_cast<const Tensor>(temp_indices));

    std::vector<int> odims = {static_cast<int>(num_rows), static_cast<int>(k)};
    auto dim = framework::make_ddim(odims);
    auto e_values = framework::EigenMatrix<T>::From(*out_tensor, dim);
    auto e_tmp_values =
        framework::EigenMatrix<T>::From(static_cast<const Tensor>(temp_values));

    EigenSlice<std::decay_t<decltype(dev)>, int64_t, 2>::Eval(
        dev, e_indices, e_tmp_indices, slice_indices, slice_sizes);
    EigenSlice<std::decay_t<decltype(dev)>, T, 2>::Eval(
        dev, e_values, e_tmp_values, slice_indices, slice_sizes);
  }
  return true;
}
}  // namespace operators
}  // namespace paddle
