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
#include "paddle/fluid/operators/top_k_function_cuda.h"
#include "paddle/fluid/platform/device/gpu/gpu_device_function.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

inline void GetDims(const framework::DDim& dim, int axis, int* pre, int* n,
                    int* post) {
  *pre = 1;
  *post = 1;
  *n = dim[axis];
  for (int i = 0; i < axis; ++i) {
    (*pre) *= dim[i];
  }
  for (int i = axis + 1; i < dim.size(); ++i) {
    (*post) *= dim[i];
  }
}

template <typename T>
__device__ __forceinline__ void AddTo(Pair<T> topk[], const Pair<T>& p,
                                      int beam_size) {
  for (int k = beam_size - 2; k >= 0; k--) {
    if (topk[k] > p) {
      topk[k + 1] = topk[k];
    } else {
      topk[k + 1] = p;
      return;
    }
  }
  topk[0] = p;
}

template <typename T, int BlockSize>
__device__ __forceinline__ void GetKth(Pair<T> topk[], const T* src, int idx,
                                       int dim, int beam_size) {
  while (idx < dim) {
    if (topk[beam_size - 1] > src[idx]) {
      Pair<T> tmp(src[idx], idx);
      AddTo<T>(topk, tmp, beam_size);
    }
    idx += BlockSize;
  }
}

template <typename T, int BlockSize>
__device__ __forceinline__ void GetKth(Pair<T> topk[], const T* src, int idx,
                                       int dim, const Pair<T>& max,
                                       int beam_size) {
  while (idx < dim) {
    if (topk[beam_size - 1] > src[idx]) {
      Pair<T> tmp(src[idx], idx);
      if (tmp > max) {
        AddTo<T>(topk, tmp, beam_size);
      }
    }
    idx += BlockSize;
  }
}

template <typename T, int MaxLength, int BlockSize>
__device__ __forceinline__ void ThreadGetK(Pair<T> topk[], int* beam,
                                           int beam_size, const T* src,
                                           bool* firstStep, bool* is_empty,
                                           Pair<T>* max, int dim,
                                           const int tid) {
  if (*beam > 0) {
    int length = (*beam) < beam_size ? *beam : beam_size;
    if (*firstStep) {
      *firstStep = false;
      GetKth<T, BlockSize>(topk, src, tid, dim, length);
    } else {
      for (int k = 0; k < MaxLength; k++) {
        if (k < MaxLength - (*beam)) {
          topk[k] = topk[k + *beam];
        } else {
          topk[k].set(-static_cast<T>(INFINITY), -1);
        }
      }
      if (!(*is_empty)) {
        GetKth<T, BlockSize>(topk + MaxLength - *beam, src, tid, dim, *max,
                             length);
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
                                            const int tid, const int warp) {
  while (true) {
    __syncthreads();
    if (tid < BlockSize / 2) {
      if (sh_topk[tid] > sh_topk[tid + BlockSize / 2]) {
        maxid[tid] = tid + BlockSize / 2;
      } else {
        maxid[tid] = tid;
      }
    }
    __syncthreads();
    for (int stride = BlockSize / 4; stride > 0; stride = stride / 2) {
      if (tid < stride) {
        if (sh_topk[maxid[tid]] > sh_topk[maxid[tid + stride]]) {
          maxid[tid] = maxid[tid + stride];
        }
      }
      __syncthreads();
    }
    __syncthreads();

    if (*k == 1 && tid == 0) {
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

template <typename T>
bool SortKthvalue(const platform::CUDADeviceContext& ctx,
                  const framework::Tensor* input_tensor, const int64_t num_cols,
                  const int64_t num_rows, const int k,
                  framework::Tensor* out_tensor,
                  framework::Tensor* indices_tensor) {
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

  unsigned int maxGridDimX = ctx.GetCUDAMaxGridDimSize().x;
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

  temp_values.Resize(dim);
  temp_indices.Resize(dim);
  sorted_values_ptr = temp_values.mutable_data<T>(ctx.GetPlace());
  sorted_indices_ptr = temp_indices.mutable_data<int64_t>(ctx.GetPlace());

  // Get temp storage buffer size, maybe can allocate a fixed buffer to save
  // time.
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
  Tensor temp_storage;
  temp_storage.mutable_data<uint8_t>(ctx.GetPlace(), temp_storage_bytes);

  err = cub::DeviceSegmentedRadixSort::SortPairs(
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
               << temp_storage_bytes << ", status: " << hipGetErrorString(err);
    return false;
  }
#else
  if (err != cudaSuccess) {
    LOG(ERROR) << "TopKOP failed as could not launch "
                  "cub::DeviceSegmentedRadixSort::SortPairs to "
                  "sort input, "
                  "temp_storage_bytes: "
               << temp_storage_bytes << ", status: " << cudaGetErrorString(err);
    return false;
  }
#endif
  auto& dev = *ctx.eigen_device();
  // copy sliced data to output.
  const Eigen::DSizes<Eigen::DenseIndex, 2> slice_indices{0, k - 1};
  const Eigen::DSizes<Eigen::DenseIndex, 2> slice_sizes{num_rows, 1};
  auto e_indices = framework::EigenMatrix<int64_t>::From(*indices_tensor, dim);
  auto e_tmp_indices = framework::EigenMatrix<int64_t>::From(
      static_cast<const Tensor>(temp_indices));

  std::vector<int> odims = {static_cast<int>(num_rows), static_cast<int>(1)};
  dim = framework::make_ddim(odims);
  auto e_values = framework::EigenMatrix<T>::From(*out_tensor, dim);
  auto e_tmp_values =
      framework::EigenMatrix<T>::From(static_cast<const Tensor>(temp_values));

  EigenSlice<std::decay_t<decltype(dev)>, int64_t, 2>::Eval(
      dev, e_indices, e_tmp_indices, slice_indices, slice_sizes);
  EigenSlice<std::decay_t<decltype(dev)>, T, 2>::Eval(
      dev, e_values, e_tmp_values, slice_indices, slice_sizes);
  return true;
}

template <typename T, int MaxLength, int BlockSize>
__global__ void KeMatrixKthvalue(T* output, int64_t* indices, const T* src,
                                 int lds, int dim, int k, int grid_dim,
                                 int num) {
  __shared__ Pair<T> sh_topk[BlockSize];
  const int tid = threadIdx.x;
  const int warp = threadIdx.x / 32;
  int top_num = k;
  const int bid = blockIdx.x;
  for (int i = bid; i < num; i += grid_dim) {
    __shared__ int maxid[BlockSize / 2];
    T* out = output + i;
    int64_t* inds = indices + i;

    Pair<T> topk[MaxLength];
    int beam = MaxLength;
    Pair<T> max(-static_cast<T>(INFINITY), -1);
    bool is_empty = false;
    bool firststep = true;

    for (int j = 0; j < MaxLength; j++) {
      topk[j].set(static_cast<T>(NAN), -1);
    }
    ThreadGetK<T, MaxLength, BlockSize>(topk, &beam, k, src + i * lds,
                                        &firststep, &is_empty, &max, dim, tid);

    sh_topk[tid] = topk[0];
    BlockReduce<T, MaxLength, BlockSize>(sh_topk, maxid, topk, &out, &inds,
                                         &beam, &top_num, tid, warp);
  }
}

}  // namespace operators
}  // namespace paddle
