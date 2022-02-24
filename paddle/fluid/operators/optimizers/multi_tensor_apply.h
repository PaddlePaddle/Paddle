// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#ifdef __NVCC__
#include "cub/cub.cuh"
#include "math.h"  // NOLINT
#endif

#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
#include "math.h"  // NOLINT
namespace cub = hipcub;
#endif
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/platform/aligned_vector.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace operators {

template <int NumTensor, int NumChunk>
struct TensorMetaList {
  static constexpr int kTensorNum = NumTensor;
  static constexpr int kChunkNum = NumChunk;

  static_assert(kTensorNum > 0 && kTensorNum < 256,
                "kTensorNum must be inside (0, 256).");
  static_assert(kChunkNum > 0 && kChunkNum < 65536,
                "kChunkNum must be inside (0, 65536).");

  int offsets[kTensorNum + 1];
  uint8_t tensor_ids[kChunkNum];
  uint16_t chunk_ids[kChunkNum];
  int start_tensor_id;
  int start_chunk_id;
};

template <typename Functor, int NumTensor, int NumChunk, typename... Args>
static __global__ void MultiTensorApplyCUDAKernel(
    Functor functor, TensorMetaList<NumTensor, NumChunk> meta, int chunk_size,
    Args... args) {
  const int block_id = blockIdx.x;
  const int tensor_id = meta.tensor_ids[block_id];
  const int chunk_id = static_cast<int>(meta.chunk_ids[block_id]) +
                       (tensor_id == 0) * meta.start_chunk_id;
  const int prev_offset = meta.offsets[tensor_id];
  const int next_offset = meta.offsets[tensor_id + 1];
  const int ptr_offset = prev_offset + chunk_id * chunk_size;
  const int size = min(next_offset - ptr_offset, chunk_size);

  functor(tensor_id + meta.start_tensor_id, chunk_id, ptr_offset, size,
          args...);
}

template <typename T>
static std::string ToString(const T *x, int n) {
  std::vector<T> vec(x, x + n);
  return "[" + string::join_strings(vec, ", ") + "]";
}

template <typename Functor, int BlockDim, int NumTensor, int NumChunk,
          typename... Args>
static void MultiTensorApply(Functor functor, gpuStream_t stream,
                             const int *offsets, int n, int chunk_size,
                             Args... args) {
  if (n == 0) return;

  TensorMetaList<NumTensor, NumChunk> metas;

  int tensor_id = 0;
  int chunk_id = 0;
  int numel_offset = 0;
  metas.start_tensor_id = 0;
  metas.start_chunk_id = 0;
  for (int i = 0; i < n; ++i) {
    auto length = offsets[i + 1] - offsets[i];
    if (tensor_id == 0) {
      metas.start_tensor_id = i;
      metas.offsets[0] = numel_offset;
    }
    metas.offsets[tensor_id + 1] = metas.offsets[tensor_id] + length;
    ++tensor_id;
    numel_offset += length;

    auto chunk_num = (length + chunk_size - 1) / chunk_size;
    int last_launch_chunk_id = 0;
    for (int j = 0; j < chunk_num; ++j) {
      metas.chunk_ids[chunk_id] = j - last_launch_chunk_id;
      metas.tensor_ids[chunk_id] = tensor_id - 1;
      ++chunk_id;

      bool tensor_full = (tensor_id == NumTensor && j + 1 == chunk_num);
      bool block_full = (chunk_id == NumChunk);
      bool last_chunk = (i + 1 == n && j + 1 == chunk_num);

      if (tensor_full || block_full || last_chunk) {
        MultiTensorApplyCUDAKernel<Functor, NumTensor,
                                   NumChunk><<<chunk_id, BlockDim, 0, stream>>>(
            functor, metas, chunk_size, args...);
        chunk_id = 0;
        if (j + 1 == chunk_num) {  // chunk for the current tensor is full
          metas.start_chunk_id = 0;
          tensor_id = 0;
        } else {
          metas.offsets[0] = metas.offsets[tensor_id - 1];
          metas.offsets[1] = metas.offsets[tensor_id];
          metas.start_tensor_id = i;
          metas.start_chunk_id = j + 1;
          last_launch_chunk_id = j + 1;
          tensor_id = 1;
        }
      }
    }
  }
}

template <typename T, int BlockDim, int VecSize>
struct L2NormFunctor {
  using MT = typename details::MPTypeTrait<T>::Type;

  DEVICE void operator()(int tensor_id, int chunk_id, int offset, int size,
                         const T *x, MT *y, int max_chunk_num) const {
    const T *ptr = x + offset;

    using BlockReduce = cub::BlockReduce<MT, BlockDim>;
    __shared__ typename BlockReduce::TempStorage storage;

    MT square_sum = static_cast<MT>(0);
    int i;
    for (i = threadIdx.x * VecSize; i + VecSize <= size;
         i += (BlockDim * VecSize)) {
      platform::AlignedVector<T, VecSize> tmp_vec;
      platform::Load(ptr + i, &tmp_vec);
#pragma unroll
      for (int j = 0; j < VecSize; ++j) {
        auto tmp = static_cast<MT>(tmp_vec[j]);
        square_sum += (tmp * tmp);
      }
    }

    for (; i < size; ++i) {
      auto tmp = static_cast<MT>(ptr[i]);
      square_sum += (tmp * tmp);
    }

    square_sum = BlockReduce(storage).Reduce(square_sum, cub::Sum());
    if (threadIdx.x == 0) {
      y[tensor_id * max_chunk_num + chunk_id] = square_sum;
    }
  }
};

template <typename InT, typename OutT, int BlockDim, bool NeedSqrt>
static __global__ void MultiTensorL2NormReduceAgainCUDAKernel(
    const InT *x, OutT *y, int max_chunk_num) {
  int tensor_id = blockIdx.x;
  x += (tensor_id * max_chunk_num);
  using BlockReduce = cub::BlockReduce<InT, BlockDim>;
  __shared__ typename BlockReduce::TempStorage storage;
  InT sum = static_cast<InT>(0);
  for (int i = threadIdx.x; i < max_chunk_num; i += BlockDim) {
    sum += x[i];
  }
  sum = BlockReduce(storage).Reduce(sum, cub::Sum());
  if (threadIdx.x == 0) {
    if (NeedSqrt) {
      y[blockIdx.x] = static_cast<OutT>(sqrtf(sum));
    } else {
      y[blockIdx.x] = static_cast<OutT>(sum);
    }
  }
}

template <typename T>
static int GetChunkedVecSize(const T *ptr, int chunk_size) {
  static_assert(!std::is_same<T, void>::value, "T cannot be void.");

  constexpr int max_load_bits = 128;
  int valid_vec_size = max_load_bits / CHAR_BIT / sizeof(T);
  auto address = reinterpret_cast<uintptr_t>(ptr);
  constexpr int vec8 = alignof(platform::AlignedVector<T, 8>);
  constexpr int vec4 = alignof(platform::AlignedVector<T, 4>);
  constexpr int vec2 = alignof(platform::AlignedVector<T, 2>);
  if (address % vec8 == 0 && chunk_size % vec8 == 0) {
    return std::min(8, valid_vec_size);
  } else if (address % vec4 == 0 && chunk_size % vec4 == 0) {
    return std::min(4, valid_vec_size);
  } else if (address % vec2 == 0 && chunk_size % vec2 == 0) {
    return std::min(2, valid_vec_size);
  } else {
    return 1;
  }
}

#define PD_VEC_MULTI_TENSOR_APPLY_CASE(__vec_size, ...) \
  case __vec_size: {                                    \
    constexpr int kVecSize = __vec_size;                \
    __VA_ARGS__;                                        \
    break;                                              \
  }

#define PD_VEC_MULTI_TENSOR_APPLY(__vec_size, ...)    \
  do {                                                \
    switch (__vec_size) {                             \
      PD_VEC_MULTI_TENSOR_APPLY_CASE(8, __VA_ARGS__); \
      PD_VEC_MULTI_TENSOR_APPLY_CASE(4, __VA_ARGS__); \
      PD_VEC_MULTI_TENSOR_APPLY_CASE(2, __VA_ARGS__); \
      PD_VEC_MULTI_TENSOR_APPLY_CASE(1, __VA_ARGS__); \
    }                                                 \
  } while (0)

template <typename InT, typename OutT, bool NeedSqrt = false>
static void MultiTensorL2Norm(const platform::CUDADeviceContext &dev_ctx,
                              const InT *x, const int *offsets, int n,
                              OutT *y) {
  if (n == 0) return;

  constexpr int kNumTensor = 110;
  constexpr int kNumChunk = 320;
  constexpr int kBlockDim = 512;

  // TODO(zengjinle): which chunk_size is better?
  constexpr int chunk_size = 2048 * 32;

  int max_chunk_num = -1;
  int vec_size = 8;

  for (int i = 0; i < n; ++i) {
    vec_size = std::min(
        vec_size, GetChunkedVecSize(x + offsets[i] - offsets[0], chunk_size));
    int length = offsets[i + 1] - offsets[i];
    auto tmp_chunk_num = (length + chunk_size - 1) / chunk_size;
    max_chunk_num = std::max(max_chunk_num, tmp_chunk_num);
  }

  using MT = typename details::MPTypeTrait<InT>::Type;
  auto place = dev_ctx.GetPlace();
  auto stream = dev_ctx.stream();
  memory::Buffer tmp_out(place);
  auto *tmp_out_ptr = tmp_out.Alloc<MT>(n * max_chunk_num);
  auto nbytes = n * max_chunk_num * sizeof(MT);
#ifdef PADDLE_WITH_CUDA
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemsetAsync(tmp_out_ptr, 0, nbytes, stream));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(hipMemsetAsync(tmp_out_ptr, 0, nbytes, stream));
#endif

#define PD_LAUNCH_MULTI_TENSOR_APPLY_KERNEL                         \
  do {                                                              \
    using FunctorT = L2NormFunctor<InT, kBlockDim, kVecSize>;       \
    VLOG(10) << __func__ << " " << typeid(InT).name()               \
             << " VecSize = " << kVecSize;                          \
    MultiTensorApply<FunctorT, kBlockDim, kNumTensor, kNumChunk>(   \
        FunctorT(), stream, offsets, n, chunk_size, x, tmp_out_ptr, \
        max_chunk_num);                                             \
  } while (0)

  PD_VEC_MULTI_TENSOR_APPLY(vec_size, PD_LAUNCH_MULTI_TENSOR_APPLY_KERNEL);
#undef PD_LAUNCH_MULTI_TENSOR_APPLY_KERNEL

  MultiTensorL2NormReduceAgainCUDAKernel<MT, OutT, kBlockDim,
                                         NeedSqrt><<<n, kBlockDim, 0, stream>>>(
      tmp_out_ptr, y, max_chunk_num);
}

}  // namespace operators
}  // namespace paddle
