// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <iostream>
#include <limits>

#include "paddle/phi/kernels/funcs/math_cuda_utils.h"

namespace phi {

// ============================================================================
// Compile-time constants for MoE permute/unpermute kernels
// ============================================================================
namespace moe {

// Smallest power-of-2 >= v.  (v must be > 0)
inline constexpr int ceil_pow2(int v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return v + 1;
}

inline constexpr int kCumsumBlockSize = 40;
inline constexpr int kCumsumInvalidTag = -1;
inline constexpr int kMaxNumExperts = 384;
inline constexpr int kMaxNumExpertsForOptKernel = 32;

// FP8-specific tuning knobs for permute_generic_kernel.
// FP8 has ~2x lighter memcpy than BF16, shifting the bottleneck to Phase-1
// scheduling and inter-block cumsum sync. Tune these independently from BF16.
//   kFp8CumsumBlockSize : rows per block   (32 enables warp-ballot
//   optimization) kFp8BlockDimX       : threads per block (tune range: 128 ..
//   512)
inline constexpr int kFp8CumsumBlockSize = 32;
inline constexpr int kFp8BlockDimX = 256;

}  // namespace moe

// ============================================================================
// Dispatch utilities: runtime num_experts -> compile-time bucket
// ============================================================================
namespace dispatch {

// Bucketed NUM_EXPERTS dispatch: selects the smallest compile-time bucket
// >= runtime num_experts to minimize register / shared-memory overhead.
// Buckets: 8, 16, 32, 64, 128, 256, 384
template <typename F>
inline void NumExperts(int num_experts, F&& f) {
  if (num_experts <= 8) {
    f(std::integral_constant<int, 8>{});
  } else if (num_experts <= 16) {
    f(std::integral_constant<int, 16>{});
  } else if (num_experts <= 32) {
    f(std::integral_constant<int, 32>{});
  } else if (num_experts <= 64) {
    f(std::integral_constant<int, 64>{});
  } else if (num_experts <= 128) {
    f(std::integral_constant<int, 128>{});
  } else if (num_experts <= 256) {
    f(std::integral_constant<int, 256>{});
  } else {
    f(std::integral_constant<int, 384>{});
  }
}

// Type tag for compile-time type passing
template <typename T>
struct TypeTag {
  using type = T;
};

// Runtime bool -> compile-time std::bool_constant
template <typename F>
inline auto Bool(bool v, F&& f) {
  return v ? f(std::true_type{}) : f(std::false_type{});
}

// Multi-bool dispatch: flattens nested conditionals
template <typename F>
inline auto Bools(F&& f) {
  return f();
}

// Recursive and variadic decay.
template <typename F, typename... Rest>
inline auto Bools(F&& f, bool first, Rest... rest) {
  return Bool(first, [&](auto tag) {
    return Bools([&](auto... tags) { return f(tag, tags...); }, rest...);
  });
}

// Token type dispatch: dtype -> (TokenT, has_scale)
template <typename F>
inline void TokenType(phi::DataType dtype, F&& f) {
  if (dtype == phi::DataType::BFLOAT16) {
    f(TypeTag<phi::bfloat16>{}, std::false_type{});
  } else if (dtype == phi::DataType::FLOAT8_E4M3FN) {
    f(TypeTag<phi::float8_e4m3fn>{}, std::true_type{});
  }
}

// Probability type dispatch
template <typename F>
inline void ProbType(phi::DataType dtype, F&& f) {
  if (dtype == phi::DataType::BFLOAT16) {
    f(TypeTag<phi::bfloat16>{});
  } else if (dtype == phi::DataType::FLOAT32) {
    f(TypeTag<float>{});
  }
}

// Scale type dispatch
template <typename F>
inline void ScaleType(bool using_ue8m0, F&& f) {
  if (using_ue8m0) {
    f(TypeTag<int32_t>{});
  } else {
    f(TypeTag<float>{});
  }
}

}  // namespace dispatch

// ============================================================================
//                               Type defs
// ============================================================================
template <typename ProbT>
struct ExpertSlotInfo {
  int row_idx;
  ProbT prob;

  __device__ __host__ ExpertSlotInfo() : row_idx(-1), prob(ProbT(0)) {}
  __device__ __host__ ExpertSlotInfo(int idx, ProbT p)
      : row_idx(idx), prob(p) {}

  __device__ __host__ ExpertSlotInfo& operator=(const ExpertSlotInfo& other) {
    row_idx = other.row_idx;
    prob = other.prob;
    return *this;
  }
};

template <paddle::DataType DType>
struct TypeMap;
template <>
struct TypeMap<paddle::DataType::BFLOAT16> {
  using type = phi::bfloat16;
};
template <>
struct TypeMap<paddle::DataType::FLOAT16> {
  using type = phi::float16;
};
template <>
struct TypeMap<paddle::DataType::FLOAT32> {
  using type = float;
};
template <>
struct TypeMap<paddle::DataType::INT32> {
  using type = int;
};
template <>
struct TypeMap<paddle::DataType::INT64> {
  using type = int64_t;
};

template <typename T, int N>
struct alignas(16) VectorType {
  T data[N];
};

template <>
struct alignas(16) VectorType<float, 4> {
  float4 data;  // Built-in CUDA vector type
};

template <>
struct alignas(16) VectorType<__nv_bfloat16, 8> {
  __nv_bfloat16 data[8];
};

template <>
struct alignas(16) VectorType<__nv_fp8_e4m3, 16> {
  __nv_fp8_e4m3 data[16];
};

template <>
struct alignas(16) VectorType<uint8_t, 16> {
  uint8_t data[16];
};

// ============================================================================
//                               Helper functions
// ============================================================================
__host__ __device__ __forceinline__ int32_t align_up(int32_t x,
                                                     int32_t alignment) {
  return ((x + alignment - 1) / alignment) * alignment;
}

template <typename T>
__device__ __forceinline__ void unrolled_memcpy(const T* src,
                                                T* dst,
                                                const int num_elements) {
#pragma unroll
  for (int idx = threadIdx.x; idx < num_elements; idx += blockDim.x) {
    dst[idx] = src[idx];
  }
}
// Helper function to perform vectorized memory copy
template <typename T, int VecSizeInBytes = 16>
__device__ __forceinline__ void vectorized_memcpy(const T* src,
                                                  T* dst,
                                                  const int num_elements) {
  constexpr int vector_size_in_bytes = VecSizeInBytes;
  const int elements_per_vector = vector_size_in_bytes / sizeof(T);

  int num_vectors = num_elements / elements_per_vector;
  int remaining_elements = num_elements % elements_per_vector;

  using VecType = VectorType<T, elements_per_vector>;
  const VecType* src_vec = reinterpret_cast<const VecType*>(src);
  VecType* dst_vec = reinterpret_cast<VecType*>(dst);

#pragma unroll
  for (int idx = threadIdx.x; idx < num_vectors; idx += blockDim.x) {
    dst_vec[idx] = src_vec[idx];
  }

  if (remaining_elements > 0) {
    int offset = num_vectors * elements_per_vector;
    for (int i = threadIdx.x; i < remaining_elements; i += blockDim.x) {
      dst[offset + i] = src[offset + i];
    }
  }
}
static inline bool is_aligned_in_bytes(std::size_t offset,
                                       std::size_t alignment = 16) {
  return (offset & (alignment - 1)) == 0;
}
template <typename T>
__device__ __forceinline__ void try_vectorized_memcpy(const T* src,
                                                      T* dst,
                                                      const int num_elements) {
  bool is_aligned_128bit =
      ((uintptr_t)src & 0xF) == 0 && ((uintptr_t)dst & 0xF) == 0;
  if (is_aligned_128bit) {
    vectorized_memcpy(src, dst, num_elements);
  } else {
    unrolled_memcpy(src, dst, num_elements);
  }
}
template <typename T>
__device__ __forceinline__ void unrolled_memset(T* ptr,
                                                T value,
                                                int num_elements) {
#pragma unroll
  for (int i = threadIdx.x; i < num_elements; i += blockDim.x) {
    ptr[i] = value;
  }
}

template <typename T, int VecSizeInBytes = 16>
__device__ __forceinline__ void vectorized_memset(T* ptr,
                                                  const T value,
                                                  const int num_elements) {
  constexpr int vector_size_in_bytes = VecSizeInBytes;
  const int elements_per_vector = vector_size_in_bytes / sizeof(T);

  int num_vectors = num_elements / elements_per_vector;
  int remaining_elements = num_elements % elements_per_vector;

  using VecType = VectorType<T, elements_per_vector>;
  VecType vec_value;
#pragma unroll
  for (int i = 0; i < elements_per_vector; i++) {
    vec_value.data[i] = value;
  }
  VecType* ptr_vec = reinterpret_cast<VecType*>(ptr);

#pragma unroll
  for (int idx = threadIdx.x; idx < num_vectors; idx += blockDim.x) {
    ptr_vec[idx] = vec_value;
  }

  if (remaining_elements > 0) {
    int offset = num_vectors * elements_per_vector;
    for (int i = threadIdx.x; i < remaining_elements; i += blockDim.x) {
      ptr[offset + i] = value;
    }
  }
}
// ============================================================================
//                               Helper Kernels
// ============================================================================
// Helper kernel for filling padding rows in pre-training circumstances,
// to prevent illegal padding area participating in split matmul.
template <typename TokenT,
          typename ScaleT,
          bool FILLING_X_UNZIPPED,
          bool FILLING_X_SCALE_UNZIPPED,
          bool FILLING_EXPERT_INDICES>
__global__ __launch_bounds__(512) void filling_padding_rows_kernel(
    TokenT* __restrict__ X_unzipped_ptr,
    ScaleT* __restrict__ XScale_unzipped_ptr,
    float* __restrict__ token_prob_unzipped_ptr,
    int* __restrict__ expert_indices_ptr,
    const int cols,
    const int quanted_cols,
    const int* __restrict__ padding_rows) {
  uint32_t rows = padding_rows[blockIdx.x];
  if constexpr (FILLING_X_UNZIPPED) {
    vectorized_memset(
        &X_unzipped_ptr[rows * cols], static_cast<TokenT>(0), cols);
  }
  if constexpr (FILLING_X_SCALE_UNZIPPED) {
    unrolled_memset(&XScale_unzipped_ptr[rows * quanted_cols],
                    static_cast<ScaleT>(0),
                    quanted_cols);
  }
  if (threadIdx.x == 0) {
    token_prob_unzipped_ptr[rows] = static_cast<float>(0.0);
    if constexpr (FILLING_EXPERT_INDICES) {
      expert_indices_ptr[rows] = -1;
    }
  }
}
template <bool FillExpertIndices, int BLOCK_SIZE>
__global__ void routemap_digest_kernel(const int32_t* __restrict__ topk_ids,
                                       int32_t* __restrict__ expert_offset,
                                       int32_t* __restrict__ expert_offset_end,
                                       int32_t* __restrict__ expert_indices,
                                       int32_t numel,
                                       int32_t num_experts,
                                       int32_t padding_alignment,
                                       int32_t override_buffer_size) {
  extern __shared__ int32_t shared[];
  int32_t* hist = shared;                       // [0, ne)
  int32_t* offset_smem = shared + num_experts;  // [ne, 2*ne)

  //  Phase 1: Histogram
  for (int i = threadIdx.x; i < num_experts; i += BLOCK_SIZE) hist[i] = 0;
  __syncthreads();

  // Vectorized int4 bulk: each thread loads 4 int32s per iteration
  const int num_vec4 = numel >> 2;
  const int4* topk_vec4 = reinterpret_cast<const int4*>(topk_ids);

  for (int i = threadIdx.x; i < num_vec4; i += BLOCK_SIZE) {
    int4 vec = topk_vec4[i];
    int32_t elems[4] = {vec.x, vec.y, vec.z, vec.w};
#pragma unroll
    for (int k = 0; k < 4; k++) {
      int32_t expert_id = elems[k];
      if (expert_id >= 0 && expert_id < num_experts)
        atomicAdd(&hist[expert_id], 1);
    }
  }

  // Scalar tail
  for (int i = (num_vec4 << 2) + threadIdx.x; i < numel; i += BLOCK_SIZE) {
    int32_t expert_id = topk_ids[i];
    if (expert_id >= 0 && expert_id < num_experts)
      atomicAdd(&hist[expert_id], 1);
  }
  __syncthreads();

  //  Phase 2: Padded prefix-sum (thread 0)
  if (threadIdx.x == 0) {
    int32_t offset = 0;
    for (int i = 0; i < num_experts; i++) {
      int32_t count = hist[i];
      int32_t padded_count = align_up(count, padding_alignment);
      expert_offset[i] = offset;
      expert_offset_end[i] = offset + count - 1;
      offset_smem[i] = offset;
      offset += padded_count;
    }
  }
  if constexpr (FillExpertIndices) {
    __syncthreads();
    //  Phase 3: Fill expert_indices – per-expert loop, single pass
    //    For each expert: write expert_id for valid slots, -1 for padding.
    for (int expert = 0; expert < num_experts; expert++) {
      int32_t offset = offset_smem[expert];
      int32_t count = hist[expert];
      int32_t padded_count = align_up(count, padding_alignment);
      for (int j = threadIdx.x; j < padded_count; j += BLOCK_SIZE) {
        int32_t write_pos = offset + j;
        if (write_pos < override_buffer_size)
          expert_indices[write_pos] = (j < count) ? expert : -1;
      }
    }
    // Phase 3b: Fill the rest of the buffer with -1 (parallel)
    // Total filled size = offset of last expert + its padded_count
    int32_t total_filled = offset_smem[num_experts - 1] +
                           align_up(hist[num_experts - 1], padding_alignment);
    // All threads parallel fill remaining positions
    for (int i = total_filled + threadIdx.x; i < override_buffer_size;
         i += BLOCK_SIZE)
      expert_indices[i] = -1;
  }
}
}  // namespace phi
