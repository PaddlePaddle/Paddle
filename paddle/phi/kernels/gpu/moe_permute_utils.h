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

inline constexpr int kCumsumBlockSize = 40;
inline constexpr int kCumsumInvalidTag = -1;
inline constexpr int kMaxNumExperts = 64;
inline constexpr int kMaxNumExpertsForOptKernel = 32;

}  // namespace moe

// ============================================================================
// Dispatch utilities: runtime bool -> compile-time bool, zero overhead
// ============================================================================
namespace dispatch {

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
template <typename probs_T>
struct expert_infos {
  int expert_row_idx;
  probs_T expert_probs;

  __device__ __host__ expert_infos()
      : expert_row_idx(-1), expert_probs(probs_T(0)) {}
  __device__ __host__ expert_infos(int idx, probs_T prob)
      : expert_row_idx(idx), expert_probs(prob) {}

  __device__ __host__ expert_infos& operator=(const expert_infos& other) {
    expert_row_idx = other.expert_row_idx;
    expert_probs = other.expert_probs;
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

}  // namespace phi
