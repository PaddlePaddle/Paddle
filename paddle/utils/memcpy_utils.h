#pragma once
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <limits>


template <typename T, int CP_ASYNC_SIZE>
__device__ __forceinline__ void
global_to_shared_load_async_impl(T *smem_ptr, const T *__restrict__ src,
                                 const int elements_to_copy) {
  int remaining_bytes = elements_to_copy * sizeof(T);
  int processed_elements = 0;
  int remaining_elements = elements_to_copy;
  while (remaining_bytes >= CP_ASYNC_SIZE)
    [[likely]] {
      uintptr_t src_addr = reinterpret_cast<uintptr_t>(src);
      uintptr_t smem_addr = reinterpret_cast<uintptr_t>(smem_ptr);

      const uint32_t smem_addr32 =
          __cvta_generic_to_shared(reinterpret_cast<void *>(smem_addr));
      const void *global_src_addr = reinterpret_cast<const void *>(src_addr);

      asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(
                       smem_addr32),
                   "l"(static_cast<const char *>(global_src_addr)),
                   "n"(CP_ASYNC_SIZE));

      processed_elements = CP_ASYNC_SIZE / sizeof(T);
      remaining_elements = elements_to_copy - processed_elements;
      remaining_bytes -= CP_ASYNC_SIZE;
    }

  // Handle remaining bytes with smaller cp.async sizes or regular copy
  if (remaining_elements > 0) [[unlikely]] {
    const int remaining_bytes = remaining_elements * sizeof(T);
    if constexpr (CP_ASYNC_SIZE == 16) {
      global_to_shared_load_async_impl<T, 8>(smem_ptr + processed_elements,
                                             src + processed_elements,
                                             remaining_bytes / sizeof(T));
    } else if constexpr (CP_ASYNC_SIZE == 8) {
      global_to_shared_load_async_impl<T, 4>(smem_ptr + processed_elements,
                                             src + processed_elements,
                                             remaining_bytes / sizeof(T));
    } else {
// For remaining bytes less than 4 bytes, cp.async haven't support yet, use
// regular copy
#pragma unroll
      for (int i = 0; i < remaining_elements; ++i) {
        smem_ptr[processed_elements + i] = src[processed_elements + i];
      }
      return;
    }
  }
}

template <typename T>
__device__ __forceinline__ void
global_to_shared_load_async(T *smem_ptr, const T *__restrict__ src,
                            const int elements_to_copy) {
  // Start with the largest cp.async size (16 bytes)
  global_to_shared_load_async_impl<T, 16>(smem_ptr, src, elements_to_copy);

  // Commit and wait for the async operations
  asm volatile("cp.async.commit_group;\n" ::);
  asm volatile("cp.async.wait_group 0;\n" ::);
  __syncthreads();
}

template <typename T, int STORE_SIZE = 16>
__forceinline__ __device__ void
shared_to_global_store_impl(T *__restrict__ dst, const T *__restrict__ smem_ptr,
                            const int elements_to_copy) {
  int remaining_bytes = elements_to_copy * sizeof(T);
  int remaining_elements = elements_to_copy;
  int offset = 0;

  // Check if we can perform vectorized stores
  while (remaining_bytes >= STORE_SIZE)
    [[likely]] {
      if constexpr (STORE_SIZE == 16) { // float4
        auto *dst_vec = reinterpret_cast<double2 *>(dst);
        const auto *src_vec = reinterpret_cast<const double2 *>(smem_ptr);
        *dst_vec = *src_vec;
      } else if constexpr (STORE_SIZE == 8) {
        auto *dst_vec = reinterpret_cast<float2 *>(dst);
        const auto *src_vec = reinterpret_cast<const float2 *>(smem_ptr);
        *dst_vec = *src_vec;
      } else if constexpr (STORE_SIZE == 4) {
        auto *dst_vec = reinterpret_cast<float *>(dst);
        const auto *src_vec = reinterpret_cast<const float *>(smem_ptr);
        *dst_vec = *src_vec;
      } else if constexpr (STORE_SIZE == 2) {
        auto *dst_vec = reinterpret_cast<half *>(dst);
        const auto *src_vec = reinterpret_cast<const half *>(smem_ptr);
        *dst_vec = *src_vec;
      } else {
        // Generic implementation for other types/sizes
        memcpy(reinterpret_cast<char *>(dst),
               reinterpret_cast<const char *>(smem_ptr), remaining_bytes);
        return;
      }
      remaining_bytes -= STORE_SIZE;
      offset += STORE_SIZE / sizeof(T);
      remaining_elements = remaining_bytes / sizeof(T);
    }

  if (remaining_bytes > 0) [[unlikely]] {
    if constexpr (STORE_SIZE == 16) {
      shared_to_global_store_impl<T, 8>(dst + offset, &smem_ptr[offset],
                                        remaining_elements);
    } else if constexpr (STORE_SIZE == 8) {
      shared_to_global_store_impl<T, 4>(dst + offset, &smem_ptr[offset],
                                        remaining_elements);
    } else if constexpr (STORE_SIZE == 4) {
      shared_to_global_store_impl<T, 2>(dst + offset, &smem_ptr[offset],
                                        remaining_elements);
    } else {
      shared_to_global_store_impl<T, 1>(dst + offset, &smem_ptr[offset],
                                        remaining_elements);
      return;
    }
  }
}

template <typename T>
__forceinline__ __device__ void
shared_to_global_store(T *__restrict__ dst, const T *__restrict__ smem_ptr,
                       const int elements_to_copy) {
  // Start with the largest possible store size (16 bytes)
  shared_to_global_store_impl<T, 16>(dst, smem_ptr, elements_to_copy);
}

template <typename T, int VEC_SIZE = (16 / sizeof(T))>
__global__ void optimized_vectorized_async_memcpy_kernel(
    T *__restrict__ dst, const T *__restrict__ src, const size_t num_elements) {
  // Aligned shared memory buffer for prefetching
  extern __shared__ __align__(16) char smem_buffer[];

  const size_t tid = threadIdx.x;
  const size_t block_offset = blockIdx.x * blockDim.x * VEC_SIZE;
  const size_t g_offset = block_offset + tid * VEC_SIZE;

  // Early exit
  if (g_offset >= num_elements)
    return;

  // Calculate elements to copy for this thread
  const size_t remaining = num_elements - g_offset;
  const int elements_to_copy = min(static_cast<int>(remaining), VEC_SIZE);

  // Shared memory offset for this thread
  const size_t smem_offset = tid * VEC_SIZE * sizeof(T);
  T *smem_ptr = reinterpret_cast<T *>(&smem_buffer[smem_offset]);

  // Call the inlined function to handle the global to shared copy
  global_to_shared_load_async<T>(smem_ptr, src + g_offset, elements_to_copy);

  // Call the inlined function to handle the shared to global copy
  shared_to_global_store<T>(dst + g_offset, smem_ptr, elements_to_copy);
}

template <typename T, int VEC_SIZE = 16 / sizeof(T)>
void launch_optimized_memcpy(T *dst, const T *src, size_t num_elements,
                             cudaStream_t stream = 0) {
  const int BLOCK_SIZE = 384; // Optimal block size for this kernel
  const size_t num_vectors = (num_elements + VEC_SIZE - 1) / VEC_SIZE;
  const size_t num_blocks = (num_vectors + BLOCK_SIZE - 1) / BLOCK_SIZE;

  const size_t smem_size = BLOCK_SIZE * VEC_SIZE * sizeof(T);

  optimized_vectorized_async_memcpy_kernel<T>
      <<<num_blocks, BLOCK_SIZE, smem_size, stream>>>(dst, src, num_elements);
}

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


// cuda结构体拷贝
template <typename T, int N>
struct alignas(16) VectorType {
  T data[N];
};

// 128Byte对齐的结构体
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
__device__ __forceinline__ void vectorized_memcpy(T* dst,
                                                  const T* src,
                                                  int num_elements) {
  constexpr int vector_size_in_bytes = 16;
  const int elements_per_vector = vector_size_in_bytes / sizeof(T);

  // Compute the misalignment of src and dst pointers
  std::uintptr_t src_ptr = reinterpret_cast<std::uintptr_t>(src);
  std::uintptr_t dst_ptr = reinterpret_cast<std::uintptr_t>(dst);

  size_t src_align = src_ptr % vector_size_in_bytes;
  size_t dst_align = dst_ptr % vector_size_in_bytes;

  // Calculate the number of elements to align both pointers
  int elements_to_align = 0;
  if (src_align != dst_align) {
    // Pointers have different misalignment, need to align both
    size_t align_bytes = vector_size_in_bytes - std::max(src_align, dst_align);
    elements_to_align = (align_bytes + sizeof(T) - 1) / sizeof(T);
    elements_to_align = min(elements_to_align, num_elements);
  } else if (src_align != 0) {
    // Pointers have same misalignment but are not aligned
    size_t align_bytes = vector_size_in_bytes - src_align;
    elements_to_align = (align_bytes + sizeof(T) - 1) / sizeof(T);
    elements_to_align = min(elements_to_align, num_elements);
  }

  // Copy initial unaligned elements scalar-wise
  for (int idx = threadIdx.x; idx < elements_to_align; idx += blockDim.x) {
    dst[idx] = src[idx];
  }

  // Adjust pointers and number of elements
  src += elements_to_align;
  dst += elements_to_align;
  num_elements -= elements_to_align;

  // Proceed with vectorized copying for the aligned portion
  int num_vectors = num_elements / elements_per_vector;
  int remaining_elements = num_elements % elements_per_vector;

  using VecType = VectorType<T, elements_per_vector>;
  const VecType* src_vec = reinterpret_cast<const VecType*>(src);
  VecType* dst_vec = reinterpret_cast<VecType*>(dst);

#pragma unroll
  for (int idx = threadIdx.x; idx < num_vectors; idx += blockDim.x) {
    dst_vec[idx] = src_vec[idx];
  }

  // Copy any remaining elements scalar-wise
  int offset = elements_to_align + num_vectors * elements_per_vector;
  for (int idx = threadIdx.x; idx < remaining_elements; idx += blockDim.x) {
    dst[offset + idx] = src[offset + idx];
  }
}
