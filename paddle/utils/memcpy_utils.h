#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>
#include <cuda.h>
#include <cuda_fp8.h>

#define COPY_ALIGN_BYTE 16

#define CHECK_CUDA_ERROR(call)                                                 \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(error));                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

__device__ __forceinline__ float4 repack_float4_naive_bytewise(
    const float4 &a, const float4 &b,
    const int byte_offset // relative misalignment offset
) {
  // combine 2 float4 into a single float4 with byte-level reordering
  char src_bytes[32];
  *(float4 *)src_bytes = a;
  *(float4 *)(src_bytes + 16) = b;

  float4 result;
  char *dst_bytes = (char *)&result;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    dst_bytes[i] = src_bytes[(byte_offset + i) % 16];
  }
  return result;
}

template <typename T>
__device__ __forceinline__ float4
repack_float4_naive(const float4 &a, const float4 &b,
              const int numel_offset // relative misalignment offset
) {
  // combine 2 float4 into a single float4 with byte-level reordering
  char src_bytes[32];
  *(float4 *)src_bytes = a;
  *(float4 *)(src_bytes + 16) = b;
  float4 result;
  const int numels_per_vec = 16 / sizeof(T);
  char *dst_bytes = (char *)&result;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    dst_bytes[i] = src_bytes[(numel_offset * sizeof(T) + i)];
  }
  return result;
}

template <typename T>
__device__ __forceinline__ float4
repack_float4(const float4 &a, const float4 &b,
              const int numel_offset // relative misalignment offset
) {
  float4 result;

  const int offset_bytes = numel_offset * sizeof(T);

  const int word_offset = offset_bytes >> 2;
  const int byte_offset = offset_bytes & 0x3;
  const int bit_offset = byte_offset * 8;

  const unsigned int *a_ptr = (const unsigned int *)&a;
  const unsigned int *b_ptr = (const unsigned int *)&b;

  unsigned int s0 = (word_offset < 4)   ? a_ptr[word_offset]
                    : (word_offset < 8) ? b_ptr[word_offset - 4]
                                        : 0;
  unsigned int s1 = (word_offset + 1 < 4)   ? a_ptr[word_offset + 1]
                    : (word_offset + 1 < 8) ? b_ptr[word_offset + 1 - 4]
                                            : 0;
  unsigned int s2 = (word_offset + 2 < 4)   ? a_ptr[word_offset + 2]
                    : (word_offset + 2 < 8) ? b_ptr[word_offset + 2 - 4]
                                            : 0;
  unsigned int s3 = (word_offset + 3 < 4)   ? a_ptr[word_offset + 3]
                    : (word_offset + 3 < 8) ? b_ptr[word_offset + 3 - 4]
                                            : 0;
  unsigned int s4 = (word_offset + 4 < 4)   ? a_ptr[word_offset + 4]
                    : (word_offset + 4 < 8) ? b_ptr[word_offset + 4 - 4]
                                            : 0;

  unsigned int r0, r1, r2, r3;

  if (byte_offset == 0) {
    r0 = s0;
    r1 = s1;
    r2 = s2;
    r3 = s3;
  } else {
    asm("shf.l.clamp.b32 %0, %1, %2, %3;"
        : "=r"(r0)
        : "r"(s0), "r"(s1), "r"(bit_offset));
    asm("shf.l.clamp.b32 %0, %1, %2, %3;"
        : "=r"(r1)
        : "r"(s1), "r"(s2), "r"(bit_offset));
    asm("shf.l.clamp.b32 %0, %1, %2, %3;"
        : "=r"(r2)
        : "r"(s2), "r"(s3), "r"(bit_offset));
    asm("shf.l.clamp.b32 %0, %1, %2, %3;"
        : "=r"(r3)
        : "r"(s3), "r"(s4), "r"(bit_offset));
  }

  ((unsigned int *)&result)[0] = r0;
  ((unsigned int *)&result)[1] = r1;
  ((unsigned int *)&result)[2] = r2;
  ((unsigned int *)&result)[3] = r3;

  return result;
}
typedef struct alignas(32) uint32x8_t {
  uint32_t data[8];
} uint32x8_t;

__device__ __forceinline__ uint4
repack_uint4(const uint32x8_t &input,
             const int offset_bytes // absolute byte offset
) {
  uint4 result;
  // 计算字(word)偏移和字节(byte)偏移
  const int word_offset = offset_bytes >> 2;  // 除以4，得到4字节对齐的偏移量
  const int byte_offset = offset_bytes & 0x3; // 取模4，得到字节偏移量
  const int bit_offset = byte_offset * 8;     // 字节偏移转换为位偏移
  // 字节不对齐情况 - 使用位移指令, 由抽屉原理，顶多排布在5个word内
  if(bit_offset !=0){

  asm("shf.l.clamp.b32 %0, %1, %2, %3;"
      : "=r"(result.x)
      : "r"(input.data[word_offset + 0]), "r"(input.data[word_offset + 1]),
        "r"(bit_offset));
  asm("shf.l.clamp.b32 %0, %1, %2, %3;"
      : "=r"(result.y)
      : "r"(input.data[word_offset + 1]), "r"(input.data[word_offset + 2]),
        "r"(bit_offset));
  asm("shf.l.clamp.b32 %0, %1, %2, %3;"
      : "=r"(result.z)
      : "r"(input.data[word_offset + 2]), "r"(input.data[word_offset + 3]),
        "r"(bit_offset));
  asm("shf.l.clamp.b32 %0, %1, %2, %3;"
      : "=r"(result.w)
      : "r"(input.data[word_offset + 3]), "r"(input.data[word_offset + 4]),
        "r"(bit_offset));
  }else{
    result.x = input.data[word_offset + 0];
    result.y = input.data[word_offset + 1];
    result.z = input.data[word_offset + 2];
    result.w = input.data[word_offset + 3];
  }

  return result;
}

template <typename T, int k_vec_numel = (COPY_ALIGN_BYTE / sizeof(T)),
          int k_thread_per_block, bool not_using_shm = true>
__global__ void copy_unaligned(const T *__restrict__ src, T *__restrict__ dst,
                               const size_t numel,
                               const size_t inner_dim_size) {
  // Padding to provide extra pack-loading for rel_misalignment != 0
  constexpr uint8_t k_shm_padding_byte = COPY_ALIGN_BYTE;
  __shared__ __align__(COPY_ALIGN_BYTE) uint8_t
      shm[k_thread_per_block * COPY_ALIGN_BYTE + k_shm_padding_byte * 2];

  const size_t block_offset_base_byte = blockIdx.x * inner_dim_size * sizeof(T);
  const uintptr_t block_src =
      reinterpret_cast<uintptr_t>(src) + block_offset_base_byte;
  const T* block_src_addr = reinterpret_cast<const T*>(block_src);
  const uintptr_t block_dst =
      reinterpret_cast<uintptr_t>(dst) + block_offset_base_byte;
  T* block_dst_addr = reinterpret_cast<T*>(block_dst);
  const uintptr_t block_src_floored =
      block_src & ~(uintptr_t)(COPY_ALIGN_BYTE - 1);
  const uintptr_t block_dst_floored =
      block_dst & ~(uintptr_t)(COPY_ALIGN_BYTE - 1);
  const uintptr_t block_dst_ceiled = block_dst_floored + COPY_ALIGN_BYTE;
  const uint4* block_src_floored_addr = reinterpret_cast<const uint4*>(block_src_floored);
  uint4* block_dst_floored_addr = reinterpret_cast<uint4*>(block_dst_floored);
  const uint8_t src_misalignment = block_src & (COPY_ALIGN_BYTE - 1);
  const uint8_t dst_misalignment = block_dst & (COPY_ALIGN_BYTE - 1);
  const int8_t rel_misalignment = dst_misalignment - src_misalignment;
  const int8_t rel_misalignment_numel = rel_misalignment / sizeof(T);
  // Calculate the maximum address offsets
  const size_t max_offset = inner_dim_size * sizeof(T);
  const size_t max_vec_st_offset =
      max_offset & ~(uintptr_t)(COPY_ALIGN_BYTE - 1);
  const size_t max_vec_numel = max_vec_st_offset / COPY_ALIGN_BYTE;
  const size_t remainder =
      ((reinterpret_cast<uintptr_t>(dst) + max_offset) & 0xF) / sizeof(T);

  // Front non-vectorizable remainder(floored dst block 0)
  const uint32_t front_numel = (block_dst_ceiled - block_dst) / sizeof(T);
  for (int offset = threadIdx.x; offset < front_numel; offset += blockDim.x) {
    block_dst_addr[offset] = block_src_addr[offset];
  }
  //const int shl_byte = 15;
  const int shl_offset = (-rel_misalignment + (rel_misalignment > 0 ? 16 : 0));

  if constexpr (not_using_shm) {
    // -------------------- DEBUG --------------------
    uint4 result, lhs, rhs;
    uint32x8_t pack;
    uint4 *pack_lhs = reinterpret_cast<uint4*>(&pack);
    uint4 *pack_rhs = pack_lhs + 1;
    const bool is_rel_misalignment = !!rel_misalignment;
    const uint8_t abs_rel_misalignment = abs(rel_misalignment);

    for (int offset = threadIdx.x + !! rel_misalignment;
         offset < max_vec_numel; offset += blockDim.x) {
      const uint4* thread_src = block_src_floored_addr + offset - is_rel_misalignment;
      uint4* thread_dst = block_dst_floored_addr + offset;
      *pack_lhs = *thread_src;
      *pack_rhs = *(thread_src + 1);
      result = repack_uint4(pack, shl_offset);
      *thread_dst = result;
    }
    if(threadIdx.x < remainder){
      block_dst_addr[inner_dim_size - 1 - threadIdx.x] = block_src_addr[inner_dim_size - 1 - threadIdx.x];
    }
  } else {
    // Main body of vectorized processing.
    for (size_t offset = (threadIdx.x + !!rel_misalignment) * COPY_ALIGN_BYTE;
         offset < max_vec_st_offset; offset += blockDim.x * COPY_ALIGN_BYTE) {
      using VecT = float4;
      // Adjust inner shared memory offset with padding
      const size_t inner_shm_offset = threadIdx.x * COPY_ALIGN_BYTE;
      uint8_t *shm_load_ptr =
          shm + inner_shm_offset + k_shm_padding_byte; // front padding
      const uintptr_t src_addr = block_src_floored + offset;
      const void *global_src_addr = reinterpret_cast<const void *>(src_addr);
      const uintptr_t smem_addr = reinterpret_cast<uintptr_t>(shm_load_ptr);
      const uint32_t smem_addr32 =
          __cvta_generic_to_shared(reinterpret_cast<void *>(smem_addr));

      // ---------------------- Working ---------------------------
      // TODO: Need to add extra async load to padded shared memory
      //       for rel_misalignment != 0
      if (rel_misalignment > 0 && threadIdx.x == 0) {
        // Head thread fetch more for first vectorized store.
        const uintptr_t prev_src_vec_addr = src_addr - COPY_ALIGN_BYTE;
        const void *global_prev_vec_addr =
            reinterpret_cast<const void *>(prev_src_vec_addr);
#if __CUDA_ARCH__ >= 800
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n"
                     :
                     : "r"(smem_addr32 - 16), "l"(global_prev_vec_addr),
                       "n"(COPY_ALIGN_BYTE));
        // Commit and wait for the async operations
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 0;\n" ::);
#else
        // Fallback for architectures where cp.async is not available
        memcpy(shm_load_ptr, global_src_addr, COPY_ALIGN_BYTE);
#endif
      } else if (rel_misalignment < 0 && threadIdx.x == (blockDim.x - 1)) {
        // Tail thread fetch more for last vectorized store.
        const uintptr_t next_src_vec_addr = src_addr - COPY_ALIGN_BYTE;
        const void *global_next_vec_addr =
            reinterpret_cast<const void *>(next_src_vec_addr);
#if __CUDA_ARCH__ >= 800
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n"
                     :
                     : "r"(smem_addr32 + 16), "l"(global_next_vec_addr),
                       "n"(COPY_ALIGN_BYTE));
        // Commit and wait for the async operations
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 0;\n" ::);
#else
        // Fallback for architectures where cp.async is not available
        memcpy(shm_load_ptr, global_src_addr, COPY_ALIGN_BYTE);
#endif
      }
      // ---------------------- Working ---------------------------
      // Use cp.async if available, else fallback to normal memcpy
#if __CUDA_ARCH__ >= 800
      asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n"
                   :
                   : "r"(smem_addr32), "l"(global_src_addr),
                     "n"(COPY_ALIGN_BYTE));
      // Commit and wait for the async operations
      asm volatile("cp.async.commit_group;\n" ::);
      asm volatile("cp.async.wait_group 0;\n" ::);
#else
      // Fallback for architectures where cp.async is not available
      memcpy(shm_load_ptr, global_src_addr, COPY_ALIGN_BYTE);
#endif
      size_t elem_offset = offset / sizeof(T);
      const uint32_t elem_remaining =
          inner_dim_size - elem_offset / k_vec_numel * k_vec_numel;
      const uint32_t num_elems = min(k_vec_numel, elem_remaining);
      // Compute destination address
      uintptr_t dst_addr = block_dst_floored + offset;
      // Pointers to shared memory vectors
      VecT *shm_vec_ptr = reinterpret_cast<VecT *>(shm_load_ptr);

      __syncthreads();
      // Adjust for misalignment
      if (rel_misalignment != 0) [[likely]] {
        // Misaligned copy(worst case)
        VecT *dst_vec_ptr = reinterpret_cast<VecT *>(dst_addr);
        // src ahead of dst, using shm_vec_ptr[0] and shm_vec_ptr[1]
        // dst ahead of src, using shm_vec_ptr[-1] and shm_vec_ptr[0]
        VecT vec_packed =
            repack_float4<T>(shm_vec_ptr[0 - (rel_misalignment > 0)],
                             shm_vec_ptr[1 - (rel_misalignment > 0)],
                             (COPY_ALIGN_BYTE - src_misalignment) / sizeof(T));
        *dst_vec_ptr = vec_packed;
      } else if (rel_misalignment == 0) {
        // Aligned copy
        VecT data = shm_vec_ptr[0];
        VecT *dst_vec_ptr = reinterpret_cast<VecT *>(dst_addr);
        *dst_vec_ptr = data;
      }
      __syncthreads(); // Ensure shared memory isn't overwritten before all
                       // threads are done
    }
    for (uint32_t offset = threadIdx.x; offset <= remainder;
         offset += blockDim.x) {
      dst[inner_dim_size - offset - 1] = src[inner_dim_size - offset - 1];
    }
  }
}

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

template <typename T, int VEC_NUMEL = (COPY_ALIGN_BYTE / sizeof(T))>
__global__ void optimized_vectorized_async_memcpy_kernel(
    T *__restrict__ dst, const T *__restrict__ src, const size_t num_elements) {

  const size_t block_offset = blockIdx.x * blockDim.x * VEC_NUMEL;
  const size_t tid = threadIdx.x;
  extern __shared__ __align__(16) char smem_buffer[];

  // Deal with misaligned addresses.
  // Global consensus buffer alignment permissions for src and dst.
  uintptr_t src_misalignment = reinterpret_cast<uintptr_t>(src) & 0xF;
  uintptr_t dst_misalignment = reinterpret_cast<uintptr_t>(dst) & 0xF;
  if(src_misalignment != dst_misalignment) return;
  src_misalignment = src_misalignment % 16;
  const int fallback_copy_numel = 16 - src_misalignment / sizeof(T); // Asserting type-aligned at least, no byte magic.
  const int remaining_elements = num_elements - fallback_copy_numel;
  const size_t g_offset =  (src_misalignment ? fallback_copy_numel : 0) + block_offset + tid * VEC_NUMEL;

  // Only block 0 handles misaligned case.
  if(blockIdx.x ==0 && src_misalignment)[[unlikely]]{
    // Handle misaligned case with blockIdx.x == 0 
    for (int i = threadIdx.x; i < fallback_copy_numel; i+=blockDim.x) {
      dst[i] = src[i];
    }
  }
  // Early exit
  if (g_offset >= num_elements)[[unlikely]]
    return;

  // Calculate elements to copy for this thread
  const size_t remaining = num_elements - g_offset;
  const int elements_to_copy = min(static_cast<int>(remaining), VEC_NUMEL);

  // Shared memory offset for this thread
  const size_t smem_offset = tid * VEC_NUMEL * sizeof(T);
  T *smem_ptr = reinterpret_cast<T *>(&smem_buffer[smem_offset]);

  // We've reached alignment, use vectorized copy for the rest
  if (remaining_elements > 0) [[likely]]{
    global_to_shared_load_async<T>(smem_ptr, src + g_offset, elements_to_copy);
    shared_to_global_store<T>(dst + g_offset, smem_ptr, elements_to_copy);
  }
}

template <typename T, int VEC_NUMEL = 16 / sizeof(T)>
void launch_optimized_memcpy(T *dst, const T *src, size_t num_elements,
                             cudaStream_t stream = 0) {
  const int BLOCK_SIZE = 384; // Optimal block size for this kernel
  const size_t num_vectors = (num_elements + VEC_NUMEL - 1) / VEC_NUMEL;
  const size_t num_blocks = (num_vectors + BLOCK_SIZE - 1) / BLOCK_SIZE;

  const size_t smem_size = BLOCK_SIZE * VEC_NUMEL * sizeof(T);

  optimized_vectorized_async_memcpy_kernel<T>
      <<<num_blocks, BLOCK_SIZE, smem_size, stream>>>(dst, src, num_elements);
}

template <typename T>
__global__
void simple_DtoD_uint32(T* __restrict__ dst, const T* __restrict__ src, const uint32_t numel, const uint32_t numel_per_thread)
{
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t base_idx = idx * numel_per_thread;
  for (uint32_t i32_idx = base_idx; i32_idx < base_idx + numel_per_thread; ++i32_idx) {
    dst[i32_idx] = src[i32_idx];
  }

}
template <typename T>
__global__
void simple_DtoD_uint64(T* __restrict__ dst, const T* __restrict__ src, const uint64_t numel_64, const uint32_t numel_per_thread)
{
  const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t base_idx = idx * numel_per_thread;
  for (uint64_t i64_idx = base_idx; i64_idx < base_idx + numel_per_thread; ++i64_idx) {
    dst[i64_idx] = src[i64_idx];
  }
}

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

// Helper function to perform vectorized memory copy
template <typename T>
__device__ __forceinline__ void vectorized_memcpy(T* dst_raw,
                                                  const T* src_raw,
                                                  int num_elements_raw) {
  constexpr int vector_size_in_bytes = 16;
  const int elements_per_vector = vector_size_in_bytes / sizeof(T);
  
  // ---------------- Misalignment fallback -----------------
  uintptr_t byte_misalignment = (reinterpret_cast<uintptr_t>(src_raw) & 0xF) % 16;
  uintptr_t byte_misalignment_dst = (reinterpret_cast<uintptr_t>(dst_raw) & 0xF) % 16;
  if(byte_misalignment != byte_misalignment_dst){
    ;
  }else{
    printf("corresponded misalignment !!\n");

  }
  const int fallback_copy_numel = 16 - byte_misalignment / sizeof(T);

  for(int i = threadIdx.x; i < fallback_copy_numel; i += blockDim.x){
    dst_raw[i] = src_raw[i];
  }

  T* dst = dst_raw + (byte_misalignment? fallback_copy_numel : 0);
  const T* src = src_raw + (byte_misalignment? fallback_copy_numel : 0);
  const int num_elements = num_elements_raw - fallback_copy_numel;

  // ----------------- Vectorized copy -----------------
  // 已知单行token向量化不会超过4G大小，用int节省整数开销
  int num_vectors = num_elements / elements_per_vector;
  int remaining_elements = num_elements % elements_per_vector;

  using VecType = VectorType<T, elements_per_vector>;
  const VecType* src_vec = reinterpret_cast<const VecType*>(src);
  VecType* dst_vec = reinterpret_cast<VecType*>(dst);

#pragma unroll
  for (int idx = threadIdx.x; idx < num_vectors; idx += blockDim.x) {
    dst_vec[idx] = src_vec[idx];
  }

  // 剩余无法向量化处理的元素
  if (remaining_elements > 0) {
    int offset = num_vectors * elements_per_vector;
    for (int i = threadIdx.x; i < remaining_elements; i += blockDim.x) {
      dst[offset + i] = src[offset + i];
    }
  }
}