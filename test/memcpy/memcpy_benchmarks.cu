#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>
#include "memcpy_utils.h"

#define VERIFY_RESULT
#define UNALIGN_TWEAK 0

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

template <typename T>
bool verify_copy(const T *host_src, const T *host_dst, size_t num_elements) {
  for (size_t i = 0; i < num_elements; ++i) {
    if (host_src[i] != host_dst[i]) {
      std::cerr << "Mismatch at index " << i << ": " << host_src[i]
                << " != " << host_dst[i] << std::endl;
      return false;
    }
  }
  return true;
}

template <typename T>
void benchmark_memcpy(size_t max_size, size_t min_size = 256,
                      int iterations = 10100) {
  constexpr int warmup_repeats = 100;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0, 100.0);

  for (size_t size_raw = min_size; size_raw <= max_size; size_raw *= 2) {
    size_t size =
        size_raw + UNALIGN_TWEAK; // tweak this to make unaligned case.
    size_t byte_size = size * sizeof(T);

    std::vector<T> h_src(size), h_dst(size), h_ref(size);
    std::generate(h_src.begin(), h_src.end(), [&] { return (T)dis(gen); });

    T *d_src, *d_dst_cuda, *d_dst_custom;
    cudaMalloc(&d_src, byte_size);
    cudaMalloc(&d_dst_cuda, byte_size);
    cudaMalloc(&d_dst_custom, byte_size);

    // H2D copy
    cudaMemcpy(d_src, h_src.data(), byte_size, cudaMemcpyHostToDevice);

    // 1. Testing official cudaMemcpy
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
      // warmup
      if (i == warmup_repeats)
        start = std::chrono::high_resolution_clock::now();
      cudaMemcpy(d_dst_cuda, d_src, byte_size, cudaMemcpyDeviceToDevice);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    double memcpy_time = std::chrono::duration<double>(end - start).count();
    double memcpy_bw = ((iterations - warmup_repeats) * byte_size * 2) /
                       (memcpy_time * 1e9); // GB/s

    // Verify official cudaMemcpy result
    cudaMemcpy(h_ref.data(), d_dst_cuda, byte_size, cudaMemcpyDeviceToHost);

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
      // warmup
      if (i == warmup_repeats)
        start = std::chrono::high_resolution_clock::now();
      launch_optimized_memcpy<T>(d_dst_custom, d_src, size);
    }
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    double kernel_time = std::chrono::duration<double>(end - start).count();
    double kernel_bw = ((iterations - warmup_repeats) * byte_size * 2) /
                       (kernel_time * 1e9); // GB/s

    // Verify mine result.
    cudaMemcpy(h_dst.data(), d_dst_custom, byte_size, cudaMemcpyDeviceToHost);

    printf("Size: %10zu elements (%8.2f MB)\t", size,
           byte_size / 1024.0 / 1024.0);
#ifdef VERIFY_RESULT
    bool memcpy_correct = verify_copy(h_src.data(), h_ref.data(), size);
    bool kernel_correct = verify_copy(h_src.data(), h_dst.data(), size);
    printf("cudaMemcpy: %8.2f GB/s (%s)\t", memcpy_bw,
           memcpy_correct ? "PASS" : "FAIL");
    printf("PZW's Custom Kernel: %8.2f GB/s (%s)\t", kernel_bw,
           kernel_correct ? "PASS" : "FAIL");
#else
    printf("cudaMemcpy: %8.2f GB/s \t", memcpy_bw);
    printf("PZW's Custom Kernel: %8.2f GB/s \t", kernel_bw);
#endif
    printf("Ratio: %.2f%%\n", (kernel_bw / memcpy_bw) * 100);

    cudaFree(d_src);
    cudaFree(d_dst_cuda);
    cudaFree(d_dst_custom);
  }
}

int main() {
  std::cout << "\n=== Testing double (8 bytes) ===" << std::endl;
  benchmark_memcpy<double>(128 * 1024 * 1024);
  std::cout << "=== Testing float (4 bytes) ===" << std::endl;
  benchmark_memcpy<float>(256 * 1024 * 1024);
  std::cout << "\n=== Testing int (4 bytes) ===" << std::endl;
  benchmark_memcpy<int>(256 * 1024 * 1024);
  std::cout << "\n=== Testing char (1 bytes) ===" << std::endl;
  benchmark_memcpy<char>(1024 * 1024 * 1024);
  return 0;
}

#undef VERIFY_RESULT