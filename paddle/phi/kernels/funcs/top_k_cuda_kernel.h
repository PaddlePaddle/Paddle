// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#ifndef PADDLE_PHI_KERNELS_FUNCS_TOP_K_CUDA_KERNEL_H_
#define PADDLE_PHI_KERNELS_FUNCS_TOP_K_CUDA_KERNEL_H_

// GPU TopK kernel implementation using radix-select and multi-tier sorting.

#include <algorithm>
#include <cstdint>
#include <limits>

// Include top_k_function_cuda.h to get CUB NumericTraits for float16/bfloat16.
// This header includes cub/cub.cuh and defines the required traits.
#include "paddle/phi/kernels/funcs/top_k_function_cuda.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

#include "paddle/phi/kernels/argsort_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/take_along_axis_kernel.h"

// ============================================================================
// Helper definitions
// All helpers are placed in an anonymous namespace to avoid ODR conflicts
// with Paddle's existing implementations.
// ============================================================================

namespace topk_detail {

// Stream type alias: gpuStream_t is in phi:: namespace, bring it into scope
using phi::gpuStream_t;

// --- Constants ---
constexpr int MAX_TENSORINFO_DIMS = 25;
constexpr int64_t MAX_GRID_SIZE = 65535LL;

// --- ceil_div and round_up ---
template <typename T>
__host__ __device__ __forceinline__ T topk_ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

template <typename T>
__host__ __device__ __forceinline__ T topk_round_up(T a, T b) {
  return topk_ceil_div(a, b) * b;
}

// --- getGridFromTiles ---
inline bool getGridFromTiles(int64_t gridTiles, dim3* grid) {
  if (gridTiles > MAX_GRID_SIZE * MAX_GRID_SIZE * MAX_GRID_SIZE) {
    return false;
  }
  int64_t gridX = gridTiles > MAX_GRID_SIZE ? MAX_GRID_SIZE : gridTiles;
  int64_t gridY = 1;
  int64_t gridZ = 1;
  if (gridTiles > MAX_GRID_SIZE) {
    gridTiles = topk_ceil_div(gridTiles, (int64_t)MAX_GRID_SIZE);
    gridY = gridTiles > MAX_GRID_SIZE ? MAX_GRID_SIZE : gridTiles;
    if (gridTiles > MAX_GRID_SIZE) {
      gridTiles = topk_ceil_div(gridTiles, (int64_t)MAX_GRID_SIZE);
      gridZ = gridTiles > MAX_GRID_SIZE ? MAX_GRID_SIZE : gridTiles;
    }
  }
  *grid = dim3(gridX, gridY, gridZ);
  return true;
}

// --- getLinearBlockId ---
template <typename index_t>
__device__ __forceinline__ index_t getLinearBlockId() {
  return static_cast<index_t>(blockIdx.z) * gridDim.y * gridDim.x +
         static_cast<index_t>(blockIdx.y) * gridDim.x + blockIdx.x;
}

// --- doLdg ---
// Generic fallback for custom types (phi::float16, phi::bfloat16, etc.)
template <typename T>
__device__ __forceinline__ T doLdg(const T* p) {
  return *p;
}

// Specializations for built-in types that support __ldg
#if !defined(__HIPCC__)
template <>
__device__ __forceinline__ float doLdg(const float* p) {
#if __CUDA_ARCH__ >= 350
  return __ldg(p);
#else
  return *p;
#endif
}
template <>
__device__ __forceinline__ double doLdg(const double* p) {
#if __CUDA_ARCH__ >= 350
  return __ldg(p);
#else
  return *p;
#endif
}
template <>
__device__ __forceinline__ int doLdg(const int* p) {
#if __CUDA_ARCH__ >= 350
  return __ldg(p);
#else
  return *p;
#endif
}
template <>
__device__ __forceinline__ unsigned int doLdg(const unsigned int* p) {
#if __CUDA_ARCH__ >= 350
  return __ldg(p);
#else
  return *p;
#endif
}
template <>
__device__ __forceinline__ long long doLdg(  // NOLINT
    const long long* p) {                    // NOLINT
#if __CUDA_ARCH__ >= 350
  return __ldg(p);
#else
  return *p;
#endif
}
template <>
__device__ __forceinline__ unsigned long long doLdg(  // NOLINT
    const unsigned long long* p) {                    // NOLINT
#if __CUDA_ARCH__ >= 350
  return __ldg(p);
#else
  return *p;
#endif
}
template <>
__device__ __forceinline__ int16_t doLdg(const int16_t* p) {
#if __CUDA_ARCH__ >= 350
  return __ldg(p);
#else
  return *p;
#endif
}
#endif  // !__HIPCC__

// --- Bitfield ---
template <typename T>
struct Bitfield {};

template <>
struct Bitfield<unsigned int> {
  static __device__ __forceinline__ unsigned int getBitfield(unsigned int val,
                                                             int pos,
                                                             int len) {
    unsigned int ret;
#if defined(__HIPCC__)
    ret = (val >> pos) & ((1u << len) - 1u);
#else
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(val), "r"(pos), "r"(len));
#endif
    return ret;
  }

  static __device__ __forceinline__ unsigned int setBitfield(
      unsigned int val, unsigned int to_insert, int pos, int len) {
    unsigned int ret;
#if defined(__HIPCC__)
    unsigned int mask = ((1u << len) - 1u) << pos;
    ret = (val & ~mask) | ((to_insert << pos) & mask);
#else
    asm("bfi.b32 %0, %1, %2, %3, %4;"
        : "=r"(ret)
        : "r"(to_insert), "r"(val), "r"(pos), "r"(len));
#endif
    return ret;
  }
};

template <>
struct Bitfield<uint64_t> {
  static __device__ __forceinline__ uint64_t getBitfield(uint64_t val,
                                                         int pos,
                                                         int len) {
    uint64_t ret;
#if defined(__HIPCC__)
    ret = (val >> pos) & ((1ULL << len) - 1ULL);
#else
    asm("bfe.u64 %0, %1, %2, %3;" : "=l"(ret) : "l"(val), "r"(pos), "r"(len));
#endif
    return ret;
  }

  static __device__ __forceinline__ uint64_t setBitfield(uint64_t val,
                                                         uint64_t to_insert,
                                                         int pos,
                                                         int len) {
    uint64_t ret;
#if defined(__HIPCC__)
    uint64_t mask = ((1ULL << len) - 1ULL) << pos;
    ret = (val & ~mask) | ((to_insert << pos) & mask);
#else
    asm("bfi.b64 %0, %1, %2, %3, %4;"
        : "=l"(ret)
        : "l"(to_insert), "l"(val), "r"(pos), "r"(len));
#endif
    return ret;
  }
};

// --- getLaneId / getLaneMaskLe ---
__device__ __forceinline__ int getLaneId() {
#if defined(__HIPCC__)
  return __lane_id();
#else
  int laneId;
  asm("mov.s32 %0, %%laneid;" : "=r"(laneId));
  return laneId;
#endif
}

__device__ __forceinline__ unsigned getLaneMaskLe() {
#if defined(__HIPCC__)
  // HIP warp size is 64, construct mask for lanes <= current lane
  return (getLaneId() == 63) ? 0xFFFFFFFFFFFFFFFFULL
                             : (1ULL << (getLaneId() + 1)) - 1ULL;
#else
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_le;" : "=r"(mask));
  return mask;
#endif
}

__device__ __forceinline__ unsigned getLaneMaskLt() {
#if defined(__HIPCC__)
  return (getLaneId() == 0) ? 0ULL : (1ULL << getLaneId()) - 1ULL;
#else
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
  return mask;
#endif
}

// --- WARP macros ---
#ifdef __HIPCC__
#define TOPK_WARP_SIZE 64
#define TOPK_WARP_BALLOT(PREDICATE) __ballot((PREDICATE))
#define TOPK_WARP_BALLOT_MASK(PREDICATE, MASK) __ballot((PREDICATE))
#define TOPK_WARP_SHFL_DOWN(VAL, DELTA) \
  __shfl_down((VAL), static_cast<unsigned int>(DELTA))
#else
#define TOPK_WARP_SIZE 32
#define TOPK_WARP_BALLOT(PREDICATE) __ballot_sync(0xffffffff, (PREDICATE))
#define TOPK_WARP_BALLOT_MASK(PREDICATE, MASK) \
  __ballot_sync((MASK), (PREDICATE))
#define TOPK_WARP_SHFL_DOWN(VAL, DELTA) \
  __shfl_down_sync(0xffffffff, (VAL), static_cast<unsigned int>(DELTA))
#endif

// --- TopKTypeConfig ---
template <typename T>
struct TopKTypeConfig {};

template <>
struct TopKTypeConfig<float> {
  typedef uint32_t RadixType;

  static inline __device__ RadixType convert(float v) {
    RadixType x = __float_as_int(v);
    RadixType mask = (x & 0x80000000) ? 0xffffffff : 0x80000000;
    return (v == v) ? (x ^ mask) : 0xffffffff;
  }

  static inline __device__ float deconvert(RadixType v) {
    RadixType mask = (v & 0x80000000) ? 0x80000000 : 0xffffffff;
    return __int_as_float(v ^ mask);
  }
};

template <>
struct TopKTypeConfig<double> {
  typedef uint64_t RadixType;

  static inline __device__ RadixType convert(double v) {
    RadixType x = __double_as_longlong(v);
    RadixType mask = -((x >> 63)) | 0x8000000000000000;
    return (v == v) ? (x ^ mask) : 0xffffffffffffffff;
  }

  static inline __device__ double deconvert(RadixType v) {
    RadixType mask = ((v >> 63) - 1) | 0x8000000000000000;
    return __longlong_as_double(v ^ mask);
  }
};

template <>
struct TopKTypeConfig<int32_t> {
  typedef uint32_t RadixType;

  static inline __device__ RadixType convert(int32_t v) {
    static_assert(sizeof(int) == 4, "");
    return 2147483648u + v;
  }

  static inline __device__ int32_t deconvert(RadixType v) {
    return v - 2147483648u;
  }
};

template <>
struct TopKTypeConfig<int64_t> {
  typedef uint64_t RadixType;

  static inline __device__ RadixType convert(int64_t v) {
    static_assert(sizeof(int64_t) == 8, "");
    return 9223372036854775808ull + v;
  }

  static inline __device__ int64_t deconvert(RadixType v) {
    return v - 9223372036854775808ull;
  }
};

template <>
struct TopKTypeConfig<phi::dtype::float16> {
  typedef uint32_t RadixType;

  static inline __device__ RadixType convert(phi::dtype::float16 v) {
    RadixType x = __half_as_ushort(v.to_half());
    RadixType mask = (x & 0x00008000) ? 0x0000ffff : 0x00008000;
    half v_h = v.to_half();
    return (v_h == v_h) ? (x ^ mask) : 0xffff;
  }

  static inline __device__ phi::dtype::float16 deconvert(RadixType v) {
    RadixType mask = (v & 0x00008000) ? 0x00008000 : 0x0000ffff;
    return static_cast<phi::dtype::float16>(__ushort_as_half(v ^ mask));
  }
};

template <>
struct TopKTypeConfig<phi::dtype::bfloat16> {
  typedef uint32_t RadixType;

  static inline __device__ RadixType convert(phi::dtype::bfloat16 v) {
    RadixType x = v.x;
    RadixType mask = (x & 0x00008000) ? 0x0000ffff : 0x00008000;
    return (v == v) ? (x ^ mask) : 0xffff;
  }

  static inline __device__ phi::dtype::bfloat16 deconvert(RadixType v) {
    RadixType mask = (v & 0x00008000) ? 0x00008000 : 0x0000ffff;
    phi::dtype::bfloat16 r;
    r.x = (v ^ mask);
    return r;
  }
};

// uint8_t is needed by the radix select
template <>
struct TopKTypeConfig<uint8_t> {
  typedef uint32_t RadixType;

  static inline __device__ RadixType convert(uint8_t v) { return v; }

  static inline __device__ uint8_t deconvert(RadixType v) { return v; }
};

// --- TensorInfo ---
template <typename T, typename IndexType>
struct TensorInfo {
  T* data;
  IndexType sizes[MAX_TENSORINFO_DIMS];
  IndexType strides[MAX_TENSORINFO_DIMS];
  int dims;

  // collapse_dims: merges contiguous dimensions for efficient indexing
  // See note on [collapse dims].
  int collapseDims(const int excludeDim = -1) {
    int stopDim = (excludeDim == -1) ? dims : excludeDim;
    int newIndex = -1;
    int oldIndex = 0;
    int remappedExcludedDim = -1;

    while (oldIndex < dims) {
      // Finds a dimension to collapse into
      for (; oldIndex < stopDim; ++oldIndex) {
        if (sizes[oldIndex] == 1) {
          continue;
        }

        ++newIndex;
        sizes[newIndex] = sizes[oldIndex];
        strides[newIndex] = strides[oldIndex];
        ++oldIndex;
        break;
      }

      // Collapses dims
      for (; oldIndex < stopDim; ++oldIndex) {
        if (sizes[oldIndex] == 1) {
          continue;
        }

        if (strides[newIndex] == sizes[oldIndex] * strides[oldIndex]) {
          sizes[newIndex] *= sizes[oldIndex];
          strides[newIndex] = strides[oldIndex];
        } else {
          ++newIndex;
          sizes[newIndex] = sizes[oldIndex];
          strides[newIndex] = strides[oldIndex];
        }
      }

      // Handles excludeDim being set (oldIndex == excludeDim)
      if (oldIndex != dims) {
        // Preserves excluded dimension
        ++newIndex;
        sizes[newIndex] = sizes[oldIndex];
        strides[newIndex] = strides[oldIndex];
        remappedExcludedDim = newIndex;

        // Restarts iteration after excludeDim
        ++oldIndex;
        stopDim = dims;
      }
    }

    // Handles special case of all dims size 1
    if (newIndex == -1 || (newIndex == 0 && sizes[0] == 1)) {
      dims = 1;
      sizes[0] = 1;
      strides[0] = 1;
      return 0;
    }

    dims = newIndex + 1;
    return remappedExcludedDim;
  }
};

// --- IndexToOffset ---
template <typename T, typename IndexType, int Dims>
struct IndexToOffset {
  static __host__ __device__ IndexType
  get(IndexType linearId, const TensorInfo<T, IndexType>& info) {
    IndexType offset = 0;
    for (int i = Dims - 1; i > 0; --i) {
      IndexType curDimIndex = linearId % info.sizes[i];
      offset += curDimIndex * info.strides[i];
      linearId /= info.sizes[i];
    }
    return offset + linearId * info.strides[0];
  }
};

// Specialization for Dim == -1 (runtime dims)
template <typename T, typename IndexType>
struct IndexToOffset<T, IndexType, -1> {
  static __host__ __device__ IndexType
  get(IndexType linearId, const TensorInfo<T, IndexType>& info) {
    IndexType offset = 0;
    for (int i = info.dims - 1; i > 0; --i) {
      IndexType curDimIndex = linearId % info.sizes[i];
      offset += curDimIndex * info.strides[i];
      linearId /= info.sizes[i];
    }
    return offset + linearId * info.strides[0];
  }
};

// Specialization for Dim == 1
template <typename T, typename IndexType>
struct IndexToOffset<T, IndexType, 1> {
  static __host__ __device__ IndexType
  get(IndexType linearId, const TensorInfo<T, IndexType>& info) {
    return linearId * info.strides[0];
  }
};

// Specialization for Dim == 2
template <typename T, typename IndexType>
struct IndexToOffset<T, IndexType, 2> {
  static __host__ __device__ IndexType
  get(IndexType linearId, const TensorInfo<T, IndexType>& info) {
    IndexType curDimIndex = linearId % info.sizes[1];
    IndexType offset = curDimIndex * info.strides[1];
    linearId /= info.sizes[1];
    return offset + linearId * info.strides[0];
  }
};

// Specialization for Dim == 3
template <typename T, typename IndexType>
struct IndexToOffset<T, IndexType, 3> {
  static __host__ __device__ IndexType
  get(IndexType linearId, const TensorInfo<T, IndexType>& info) {
    IndexType curDimIndex = linearId % info.sizes[2];
    IndexType offset = curDimIndex * info.strides[2];
    linearId /= info.sizes[2];
    curDimIndex = linearId % info.sizes[1];
    offset += curDimIndex * info.strides[1];
    linearId /= info.sizes[1];
    return offset + linearId * info.strides[0];
  }
};

// --- inclusiveBinaryPrefixScan / exclusiveBinaryPrefixScan ---
// Prefix scan utilities

template <typename T>
__device__ inline void swapVars(T* t1, T* t2) {
  T tmp = *t1;
  *t1 = *t2;
  *t2 = tmp;
}

template <typename Comparator, typename K, typename V>
__device__ inline void bitonicSwap(K* kA,
                                   V* vA,
                                   bool* validA,
                                   K* kB,
                                   V* vB,
                                   bool* validB,
                                   bool dir,
                                   const Comparator& comp) {
  // Invalid entries always sort to the end
  bool swap = (comp(*kA, *kB) && *validA) || !*validB;
  if (swap == dir) {
    swapVars(kA, kB);
    swapVars(vA, vB);
    swapVars(validA, validB);
  }
}

template <int Power2SortSize,
          typename IndexType,
          typename Comparator,
          typename K,
          typename V>
__device__ inline void bitonicSort(K* keys,
                                   V* values,
                                   bool* valid,
                                   const Comparator& comp) {
#pragma unroll
  for (unsigned int size = 2; size < Power2SortSize; size *= 2) {
    bool flag = ((threadIdx.x & (size / 2)) != 0);

#pragma unroll
    for (unsigned int stride = size / 2; stride > 0; stride /= 2) {
      __syncthreads();

      unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
      bitonicSwap<Comparator, K, V>(&keys[pos],
                                    &values[pos],
                                    &valid[pos],
                                    &keys[pos + stride],
                                    &values[pos + stride],
                                    &valid[pos + stride],
                                    flag,
                                    comp);
    }
  }

#pragma unroll
  for (unsigned int stride = Power2SortSize / 2; stride > 0; stride /= 2) {
    __syncthreads();

    unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
    bitonicSwap<Comparator, K, V>(&keys[pos],
                                  &values[pos],
                                  &valid[pos],
                                  &keys[pos + stride],
                                  &values[pos + stride],
                                  &valid[pos + stride],
                                  false,
                                  comp);
  }

  __syncthreads();
}

template <int KeyDims,
          int ValueDims,
          int block_dim_x,
          int max_block_dim_y,
          typename K,
          typename V,
          typename Comparator,
          typename IndexType>
__global__ void __launch_bounds__(block_dim_x* max_block_dim_y)
    bitonicSortKVInPlace(TensorInfo<K, IndexType> keys,
                         IndexType keySlices,
                         IndexType keySliceSize,
                         IndexType keySliceStride,
                         TensorInfo<V, IndexType> values,
                         IndexType valueSliceStride,
                         Comparator comp) {
  const IndexType blockIndex = getLinearBlockId<IndexType>();
  const IndexType linearIndex = blockIndex * blockDim.y + threadIdx.y;

  if (blockIndex * blockDim.y >= keySlices) {
    return;
  }
  const bool row_valid = linearIndex < keySlices;

  constexpr int items_per_thread = 2;
  constexpr int Power2SortSize = block_dim_x * items_per_thread;

  __shared__ K blockSharedKeys[max_block_dim_y][Power2SortSize];
  __shared__ V blockSharedValues[max_block_dim_y][Power2SortSize];
  __shared__ bool blockSharedValid[max_block_dim_y][Power2SortSize];

  auto sharedKeys = blockSharedKeys[threadIdx.y];
  auto sharedValues = blockSharedValues[threadIdx.y];
  auto sharedValid = blockSharedValid[threadIdx.y];

  const IndexType keyStartOffset =
      IndexToOffset<K, IndexType, KeyDims>::get(linearIndex, keys);
  const IndexType valueStartOffset =
      IndexToOffset<V, IndexType, ValueDims>::get(linearIndex, values);

#pragma unroll
  for (int k = 0; k < items_per_thread; ++k) {
    auto idx = threadIdx.x + k * blockDim.x;
    bool valid = row_valid && idx < keySliceSize;

    sharedKeys[idx] =
        valid ? keys.data[idx * keySliceStride + keyStartOffset] : K{};
    sharedValues[idx] =
        valid ? values.data[idx * valueSliceStride + valueStartOffset] : V{};
    sharedValid[idx] = valid;
  }

  bitonicSort<Power2SortSize, IndexType>(
      sharedKeys, sharedValues, sharedValid, comp);

  if (!row_valid) {
    return;
  }

#pragma unroll
  for (int k = 0; k < items_per_thread; ++k) {
    auto idx = threadIdx.x + k * blockDim.x;
    if (idx < keySliceSize) {
      keys.data[idx * keySliceStride + keyStartOffset] = sharedKeys[idx];
      values.data[idx * valueSliceStride + valueStartOffset] =
          sharedValues[idx];
    }
  }
}

template <typename scalar_t, bool handleNaN = false>
struct GTOp {
  __device__ bool operator()(const scalar_t& lhs, const scalar_t& rhs) const {
    return (handleNaN && (lhs != lhs) && !(rhs != rhs)) ||
           (static_cast<scalar_t>(lhs) > static_cast<scalar_t>(rhs));
  }
};

template <typename scalar_t, bool handleNaN = false>
struct LTOp {
  __device__ bool operator()(const scalar_t& lhs, const scalar_t& rhs) const {
    return (handleNaN && !(lhs != lhs) && (rhs != rhs)) ||
           (static_cast<scalar_t>(lhs) < static_cast<scalar_t>(rhs));
  }
};

template <typename T, typename IndexType, int Dim>
void launch_bitonic_sort(TensorInfo<T, IndexType> keyInfo,
                         IndexType keySlices,
                         IndexType keySliceSize,
                         IndexType keySliceStride,
                         TensorInfo<int64_t, IndexType> valueInfo,
                         IndexType valueSliceStride,
                         bool largest,
                         gpuStream_t stream) {
  constexpr int sort_size = 32;
  constexpr int max_block_y = 16;
  constexpr int items_per_thread = 2;
  constexpr int block_x = sort_size / items_per_thread;

  const int block_y = std::min(
      static_cast<int>(max_block_y),
      static_cast<int>(std::max(static_cast<IndexType>(1), keySlices)));
  dim3 block(block_x, block_y);

  dim3 grid;
  const int grid_count = (keySlices + block_y - 1) / block_y;
  getGridFromTiles(grid_count, &grid);

  if (largest) {
    bitonicSortKVInPlace<Dim, Dim, block_x, max_block_y>
        <<<grid, block, 0, stream>>>(keyInfo,
                                     keySlices,
                                     keySliceSize,
                                     keySliceStride,
                                     valueInfo,
                                     valueSliceStride,
                                     GTOp<T, true>());
  } else {
    bitonicSortKVInPlace<Dim, Dim, block_x, max_block_y>
        <<<grid, block, 0, stream>>>(keyInfo,
                                     keySlices,
                                     keySliceSize,
                                     keySliceStride,
                                     valueInfo,
                                     valueSliceStride,
                                     LTOp<T, true>());
  }
}

// ============================================================================
// StridedRandomAccessor
// Required by CUB WarpLoad/WarpStore and BlockLoad/BlockStore for strided
// tensor access.
// ============================================================================

template <typename T, typename index_t = int64_t>
class ConstStridedRandomAccessor {
 public:
  using difference_type = index_t;
  using value_type = const T;
  using pointer = const T*;
  using reference = const T&;
  using iterator_category = std::random_access_iterator_tag;
  using PtrType = T*;
  using index_type = index_t;

  __host__ __device__ ConstStridedRandomAccessor(PtrType ptr, index_t stride)
      : ptr_{ptr}, stride_{stride} {}
  __host__ __device__ explicit ConstStridedRandomAccessor(PtrType ptr)
      : ptr_{ptr}, stride_{1} {}
  __host__ __device__ ConstStridedRandomAccessor()
      : ptr_{nullptr}, stride_{1} {}

  __host__ __device__ reference operator*() const { return *ptr_; }
  __host__ __device__ const T* operator->() const {
    return reinterpret_cast<const T*>(ptr_);
  }
  __host__ __device__ reference operator[](index_t idx) const {
    return ptr_[idx * stride_];
  }

  __host__ __device__ ConstStridedRandomAccessor& operator++() {
    ptr_ += stride_;
    return *this;
  }
  __host__ __device__ ConstStridedRandomAccessor operator++(int) {
    ConstStridedRandomAccessor copy(*this);
    ++*this;
    return copy;
  }
  __host__ __device__ ConstStridedRandomAccessor& operator--() {
    ptr_ -= stride_;
    return *this;
  }
  __host__ __device__ ConstStridedRandomAccessor operator--(int) {
    ConstStridedRandomAccessor copy(*this);
    --*this;
    return copy;
  }
  __host__ __device__ ConstStridedRandomAccessor& operator+=(index_t offset) {
    ptr_ += offset * stride_;
    return *this;
  }
  __host__ __device__ ConstStridedRandomAccessor
  operator+(index_t offset) const {
    return ConstStridedRandomAccessor(ptr_ + offset * stride_, stride_);
  }
  __host__ __device__ friend ConstStridedRandomAccessor operator+(
      index_t offset, const ConstStridedRandomAccessor& accessor) {
    return accessor + offset;
  }
  __host__ __device__ ConstStridedRandomAccessor& operator-=(index_t offset) {
    ptr_ -= offset * stride_;
    return *this;
  }
  __host__ __device__ ConstStridedRandomAccessor
  operator-(index_t offset) const {
    return ConstStridedRandomAccessor(ptr_ - offset * stride_, stride_);
  }
  __host__ __device__ difference_type
  operator-(const ConstStridedRandomAccessor& other) const {
    return (ptr_ - other.ptr_) / stride_;
  }
  __host__ __device__ bool operator==(
      const ConstStridedRandomAccessor& other) const {
    return (ptr_ == other.ptr_) && (stride_ == other.stride_);
  }
  __host__ __device__ bool operator!=(
      const ConstStridedRandomAccessor& other) const {
    return !(*this == other);
  }
  __host__ __device__ bool operator<(
      const ConstStridedRandomAccessor& other) const {
    return ptr_ < other.ptr_;
  }
  __host__ __device__ bool operator<=(
      const ConstStridedRandomAccessor& other) const {
    return (*this < other) || (*this == other);
  }
  __host__ __device__ bool operator>(
      const ConstStridedRandomAccessor& other) const {
    return !(*this <= other);
  }
  __host__ __device__ bool operator>=(
      const ConstStridedRandomAccessor& other) const {
    return !(*this < other);
  }

 protected:
  PtrType ptr_;
  index_t stride_;
};

template <typename T, typename index_t = int64_t>
class StridedRandomAccessor : public ConstStridedRandomAccessor<T, index_t> {
 public:
  using difference_type = index_t;
  using value_type = T;
  using pointer = T*;
  using reference = T&;
  using BaseType = ConstStridedRandomAccessor<T, index_t>;
  using PtrType = T*;

  __host__ __device__ StridedRandomAccessor(PtrType ptr, index_t stride)
      : BaseType(ptr, stride) {}
  __host__ __device__ explicit StridedRandomAccessor(PtrType ptr)
      : BaseType(ptr) {}
  __host__ __device__ StridedRandomAccessor() : BaseType() {}

  __host__ __device__ reference operator*() const { return *this->ptr_; }
  __host__ __device__ T* operator->() const {
    return reinterpret_cast<T*>(this->ptr_);
  }
  __host__ __device__ reference operator[](index_t idx) const {
    return this->ptr_[idx * this->stride_];
  }

  __host__ __device__ StridedRandomAccessor& operator++() {
    this->ptr_ += this->stride_;
    return *this;
  }
  __host__ __device__ StridedRandomAccessor operator++(int) {
    StridedRandomAccessor copy(*this);
    ++*this;
    return copy;
  }
  __host__ __device__ StridedRandomAccessor& operator--() {
    this->ptr_ -= this->stride_;
    return *this;
  }
  __host__ __device__ StridedRandomAccessor operator--(int) {
    StridedRandomAccessor copy(*this);
    --*this;
    return copy;
  }
  __host__ __device__ StridedRandomAccessor& operator+=(index_t offset) {
    this->ptr_ += offset * this->stride_;
    return *this;
  }
  __host__ __device__ StridedRandomAccessor operator+(index_t offset) const {
    return StridedRandomAccessor(this->ptr_ + offset * this->stride_,
                                 this->stride_);
  }
  __host__ __device__ friend StridedRandomAccessor operator+(
      index_t offset, const StridedRandomAccessor& accessor) {
    return accessor + offset;
  }
  __host__ __device__ StridedRandomAccessor& operator-=(index_t offset) {
    this->ptr_ -= offset * this->stride_;
    return *this;
  }
  __host__ __device__ StridedRandomAccessor operator-(index_t offset) const {
    return StridedRandomAccessor(this->ptr_ - offset * this->stride_,
                                 this->stride_);
  }
  __host__ __device__ difference_type operator-(const BaseType& other) const {
    return (static_cast<const BaseType&>(*this) - other);
  }
};

// ============================================================================
// CubKeyType mapping - maps Paddle types to CUB-compatible CUDA types
// For BlockRadixSort, CUB needs __half / __nv_bfloat16 instead of
// phi::float16 / phi::bfloat16.
// ============================================================================
template <typename T>
struct CubKeyType {
  using type = T;
};

template <>
struct CubKeyType<phi::dtype::float16> {
  using type = __half;
};

template <>
struct CubKeyType<phi::dtype::bfloat16> {
#if defined(__HIPCC__)
  using type = hip_bfloat16;
#else
  using type = __nv_bfloat16;
#endif
};

// ============================================================================
// Utility functions
// ============================================================================

inline int64_t nextHighestPowerOf2(int64_t n) {
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n |= n >> 32;
  n++;
  return n;
}

template <typename T>
static int minimum_grid_for_occupancy(T kernel, int max_block_size) {
  int minGridSize = 0;
  int blockSize = 0;
  cudaOccupancyMaxPotentialBlockSize(
      &minGridSize, &blockSize, kernel, /*dynamicSMemSize=*/0, max_block_size);
  return minGridSize;
}

template <typename T>
constexpr bool type_has_nan() {
  if constexpr (std::numeric_limits<T>::is_specialized) {
    return std::numeric_limits<T>::has_quiet_NaN;
  } else if constexpr (std::is_same_v<T, phi::dtype::float16> ||  // NOLINT
                       std::is_same_v<T, phi::dtype::bfloat16>) {
    return true;
  } else {
    return false;
  }
}

// ============================================================================
// warpMergeSortKVInPlace kernel
// For sort sizes 33..128, uses CUB WarpMergeSort (one warp per slice,
// multiple slices per block via blockDim.y).
// ============================================================================

template <int KeyDims,
          int ValueDims,
          int sort_size,
          int max_block_dim_y,
          typename K,
          typename V,
          typename Comparator,
          typename IndexType>
__global__ void __launch_bounds__(32 * max_block_dim_y)
    warpMergeSortKVInPlace(TensorInfo<K, IndexType> keys,
                           IndexType keySlices,
                           IndexType keySliceSize,
                           IndexType keySliceStride,
                           TensorInfo<V, IndexType> values,
                           IndexType valueSliceStride,
                           Comparator comp,
                           K invalid_key) {
  const IndexType blockIndex = getLinearBlockId<IndexType>();
  const IndexType linearIndex = blockIndex * blockDim.y + threadIdx.y;

  if (linearIndex >= keySlices) {
    return;
  }

  const IndexType keyStartOffset =
      IndexToOffset<K, IndexType, KeyDims>::get(linearIndex, keys);
  const IndexType valueStartOffset =
      IndexToOffset<V, IndexType, ValueDims>::get(linearIndex, values);

  K* keys_slice = &keys.data[keyStartOffset];
  V* values_slice = &values.data[valueStartOffset];

  StridedRandomAccessor<K, IndexType> keys_iter(keys_slice, keySliceStride);
  StridedRandomAccessor<V, IndexType> values_iter(values_slice,
                                                  valueSliceStride);

  constexpr int warp_size = 32;
  constexpr int kItemsPerThread = sort_size / warp_size;
  static_assert(kItemsPerThread * warp_size == sort_size,
                "sort_size must be a multiple of warp_size (32)");

  using LoadKeys = cub::WarpLoad<K, kItemsPerThread, cub::WARP_LOAD_TRANSPOSE>;
  using LoadValues =
      cub::WarpLoad<V, kItemsPerThread, cub::WARP_LOAD_TRANSPOSE>;
  using Sort = cub::WarpMergeSort<K, kItemsPerThread, warp_size, V>;
  using StoreKeys =
      cub::WarpStore<K, kItemsPerThread, cub::WARP_STORE_TRANSPOSE>;
  using StoreValues =
      cub::WarpStore<V, kItemsPerThread, cub::WARP_STORE_TRANSPOSE>;

  __shared__ union {
    typename LoadKeys::TempStorage load_keys;
    typename LoadValues::TempStorage load_values;
    typename Sort::TempStorage sort;
    typename StoreKeys::TempStorage store_keys;
    typename StoreValues::TempStorage store_values;
  } tmp_storage[max_block_dim_y];

  auto& warp_storage = tmp_storage[threadIdx.y];

  K local_keys[kItemsPerThread];
  V local_values[kItemsPerThread];

  const auto invalid_value = V{};
  LoadKeys(warp_storage.load_keys)
      .Load(keys_iter, local_keys, keySliceSize, invalid_key);
#if !defined(__HIPCC__)
  __syncwarp();
#endif
  LoadValues(warp_storage.load_values)
      .Load(values_iter, local_values, keySliceSize, invalid_value);
#if !defined(__HIPCC__)
  __syncwarp();
#endif

  Sort(warp_storage.sort)
      .StableSort(local_keys, local_values, comp, keySliceSize, invalid_key);
#if !defined(__HIPCC__)
  __syncwarp();
#endif

  StoreKeys(warp_storage.store_keys).Store(keys_iter, local_keys, keySliceSize);
#if !defined(__HIPCC__)
  __syncwarp();
#endif
  StoreValues(warp_storage.store_values)
      .Store(values_iter, local_values, keySliceSize);
}

// ============================================================================
// radixSortKVInPlace kernel
// For sort sizes 129..4096, uses CUB BlockRadixSort (one block per slice).
// ============================================================================

template <int KeyDims,
          int ValueDims,
          int block_size,
          int kItemsPerThread,
          typename K,
          typename V,
          typename IndexType>
__global__ void __launch_bounds__(block_size)
    radixSortKVInPlace(TensorInfo<K, IndexType> keys,
                       IndexType keySlices,
                       IndexType keySliceSize,
                       IndexType keySliceStride,
                       TensorInfo<V, IndexType> values,
                       IndexType valueSliceStride,
                       bool descending) {
  static_assert(block_size > 0, "");

  const IndexType linearIndex = getLinearBlockId<IndexType>();
  if (linearIndex >= keySlices) {
    return;
  }

  const IndexType keyStartOffset =
      IndexToOffset<K, IndexType, KeyDims>::get(linearIndex, keys);
  const IndexType valueStartOffset =
      IndexToOffset<V, IndexType, ValueDims>::get(linearIndex, values);

  K* keys_slice = &keys.data[keyStartOffset];
  V* values_slice = &values.data[valueStartOffset];

  StridedRandomAccessor<K, IndexType> keys_iter(keys_slice, keySliceStride);
  StridedRandomAccessor<V, IndexType> values_iter(values_slice,
                                                  valueSliceStride);

  using key_t = typename CubKeyType<K>::type;
  using LoadKeys =
      cub::BlockLoad<K, block_size, kItemsPerThread, cub::BLOCK_LOAD_TRANSPOSE>;
  using LoadValues =
      cub::BlockLoad<V, block_size, kItemsPerThread, cub::BLOCK_LOAD_TRANSPOSE>;
  using Sort = cub::BlockRadixSort<key_t, block_size, kItemsPerThread, V>;
  using StoreKeys = cub::
      BlockStore<K, block_size, kItemsPerThread, cub::BLOCK_STORE_TRANSPOSE>;
  using StoreValues = cub::
      BlockStore<V, block_size, kItemsPerThread, cub::BLOCK_STORE_TRANSPOSE>;

  __shared__ union {
    typename LoadKeys::TempStorage load_keys;
    typename LoadValues::TempStorage load_values;
    typename Sort::TempStorage sort;
    typename StoreKeys::TempStorage store_keys;
    typename StoreValues::TempStorage store_values;
  } tmp_storage;

  // Compute invalid key: always sorts higher than any valid key
  const K invalid_key = [descending] {
    using radix_t = typename cub::Traits<key_t>::UnsignedBits;
    union {
      K key;
      radix_t radix;
    } tmp;
    tmp.radix = descending ? cub::Traits<key_t>::LOWEST_KEY
                           : cub::Traits<key_t>::MAX_KEY;
    return tmp.key;
  }();
  const V invalid_value = static_cast<V>(0);

  K local_keys[kItemsPerThread];
  V local_values[kItemsPerThread];

  LoadKeys(tmp_storage.load_keys)
      .Load(keys_iter, local_keys, keySliceSize, invalid_key);
  __syncthreads();
  LoadValues(tmp_storage.load_values)
      .Load(values_iter, local_values, keySliceSize, invalid_value);
  __syncthreads();

  if (descending) {
    Sort(tmp_storage.sort)
        .SortDescending(reinterpret_cast<key_t(&)[kItemsPerThread]>(local_keys),
                        local_values);
  } else {
    Sort(tmp_storage.sort)
        .Sort(reinterpret_cast<key_t(&)[kItemsPerThread]>(local_keys),
              local_values);
  }
  __syncthreads();

  StoreKeys(tmp_storage.store_keys).Store(keys_iter, local_keys, keySliceSize);
  __syncthreads();
  StoreValues(tmp_storage.store_values)
      .Store(values_iter, local_values, keySliceSize);
}

// ============================================================================
// launch_warp_merge_sort - wrapper for CUB WarpMergeSort<128>
// ============================================================================

template <typename T, typename IndexType, int Dim>
void launch_warp_merge_sort(TensorInfo<T, IndexType> keyInfo,
                            IndexType keySlices,
                            IndexType keySliceSize,
                            IndexType keySliceStride,
                            TensorInfo<int64_t, IndexType> valueInfo,
                            IndexType valueSliceStride,
                            bool largest,
                            gpuStream_t stream) {
  constexpr int sort_size = 128;
  constexpr int max_block_dim_y = 16;
  constexpr int warp_size = 32;

  // Scale batch size down if the grid would be too small
  const auto min_grid =
      minimum_grid_for_occupancy(warpMergeSortKVInPlace<Dim,
                                                        Dim,
                                                        sort_size,
                                                        max_block_dim_y,
                                                        T,
                                                        int64_t,
                                                        LTOp<T, true>,
                                                        IndexType>,
                                 warp_size * max_block_dim_y);
  const auto max_batch =
      std::max(IndexType{1}, keySlices / (IndexType)min_grid);
  const int block_y = std::min((IndexType)max_block_dim_y, max_batch);
  dim3 block(warp_size, block_y);

  dim3 grid;
  const int grid_count = (keySlices + block_y - 1) / block_y;
  getGridFromTiles(grid_count, &grid);

  if (largest) {
    // Use numeric limits for invalid_key: lower_bound for descending
    const T invalid_key = std::numeric_limits<T>::lowest();
    warpMergeSortKVInPlace<Dim, Dim, sort_size, max_block_dim_y>
        <<<grid, block, 0, stream>>>(keyInfo,
                                     keySlices,
                                     keySliceSize,
                                     keySliceStride,
                                     valueInfo,
                                     valueSliceStride,
                                     GTOp<T, true>(),
                                     invalid_key);
  } else {
    // For ascending: NAN sorts after inf, otherwise use upper_bound
    const T invalid_key = [] {
      if constexpr (type_has_nan<T>()) {
        return T(NAN);
      }
      return std::numeric_limits<T>::max();
    }();
    warpMergeSortKVInPlace<Dim, Dim, sort_size, max_block_dim_y>
        <<<grid, block, 0, stream>>>(keyInfo,
                                     keySlices,
                                     keySliceSize,
                                     keySliceStride,
                                     valueInfo,
                                     valueSliceStride,
                                     LTOp<T, true>(),
                                     invalid_key);
  }
}

// ============================================================================
// launch_medium_radix_sort - wrapper for CUB BlockRadixSort
// ============================================================================

template <int Dim,
          int sort_size,
          int items_per_thread,
          typename K,
          typename IndexType>
void fixed_size_radix_sort(TensorInfo<K, IndexType> keyInfo,
                           IndexType keySlices,
                           IndexType keySliceSize,
                           IndexType keySliceStride,
                           TensorInfo<int64_t, IndexType> valueInfo,
                           IndexType valueSliceStride,
                           bool descending,
                           gpuStream_t stream) {
  static_assert(sort_size % items_per_thread == 0, "");
  constexpr int block = sort_size / items_per_thread;
  dim3 grid;
  getGridFromTiles(keySlices, &grid);

  radixSortKVInPlace<Dim, Dim, block, items_per_thread>
      <<<grid, block, 0, stream>>>(keyInfo,
                                   keySlices,
                                   keySliceSize,
                                   keySliceStride,
                                   valueInfo,
                                   valueSliceStride,
                                   descending);
}

template <typename T, typename IndexType, int Dim>
void launch_medium_radix_sort(TensorInfo<T, IndexType> keyInfo,
                              IndexType keySlices,
                              IndexType keySliceSize,
                              IndexType keySliceStride,
                              TensorInfo<int64_t, IndexType> valueInfo,
                              IndexType valueSliceStride,
                              bool descending,
                              gpuStream_t stream) {
  int64_t ceilPowerOf2 = nextHighestPowerOf2(keySliceSize);
  constexpr int default_ipt = 32;

#define HANDLE_RADIX_CASE(SIZE, IPT)                      \
  fixed_size_radix_sort<Dim, SIZE, IPT>(keyInfo,          \
                                        keySlices,        \
                                        keySliceSize,     \
                                        keySliceStride,   \
                                        valueInfo,        \
                                        valueSliceStride, \
                                        descending,       \
                                        stream)

  switch (ceilPowerOf2) {
    case 4096:
      HANDLE_RADIX_CASE(4096, default_ipt);
      break;
    case 2048:
      HANDLE_RADIX_CASE(2048, default_ipt);
      break;
    case 1024:
    case 512:
    case 256:
      HANDLE_RADIX_CASE(1024, default_ipt);
      break;
    // sizes <= 128 should have been handled by WarpMergeSort
    default:
      break;
  }
#undef HANDLE_RADIX_CASE
}

template <typename T, bool KillWARDependency, class BinaryFunction>
__device__ void inclusiveBinaryPrefixScan(T* smem,
                                          bool in,
                                          T* out,
                                          BinaryFunction binop) {
  T vote = TOPK_WARP_BALLOT(in);
  T index = __popc(getLaneMaskLe() & vote);
  T carry = __popc(vote);

  int warp = threadIdx.x / TOPK_WARP_SIZE;

  if (getLaneId() == 0) {
    smem[warp] = carry;
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    int current = 0;
    for (int i = 0;
         i < topk_ceil_div(static_cast<int>(blockDim.x), TOPK_WARP_SIZE);
         ++i) {
      T v = smem[i];
      smem[i] = binop(smem[i], current);
      current = binop(current, v);
    }
  }

  __syncthreads();

  if (warp >= 1) {
    index = binop(index, smem[warp - 1]);
  }

  *out = index;

  if (KillWARDependency) {
    __syncthreads();
  }
}

template <typename T, bool KillWARDependency, class BinaryFunction>
__device__ void exclusiveBinaryPrefixScan(
    T* smem, bool in, T* out, T* carry, BinaryFunction binop) {
  inclusiveBinaryPrefixScan<T, false, BinaryFunction>(smem, in, out, binop);
  *out -= static_cast<T>(in);
  *carry =
      smem[topk_ceil_div(static_cast<int>(blockDim.x), TOPK_WARP_SIZE) - 1];
  if (KillWARDependency) {
    __syncthreads();
  }
}

// --- AddOp ---
template <typename T>
struct AddOp {
  __device__ __forceinline__ T operator()(T const& lhs, T const& rhs) {
    return (lhs + rhs);
  }
};

// ============================================================================
// SortingRadixSelect.cuh ported content
// ============================================================================

namespace radix_select {

// Over what radix we are selecting values (single-block variant)
constexpr int RADIX_BITS = 2;
constexpr int RADIX_SIZE = 4;  // 2 ^ RADIX_BITS
constexpr int RADIX_MASK = (RADIX_SIZE - 1);

// CountType is separate from IndexType — counts always fit in int32
// because indices are limited to integer fp precision.
template <typename T,
          typename RadixType,
          typename IndexType,
          typename CountType,
          int RadixSize,
          int RadixBits>
__device__ void countRadixUsingMask(const T* data,
                                    CountType counts[RadixSize],
                                    CountType* smem,
                                    RadixType desired,
                                    RadixType desiredMask,
                                    int radixDigitPos,
                                    IndexType sliceSize,
                                    IndexType withinSliceStride) {
#pragma unroll
  for (int i = 0; i < RadixSize; ++i) {
    counts[i] = 0;
  }
  if (threadIdx.x < RadixSize) {
    smem[threadIdx.x] = 0;
  }
  __syncthreads();

  // Must be called outside of loop to ensure all threads participate.
  // This creates a dynamic mask of which threads will enter the loop.
  // When sliceSize < blockDim.x, only threads with threadIdx.x < sliceSize
  // will enter the loop body, so we need a mask to avoid deadlock in
  // __ballot_sync.
#if !defined(__HIPCC__)
  unsigned mask = TOPK_WARP_BALLOT(threadIdx.x < sliceSize);
#endif
  for (IndexType i = threadIdx.x; i < sliceSize;) {
    RadixType val =
        TopKTypeConfig<T>::convert(doLdg(&data[i * withinSliceStride]));
    bool hasVal = ((val & desiredMask) == desired);
    RadixType digitInRadix =
        Bitfield<RadixType>::getBitfield(val, radixDigitPos, RadixBits);
#pragma unroll
    for (uint32_t j = 0; j < RadixSize; ++j) {
      bool vote = hasVal && (digitInRadix == j);
#if defined(__HIPCC__)
      counts[j] += __popcll(TOPK_WARP_BALLOT(vote));
#else
      counts[j] += __popc(TOPK_WARP_BALLOT_MASK(vote, mask));
#endif
    }
    i += blockDim.x;
#if !defined(__HIPCC__)
    mask = TOPK_WARP_BALLOT_MASK(i < sliceSize, mask);
#endif
  }

  if (getLaneId() == 0) {
#pragma unroll
    for (uint32_t i = 0; i < RadixSize; ++i) {
      atomicAdd(&smem[i], counts[i]);
    }
  }
  __syncthreads();

#pragma unroll
  for (uint32_t i = 0; i < RadixSize; ++i) {
    counts[i] = smem[i];
  }
  __syncthreads();
}

template <typename T, typename RadixType, typename IndexType>
__device__ T findPattern(const T* data,
                         T* smem,
                         IndexType sliceSize,
                         IndexType withinSliceStride,
                         RadixType desired,
                         RadixType desiredMask) {
  if (threadIdx.x < 2) {
    smem[threadIdx.x] = static_cast<T>(0);
  }
  __syncthreads();

  IndexType numIterations = topk_round_up(sliceSize, (IndexType)blockDim.x);
  for (IndexType i = threadIdx.x; i < numIterations; i += blockDim.x) {
    bool inRange = (i < sliceSize);
    T v = inRange ? doLdg(&data[i * withinSliceStride]) : static_cast<T>(0);
    if (inRange && ((TopKTypeConfig<T>::convert(v) & desiredMask) == desired)) {
      smem[0] = static_cast<T>(1);
      smem[1] = v;
    }
    __syncthreads();
    T found = smem[0];
    T val = smem[1];
    __syncthreads();
    if (found != static_cast<T>(0)) {
      return val;
    }
  }
  // should not get here
  assert(false);
  return static_cast<T>(0);
}

template <typename T, typename RadixType, typename IndexType>
__device__ void radixSelect(const T* data,
                            IndexType k,
                            bool largest,
                            IndexType sliceSize,
                            IndexType withinSliceStride,
                            int* smem,
                            T* topKValue) {
  // Indices are limited to integer fp precision, so counts can fit in
  // int32, regardless of IndexType
  int counts[RADIX_SIZE];
  RadixType desired = 0;
  RadixType desiredMask = 0;

  IndexType kToFind = k;

#pragma unroll
  for (int digitPos = sizeof(T) * 8 - RADIX_BITS; digitPos >= 0;
       digitPos -= RADIX_BITS) {
    countRadixUsingMask<T, RadixType, IndexType, int, RADIX_SIZE, RADIX_BITS>(
        data,
        counts,
        smem,
        desired,
        desiredMask,
        digitPos,
        sliceSize,
        withinSliceStride);

    auto found_unique = [&](int i, int count) -> bool {
      if (count == 1 && kToFind == 1) {
        desired =
            Bitfield<RadixType>::setBitfield(desired, i, digitPos, RADIX_BITS);
        desiredMask = Bitfield<RadixType>::setBitfield(
            desiredMask, RADIX_MASK, digitPos, RADIX_BITS);
        *topKValue =
            findPattern<T, RadixType, IndexType>(data,
                                                 reinterpret_cast<T*>(smem),
                                                 sliceSize,
                                                 withinSliceStride,
                                                 desired,
                                                 desiredMask);
        return true;
      }
      return false;
    };
    auto found_non_unique = [&](int i, int count) -> bool {
      if (count >= kToFind) {
        desired =
            Bitfield<RadixType>::setBitfield(desired, i, digitPos, RADIX_BITS);
        desiredMask = Bitfield<RadixType>::setBitfield(
            desiredMask, RADIX_MASK, digitPos, RADIX_BITS);
        return true;
      }
      kToFind -= count;
      return false;
    };

    if (largest) {
#pragma unroll
      for (int i = RADIX_SIZE - 1; i >= 0; --i) {
        int count = counts[i];
        if (found_unique(i, count)) return;
        if (found_non_unique(i, count)) break;
      }
    } else {
#pragma unroll
      for (int i = 0; i < RADIX_SIZE; ++i) {
        int count = counts[i];
        if (found_unique(i, count)) return;
        if (found_non_unique(i, count)) break;
      }
    }
  }
  *topKValue = TopKTypeConfig<T>::deconvert(desired);
}

}  // namespace radix_select

// ============================================================================
// CUDA_KERNEL_LOOP_TYPE macro
// ============================================================================
#define TOPK_CUDA_KERNEL_LOOP_TYPE(i, n, index_type)                  \
  for (index_type i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

// ============================================================================
// CUB_SUPPORTS_SCAN_BY_KEY check
// CUB >= 1.15 supports DeviceScan::InclusiveSumByKey
// ============================================================================
#ifndef __HIPCC__
// CUDA path: check CUB version
#if defined(CUB_VERSION) && CUB_VERSION >= 101500
#define TOPK_CUB_SUPPORTS_SCAN_BY_KEY() 1
#else
// Try to detect based on CUDA version (CUDA 11.6+ bundles CUB >= 1.15)
#if CUDART_VERSION >= 11060
#define TOPK_CUB_SUPPORTS_SCAN_BY_KEY() 1
#else
#define TOPK_CUB_SUPPORTS_SCAN_BY_KEY() 0
#endif
#endif
#else
// HIP/ROCm path
#define TOPK_CUB_SUPPORTS_SCAN_BY_KEY() 0
#endif

}  // namespace topk_detail

// ============================================================================
// Main TopK implementation
// ============================================================================

namespace topk_impl {

using namespace topk_detail;  // NOLINT

// getTensorInfo: builds TensorInfo from DenseTensor
template <typename T, typename IndexType>
TensorInfo<T, IndexType> getTensorInfo(const phi::DenseTensor& tensor) {
  TensorInfo<T, IndexType> info;
  info.data = reinterpret_cast<T*>(const_cast<void*>(tensor.data()));
  info.dims = tensor.dims().size();
  for (int i = 0; i < info.dims; i++) {
    info.sizes[i] = tensor.dims()[i];
    info.strides[i] = tensor.strides()[i];
  }
  return info;
}

// SegmentOffsetIter for sorted output - must be at namespace scope for CUDA
struct SegmentOffsetIter {
  int64_t k;
  __host__ __device__ __forceinline__ int64_t operator()(int64_t idx) const {
    return idx * k;
  }
};

template <typename T, typename Context>
void sortKeyValueInplace(const Context& dev_ctx,
                         phi::DenseTensor* out,
                         phi::DenseTensor* indices,
                         int axis,
                         bool largest) {
  const auto& out_dims = out->dims();
  int dim = axis;
  int64_t sliceSize = out_dims[dim];
  int64_t numSlices = out->numel() / sliceSize;
  auto stream = dev_ctx.stream();

  if (sliceSize <= 1) return;

  auto keyInfo = getTensorInfo<T, uint32_t>(*out);
  auto valueInfo = getTensorInfo<int64_t, uint32_t>(*indices);

  auto strideKey = keyInfo.strides[dim];
  keyInfo.sizes[dim] = 1;
  int collapseKeyDim = keyInfo.collapseDims(dim);
  keyInfo.strides[collapseKeyDim] = strideKey;

  auto strideValue = valueInfo.strides[dim];
  valueInfo.sizes[dim] = 1;
  int collapseValueDim = valueInfo.collapseDims(dim);
  valueInfo.strides[collapseValueDim] = strideValue;

  // Three-tier sort dispatch:
  //   1. sliceSize <= 32:  Bitonic Sort (unstable, fast, no extra memory)
  //   2. sliceSize <= 128: WarpMergeSort (CUB, one slice per warp)
  //   3. sliceSize <= 4096: BlockRadixSort (CUB, one slice per block)
  // Dispatch on the actual number of collapsed dims (keyInfo.dims),
  // NOT on collapseKeyDim (the remapped excluded-dim index).
  // When the excluded dim is in the middle (e.g. dim=1 of a 3-D tensor),
  // collapseKeyDim==1 but keyInfo.dims==3; using DIM=1 would make
  // IndexToOffset ignore the trailing dimensions, producing wrong offsets.

#define TOPK_SORT_DIM_DISPATCH(LAUNCH_FUNC) \
  if (keyInfo.dims == 1) {                  \
    LAUNCH_FUNC(1);                         \
  } else if (keyInfo.dims == 2) {           \
    LAUNCH_FUNC(2);                         \
  } else if (keyInfo.dims == 3) {           \
    LAUNCH_FUNC(3);                         \
  } else {                                  \
    LAUNCH_FUNC(-1);                        \
  }

  if (sliceSize <= 32) {
    // Bitonic sort (unstable)
#define LAUNCH_BITONIC(DIM)                          \
  launch_bitonic_sort<T, uint32_t, DIM>(keyInfo,     \
                                        numSlices,   \
                                        sliceSize,   \
                                        strideKey,   \
                                        valueInfo,   \
                                        strideValue, \
                                        largest,     \
                                        stream)
    TOPK_SORT_DIM_DISPATCH(LAUNCH_BITONIC);
#undef LAUNCH_BITONIC
  } else if (sliceSize <= 128) {
    // WarpMergeSort (stable, uses CUB WarpMergeSort)
#define LAUNCH_WARP(DIM)                                \
  launch_warp_merge_sort<T, uint32_t, DIM>(keyInfo,     \
                                           numSlices,   \
                                           sliceSize,   \
                                           strideKey,   \
                                           valueInfo,   \
                                           strideValue, \
                                           largest,     \
                                           stream)
    TOPK_SORT_DIM_DISPATCH(LAUNCH_WARP);
#undef LAUNCH_WARP
  } else {
    // BlockRadixSort (for sizes up to 4096)
    bool descending = largest;
#define LAUNCH_RADIX(DIM)                                 \
  launch_medium_radix_sort<T, uint32_t, DIM>(keyInfo,     \
                                             numSlices,   \
                                             sliceSize,   \
                                             strideKey,   \
                                             valueInfo,   \
                                             strideValue, \
                                             descending,  \
                                             stream)
    TOPK_SORT_DIM_DISPATCH(LAUNCH_RADIX);
#undef LAUNCH_RADIX
  }

#undef TOPK_SORT_DIM_DISPATCH
}

namespace sbtopk {  // single_block_topk

template <typename T, typename IndexType, int Dim, bool WithKthValues>
__global__ void __launch_bounds__(1024)
    gatherTopK(TensorInfo<const T, IndexType> input,
               IndexType inputSliceSize,
               IndexType outputSliceSize,  // aka `k`
               bool largest,
               IndexType numInputSlices,
               IndexType inputWithinSliceStride,
               TensorInfo<T, IndexType> topK,
               IndexType topKWithinSliceStride,
               TensorInfo<int64_t, IndexType> indices,
               IndexType indicesWithinSliceStride,
               T* kthValues) {
  // Indices are limited to integer fp precision, so counts can fit in
  // int32, regardless of IndexType
#if defined(__HIPCC__)
  __shared__ int smem[64];
#else
  __shared__ int smem[32];  // one per each warp, up to warp limit
#endif
  IndexType slice = getLinearBlockId<IndexType>();
  if (slice >= numInputSlices) {
    return;
  }

  // Find the start offset for our slice
  IndexType sliceStartIndex =
      IndexToOffset<const T, IndexType, Dim>::get(slice, input);
  IndexType topKSliceStartIndex =
      IndexToOffset<T, IndexType, Dim>::get(slice, topK);
  IndexType indicesSliceStartIndex =
      IndexToOffset<int64_t, IndexType, Dim>::get(slice, indices);

  const T* inputSliceStart = &input.data[sliceStartIndex];
  T* topKSliceStart = &topK.data[topKSliceStartIndex];
  int64_t* indicesSliceStart = &indices.data[indicesSliceStartIndex];

  // Find the k-th highest element in our input
  T topKValue;
  if (WithKthValues) {
    topKValue = kthValues[slice];
  } else {
    topKValue = static_cast<T>(0);
    radix_select::radixSelect<T,
                              typename TopKTypeConfig<T>::RadixType,
                              IndexType>(inputSliceStart,
                                         outputSliceSize,
                                         largest,
                                         inputSliceSize,
                                         inputWithinSliceStride,
                                         smem,
                                         &topKValue);
  }
  const auto topKConverted = TopKTypeConfig<T>::convert(topKValue);

  IndexType numIterations =
      topk_round_up(inputSliceSize, (IndexType)blockDim.x);
  IndexType writeIndexStart = 0;

  for (IndexType i = threadIdx.x; i < numIterations; i += blockDim.x) {
    bool inRange = (i < inputSliceSize);
    T v = inRange ? doLdg(&inputSliceStart[i * inputWithinSliceStride])
                  : static_cast<T>(0);
    const auto convertedV = TopKTypeConfig<T>::convert(v);
    bool hasTopK;
    if (largest) {
      hasTopK = inRange && (convertedV > topKConverted);
    } else {
      hasTopK = inRange && (convertedV < topKConverted);
    }

    int index;
    int carry;
    exclusiveBinaryPrefixScan<int, true>(
        smem, hasTopK, &index, &carry, AddOp<int>());

    if (hasTopK) {
      int writeIndex = writeIndexStart + index;
      assert(writeIndex < outputSliceSize);
      IndexType topKOffset = writeIndex * topKWithinSliceStride;
      IndexType indexOffset = writeIndex * indicesWithinSliceStride;
      topKSliceStart[topKOffset] = v;
      indicesSliceStart[indexOffset] = i;
    }
    writeIndexStart += carry;
  }

  // Fill in the rest with actual == top-K values.
  assert(outputSliceSize >= writeIndexStart);
  IndexType topKRemaining = (outputSliceSize - writeIndexStart);

  for (IndexType i = threadIdx.x; i < numIterations; i += blockDim.x) {
    bool inRange = (i < inputSliceSize);
    T v = inRange ? doLdg(&inputSliceStart[i * inputWithinSliceStride])
                  : static_cast<T>(0);
    const auto convertedV = TopKTypeConfig<T>::convert(v);
    bool hasTopK = inRange && (convertedV == topKConverted);

    int index;
    int carry;
    exclusiveBinaryPrefixScan<int, true>(
        smem, hasTopK, &index, &carry, AddOp<int>());

    if (hasTopK && index < topKRemaining) {
      int writeIndex = writeIndexStart + index;
      assert(writeIndex < outputSliceSize);
      IndexType topKOffset = writeIndex * topKWithinSliceStride;
      IndexType indexOffset = writeIndex * indicesWithinSliceStride;
      topKSliceStart[topKOffset] = v;
      indicesSliceStart[indexOffset] = i;
    }

    if (carry >= topKRemaining) {
      break;
    }
    topKRemaining -= carry;
    writeIndexStart += carry;
  }
}

template <typename T, typename IndexType, int Dim>
void launch(TensorInfo<const T, IndexType> input,
            IndexType inputSliceSize,
            IndexType outputSliceSize,
            bool largest,
            IndexType numInputSlices,
            IndexType inputWithinSliceStride,
            TensorInfo<T, IndexType> topK,
            IndexType topKWithinSliceStride,
            TensorInfo<int64_t, IndexType> indices,
            IndexType indicesWithinSliceStride,
            gpuStream_t stream) {
  dim3 grid;
  bool ok = getGridFromTiles(numInputSlices, &grid);
  assert(ok);
  (void)ok;
  int warp_size = TOPK_WARP_SIZE;
  dim3 block(
      std::min(topk_ceil_div((int64_t)inputSliceSize, (int64_t)warp_size) *
                   (int64_t)warp_size,
               (int64_t)1024));
  gatherTopK<T, IndexType, Dim, /*WithKthValues=*/false>
      <<<grid, block, 0, stream>>>(input,
                                   inputSliceSize,
                                   outputSliceSize,
                                   largest,
                                   numInputSlices,
                                   inputWithinSliceStride,
                                   topK,
                                   topKWithinSliceStride,
                                   indices,
                                   indicesWithinSliceStride,
                                   nullptr);
}

}  // namespace sbtopk

namespace mbtopk {  // multi_block_topk

constexpr int BLOCK_THREADS = 256;
constexpr int RADIX_BITS = 8;
constexpr int RADIX_DIGITS = 1 << RADIX_BITS;  // 256
constexpr int RADIX_MASK = (RADIX_DIGITS - 1);
static_assert(
    RADIX_DIGITS <= BLOCK_THREADS,
    "radixFindKthValues kernel requires RADIX_DIGITS <= BLOCK_THREADS");
constexpr int MIN_ITEMS_PER_THREAD = 4;
constexpr int MAX_ITEMS_PER_THREAD = 64;

template <typename T, typename IndexType>
__global__ void fill(T* x, T value, IndexType size) {
  IndexType idx =
      static_cast<IndexType>(blockIdx.x) * static_cast<IndexType>(blockDim.x) +
      static_cast<IndexType>(threadIdx.x);
  for (IndexType i = idx; i < size; i += static_cast<IndexType>(gridDim.x) *
                                         static_cast<IndexType>(blockDim.x)) {
    x[i] = value;
  }
}

template <typename T, typename IndexType, typename Bitwise, int Dim>
__global__ void __launch_bounds__(BLOCK_THREADS)
    radixFindKthValues(TensorInfo<const T, IndexType> input,
                       uint32_t slice_size,
                       uint32_t* ks_to_find,
                       uint32_t num_slices,
                       IndexType withinSliceStride,
                       int current_bit,
                       int items_per_thread,
                       uint32_t blocks_per_slice,
                       Bitwise desiredMask,
                       Bitwise* desires,
                       int16_t* counts) {
  int items_per_block = items_per_thread * BLOCK_THREADS;
  int tidx = threadIdx.x;
  uint32_t block_idx = getLinearBlockId<uint32_t>();
  uint32_t slice_idx = block_idx / blocks_per_slice;
  uint32_t blk_idx_in_slice = block_idx % blocks_per_slice;
  if (slice_idx >= num_slices) {
    return;
  }

  Bitwise desired = desires[slice_idx];
  IndexType slice_start_index =
      IndexToOffset<const T, IndexType, Dim>::get(slice_idx, input);
  const T* data = &input.data[slice_start_index];

  static_assert(MAX_ITEMS_PER_THREAD * BLOCK_THREADS <
                    std::numeric_limits<int16_t>::max(),
                "blockwise counter too large");
  union __align__(16) TempStorage {
    uint32_t digit_counters[RADIX_DIGITS];
  };
  __shared__ TempStorage temp_storage;

  if (tidx < RADIX_DIGITS) {
    temp_storage.digit_counters[tidx] = 0;
  }
  __syncthreads();

  items_per_thread =
      (blk_idx_in_slice + 1 < blocks_per_slice)
          ? items_per_thread
          : topk_ceil_div(
                (int64_t)(slice_size - blk_idx_in_slice * items_per_block),
                (int64_t)BLOCK_THREADS);

  for (int i = 0; i < items_per_thread; ++i) {
    IndexType idx =
        blk_idx_in_slice * items_per_block + i * BLOCK_THREADS + tidx;
    if (idx < slice_size) {
      idx *= withinSliceStride;
      Bitwise val = TopKTypeConfig<T>::convert(doLdg(&data[idx]));
      bool has_val = ((val & desiredMask) == (desired & desiredMask));
      Bitwise digit =
          Bitfield<Bitwise>::getBitfield(val, current_bit, RADIX_BITS);
      if (has_val) {
        atomicAdd(&temp_storage.digit_counters[digit], 1);
      }
    }
  }

  __syncthreads();

  static_assert(RADIX_DIGITS <= BLOCK_THREADS,
                "this kernel requires RADIX_DIGITS <= BLOCK_THREADS");
  uint32_t digit_count = 0;
  if (tidx < RADIX_DIGITS) {
    digit_count = temp_storage.digit_counters[tidx];
  }

  if (tidx < RADIX_DIGITS) {
    counts[block_idx * RADIX_DIGITS + tidx] = digit_count;
  }
}

template <typename Bitwise, typename T>
__global__ void __launch_bounds__(RADIX_DIGITS)
    computeBlockwiseWithinKCounts(Bitwise* desires_in,
                                  int16_t* counts,
                                  uint32_t* ks_to_find_in,
                                  uint32_t blocks_per_slice,
                                  int current_bit,
                                  bool largest,
                                  uint32_t* withinKCounts,
                                  T* kthValues,
                                  uint32_t* ks_to_find_out,
                                  Bitwise* desires_out,
                                  uint32_t num_blocks) {
  int tidx = threadIdx.x;
  uint32_t block_idx = getLinearBlockId<uint32_t>();
  uint32_t slice_idx = block_idx / blocks_per_slice;

  if (block_idx >= num_blocks) {
    return;
  }

  typedef cub::BlockScan<uint32_t, RADIX_DIGITS> BlockScan;
  union __align__(16) TempStorage {
    uint32_t digit_count_cumsum[RADIX_DIGITS];
    typename BlockScan::TempStorage scan_storage;
  };
  __shared__ TempStorage temp_storage;

  uint32_t digit_count = 0;
  if (tidx < RADIX_DIGITS) {
    for (uint32_t blk = 0; blk < blocks_per_slice; ++blk) {
      digit_count +=
          counts[(slice_idx * blocks_per_slice + blk) * RADIX_DIGITS + tidx];
    }
  }

  uint32_t digit_count_cumsum;
  BlockScan(temp_storage.scan_storage)
      .InclusiveSum(digit_count, digit_count_cumsum);
  __syncthreads();
  if (tidx < RADIX_DIGITS) {
    temp_storage.digit_count_cumsum[tidx] = digit_count_cumsum;
  }
  __syncthreads();

  __shared__ Bitwise desired;
  uint32_t k_to_find = ks_to_find_in[slice_idx];

  if (tidx < RADIX_DIGITS) {
    uint32_t digit_count_cumsum_left =
        (tidx == 0) ? 0 : temp_storage.digit_count_cumsum[tidx - 1];

    if (digit_count_cumsum_left < k_to_find &&
        k_to_find <= digit_count_cumsum) {
      desired = desires_in[slice_idx];
      desired = Bitfield<Bitwise>::setBitfield(
          desired, tidx, current_bit, RADIX_BITS);
      if (block_idx == slice_idx * blocks_per_slice) {
        desires_out[slice_idx] = desired;
        if (current_bit > 0) {
          ks_to_find_out[slice_idx] = k_to_find - digit_count_cumsum_left;
        } else {
          kthValues[slice_idx] = TopKTypeConfig<T>::deconvert(desired);
        }
      }
    }
  }
  __syncthreads();

#if !TOPK_CUB_SUPPORTS_SCAN_BY_KEY()
  return;
#endif

  Bitwise desired_digit =
      Bitfield<Bitwise>::getBitfield(desired, current_bit, RADIX_BITS);

  bool warp_is_active, thread_is_active;
  int warp = tidx / TOPK_WARP_SIZE;
  if (largest) {
    int end_of_warp = warp * TOPK_WARP_SIZE + TOPK_WARP_SIZE - 1;
    warp_is_active = end_of_warp > static_cast<int>(desired_digit);
    thread_is_active = tidx > static_cast<int>(desired_digit);
  } else {
    int start_of_warp = warp * TOPK_WARP_SIZE;
    warp_is_active = start_of_warp < static_cast<int>(desired_digit);
    thread_is_active = tidx < static_cast<int>(desired_digit);
  }
  uint32_t count = 0;
  if (warp_is_active) {
    if (thread_is_active) {
      count = doLdg(counts + block_idx * RADIX_DIGITS + tidx);
    }
    for (int offset = TOPK_WARP_SIZE / 2; offset > 0; offset /= 2) {
      count += TOPK_WARP_SHFL_DOWN(count, offset);
    }
  }

  constexpr int num_warps = RADIX_DIGITS / TOPK_WARP_SIZE;
  __shared__ uint32_t warp_counts[num_warps];
  if (tidx % TOPK_WARP_SIZE == 0) {
    warp_counts[warp] = count;
  }
  __syncthreads();
#ifdef __HIPCC__
  assert(RADIX_DIGITS < TOPK_WARP_SIZE * TOPK_WARP_SIZE);
#else
  static_assert(RADIX_DIGITS < TOPK_WARP_SIZE * TOPK_WARP_SIZE,
                "Assuming only 1 warp is needed for final reduction");
#endif
  if (warp != 0) {
    return;
  }
  count = 0;
  if (tidx < num_warps) {
    count = warp_counts[tidx];
  }
  for (int offset = num_warps / 2; offset > 0; offset /= 2) {
    count += TOPK_WARP_SHFL_DOWN(count, offset);
  }
  if (tidx == 0) {
    withinKCounts[block_idx] += count;
  }
}

#if TOPK_CUB_SUPPORTS_SCAN_BY_KEY()
template <typename Bitwise>
__global__ void computeBlockwiseKthCounts(Bitwise* desires,
                                          int16_t* counts,
                                          uint32_t num_blocks,
                                          uint32_t blocks_per_slice,
                                          uint32_t* kthCounts) {
  TOPK_CUDA_KERNEL_LOOP_TYPE(idx, num_blocks, uint32_t) {
    uint32_t slice_idx = idx / blocks_per_slice;
    Bitwise desired = doLdg(desires + slice_idx);
    Bitwise desired_digit =
        Bitfield<Bitwise>::getBitfield(desired, 0, RADIX_BITS);
    kthCounts[idx] = doLdg(counts + idx * RADIX_DIGITS + desired_digit);
  }
}

template <typename T, typename IndexType, int Dim>
__global__ void __launch_bounds__(BLOCK_THREADS)
    gatherTopK(TensorInfo<const T, IndexType> input,
               IndexType inputSliceSize,
               IndexType outputSliceSize,
               bool largest,
               uint32_t numInputSlices,
               IndexType inputWithinSliceStride,
               TensorInfo<T, IndexType> topK,
               IndexType topKWithinSliceStride,
               TensorInfo<int64_t, IndexType> indices,
               IndexType indicesWithinSliceStride,
               uint32_t items_per_thread,
               uint32_t blocks_per_slice,
               T* kthValues,
               uint32_t* withinKCounts,
               uint32_t* kthCounts,
               uint32_t num_blocks) {
  uint32_t items_per_block = items_per_thread * BLOCK_THREADS;
  uint32_t tidx = threadIdx.x;
  uint32_t block_idx = getLinearBlockId<uint32_t>();

  if (block_idx >= num_blocks) {
    return;
  }

  uint32_t slice_idx = block_idx / blocks_per_slice;
  uint32_t blk_idx_in_slice = block_idx % blocks_per_slice;

  items_per_thread =
      (blk_idx_in_slice + 1 < blocks_per_slice)
          ? items_per_thread
          : topk_ceil_div(
                (int64_t)(inputSliceSize - blk_idx_in_slice * items_per_block),
                (int64_t)BLOCK_THREADS);

  IndexType sliceStartIndex =
      IndexToOffset<const T, IndexType, Dim>::get(slice_idx, input);
  IndexType topKSliceStartIndex =
      IndexToOffset<T, IndexType, Dim>::get(slice_idx, topK);
  IndexType indicesSliceStartIndex =
      IndexToOffset<int64_t, IndexType, Dim>::get(slice_idx, indices);

  const T* inputSliceStart = &input.data[sliceStartIndex];
  T* topKSliceStart = &topK.data[topKSliceStartIndex];
  int64_t* indicesSliceStart = &indices.data[indicesSliceStartIndex];

  T kthValue = kthValues[slice_idx];
  const auto kthValueConverted = TopKTypeConfig<T>::convert(kthValue);

  uint32_t startWithinK = 0;
  if (blk_idx_in_slice > 0) {
    startWithinK = withinKCounts[block_idx - 1];
  }
  uint32_t startKth =
      withinKCounts[slice_idx * blocks_per_slice + blocks_per_slice - 1];
  if (blk_idx_in_slice > 0) {
    startKth += kthCounts[block_idx - 1];
  }

  typedef cub::BlockScan<uint32_t, BLOCK_THREADS> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;
  for (uint32_t i = 0; i < items_per_thread; ++i) {
    IndexType idx =
        blk_idx_in_slice * items_per_block + i * BLOCK_THREADS + tidx;
    T val;
    int withinK = 0;
    int kth = 0;
    if (idx < inputSliceSize) {
      val = doLdg(inputSliceStart + idx * inputWithinSliceStride);
      const auto valConverted = TopKTypeConfig<T>::convert(val);
      withinK = (largest ? valConverted > kthValueConverted
                         : valConverted < kthValueConverted);
      kth = (valConverted == kthValueConverted);
    }

    uint32_t withinKIndex;
    uint32_t numWithinK;
    BlockScan(temp_storage).ExclusiveSum(withinK, withinKIndex, numWithinK);
    __syncthreads();
    if (withinK) {
      uint32_t offset = withinKIndex + startWithinK;
      topKSliceStart[offset * topKWithinSliceStride] = val;
      indicesSliceStart[offset * indicesWithinSliceStride] = idx;
    }
    startWithinK += numWithinK;

    if (startKth < outputSliceSize) {
      uint32_t kthIndex;
      uint32_t numKth;
      BlockScan(temp_storage).ExclusiveSum(kth, kthIndex, numKth);
      __syncthreads();
      if (kth) {
        uint32_t offset = kthIndex + startKth;
        if (offset < outputSliceSize) {
          topKSliceStart[offset * topKWithinSliceStride] = val;
          indicesSliceStart[offset * indicesWithinSliceStride] = idx;
        }
      }
      startKth += numKth;
    }
  }
}
#endif  // TOPK_CUB_SUPPORTS_SCAN_BY_KEY

// get_items_per_thread: compute optimal items per thread based on GPU occupancy
int get_items_per_thread(uint64_t num_slices,
                         uint64_t slice_size,
                         int device_id) {
  constexpr int REGS_PER_THREAD = 40;
  constexpr int REGS_PER_BLOCK = REGS_PER_THREAD * BLOCK_THREADS;
  const auto& prop = phi::backends::gpu::GetDeviceProperties(device_id);
  int mpc = prop.multiProcessorCount;
#ifdef PADDLE_WITH_HIP
  // HIP/DCU: hipDeviceProp_t lacks regsPerMultiprocessor and
  // maxBlocksPerMultiProcessor. Use conservative defaults:
  // 65536 registers per CU is typical for AMD GCN/CDNA architectures.
  // maxThreadsPerMultiProcessor / BLOCK_THREADS as blocks_per_mp estimate.
  int regs_per_mp = 65536;
  int max_blocks_per_mp = prop.maxThreadsPerMultiProcessor / BLOCK_THREADS;
#else
  int regs_per_mp = prop.regsPerMultiprocessor;
  int max_blocks_per_mp = prop.maxBlocksPerMultiProcessor;
#endif
  int blocks_per_mp = std::min(regs_per_mp / REGS_PER_BLOCK, max_blocks_per_mp);
  int64_t items_per_thread =
      topk_ceil_div((int64_t)(slice_size * num_slices),
                    (int64_t)(mpc * blocks_per_mp * BLOCK_THREADS));
  items_per_thread = std::max(
      MIN_ITEMS_PER_THREAD,
      std::min(static_cast<int>(items_per_thread), MAX_ITEMS_PER_THREAD));
  return items_per_thread;
}

class BlockIdxToKey {
  uint32_t blocks_per_slice;

 public:
  explicit BlockIdxToKey(uint32_t blocks_per_slice)
      : blocks_per_slice(blocks_per_slice) {}
  __device__ __forceinline__ uint32_t operator()(uint32_t blk) const {
    return blk / blocks_per_slice;
  }
};

template <typename T, typename IndexType, int Dim>
void launch(TensorInfo<const T, IndexType> input,
            IndexType inputSliceSize,
            IndexType outputSliceSize,
            bool largest,
            uint32_t numInputSlices,
            IndexType inputWithinSliceStride,
            TensorInfo<T, IndexType> topK,
            IndexType topKWithinSliceStride,
            TensorInfo<int64_t, IndexType> indices,
            IndexType indicesWithinSliceStride,
            gpuStream_t stream,
            int device_id,
            const phi::Place& place) {
  int items_per_thread =
      get_items_per_thread(numInputSlices, inputSliceSize, device_id);
  int items_per_block = items_per_thread * BLOCK_THREADS;

  using Bitwise = typename TopKTypeConfig<T>::RadixType;
  uint32_t blocks_per_slice =
      topk_ceil_div((int64_t)inputSliceSize, (int64_t)items_per_block);
  uint32_t num_blocks = numInputSlices * blocks_per_slice;

  // Temporary storage allocation using phi::memory_utils
  auto phi_stream = phi::Stream(reinterpret_cast<phi::StreamId>(stream));

  auto kthValues_buffer =
      phi::memory_utils::Alloc(place, numInputSlices * sizeof(T), phi_stream);
  T* kthValues = reinterpret_cast<T*>(kthValues_buffer->ptr());

  auto semaphores_buffer = phi::memory_utils::Alloc(
      place, numInputSlices * sizeof(uint32_t), phi_stream);
  uint32_t* semaphores = reinterpret_cast<uint32_t*>(semaphores_buffer->ptr());
#ifdef PADDLE_WITH_HIP
  hipMemsetAsync(semaphores, 0, numInputSlices * sizeof(uint32_t), stream);
#else
  cudaMemsetAsync(semaphores, 0, numInputSlices * sizeof(uint32_t), stream);
#endif

  auto ks_to_find_buffer = phi::memory_utils::Alloc(
      place, 2 * numInputSlices * sizeof(uint32_t), phi_stream);
  uint32_t* ks_to_find = reinterpret_cast<uint32_t*>(ks_to_find_buffer->ptr());
  uint32_t k_to_find =
      largest ? inputSliceSize - outputSliceSize + 1 : outputSliceSize;
  fill<uint32_t>
      <<<std::min(((int64_t)numInputSlices + 511) / 512, (int64_t)1073741824),
         512,
         0,
         stream>>>(ks_to_find, k_to_find, numInputSlices);

  auto desired_buffer = phi::memory_utils::Alloc(
      place, 2 * numInputSlices * sizeof(Bitwise), phi_stream);
  Bitwise* desired = reinterpret_cast<Bitwise*>(desired_buffer->ptr());

  auto counts_buffer = phi::memory_utils::Alloc(
      place, num_blocks * RADIX_DIGITS * sizeof(int16_t), phi_stream);
  int16_t* counts = reinterpret_cast<int16_t*>(counts_buffer->ptr());
  static_assert(MAX_ITEMS_PER_THREAD * BLOCK_THREADS <
                    std::numeric_limits<int16_t>::max(),
                "blockwise counter too large");

#if TOPK_CUB_SUPPORTS_SCAN_BY_KEY()
  auto withinKCounts_buffer = phi::memory_utils::Alloc(
      place, num_blocks * sizeof(uint32_t), phi_stream);
  uint32_t* withinKCounts =
      reinterpret_cast<uint32_t*>(withinKCounts_buffer->ptr());
#ifdef PADDLE_WITH_HIP
  hipMemsetAsync(withinKCounts, 0, num_blocks * sizeof(uint32_t), stream);
#else
  cudaMemsetAsync(withinKCounts, 0, num_blocks * sizeof(uint32_t), stream);
#endif

  auto kthCounts_buffer = phi::memory_utils::Alloc(
      place, num_blocks * sizeof(uint32_t), phi_stream);
  uint32_t* kthCounts = reinterpret_cast<uint32_t*>(kthCounts_buffer->ptr());
#else
  uint32_t* withinKCounts = nullptr;
#endif

  Bitwise desiredMask = 0;
  dim3 grid;
  bool ok = getGridFromTiles(num_blocks, &grid);
  assert(ok);
  (void)ok;
  dim3 block(BLOCK_THREADS);

  uint32_t* ks_to_find_in = ks_to_find;
  uint32_t* ks_to_find_out = ks_to_find + numInputSlices;
  Bitwise* desired_in = desired;
  Bitwise* desired_out = desired + numInputSlices;

  for (int current_bit = sizeof(T) * 8 - RADIX_BITS; current_bit >= 0;
       current_bit -= RADIX_BITS) {
    radixFindKthValues<T, IndexType, Bitwise, Dim>
        <<<grid, block, 0, stream>>>(input,
                                     inputSliceSize,
                                     ks_to_find_in,
                                     numInputSlices,
                                     inputWithinSliceStride,
                                     current_bit,
                                     items_per_thread,
                                     blocks_per_slice,
                                     desiredMask,
                                     desired_in,
                                     counts);

    computeBlockwiseWithinKCounts<Bitwise, T>
        <<<grid, RADIX_DIGITS, 0, stream>>>(desired_in,
                                            counts,
                                            ks_to_find_in,
                                            blocks_per_slice,
                                            current_bit,
                                            largest,
                                            withinKCounts,
                                            kthValues,
                                            ks_to_find_out,
                                            desired_out,
                                            num_blocks);

    auto tmp_desired = desired_in;
    desired_in = desired_out;
    desired_out = tmp_desired;
    auto tmp_ks = ks_to_find_in;
    ks_to_find_in = ks_to_find_out;
    ks_to_find_out = tmp_ks;
    // Host-side equivalent of Bitfield<Bitwise>::setBitfield(desiredMask,
    // RADIX_MASK, current_bit, RADIX_BITS) Cannot use Bitfield::setBitfield
    // here because it's __device__-only (uses PTX asm)
    {
      Bitwise mask = ((Bitwise(1) << RADIX_BITS) - 1) << current_bit;
      desiredMask =
          (desiredMask & ~mask) | ((Bitwise(RADIX_MASK) << current_bit) & mask);
    }
  }
  desired = desired_in;

#if TOPK_CUB_SUPPORTS_SCAN_BY_KEY()
  computeBlockwiseKthCounts<Bitwise>
      <<<std::min(((int64_t)numInputSlices + 255) / 256, (int64_t)1073741824),
         256,
         0,
         stream>>>(desired, counts, num_blocks, blocks_per_slice, kthCounts);

  // Use cub::DeviceScan::InclusiveSumByKey
  using counting_iter_t = cub::CountingInputIterator<uint32_t>;
  using slice_idx_iter_t =
      cub::TransformInputIterator<uint32_t, BlockIdxToKey, counting_iter_t>;
  slice_idx_iter_t slice_idx_iter(counting_iter_t(0),
                                  BlockIdxToKey(blocks_per_slice));

  // InclusiveSumByKey for withinKCounts
  {
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSumByKey(nullptr,
                                       temp_storage_bytes,
                                       slice_idx_iter,
                                       withinKCounts,
                                       withinKCounts,
                                       num_blocks,
                                       cub::Equality(),
                                       stream);
    auto temp_buf =
        phi::memory_utils::Alloc(place, temp_storage_bytes, phi_stream);
    cub::DeviceScan::InclusiveSumByKey(temp_buf->ptr(),
                                       temp_storage_bytes,
                                       slice_idx_iter,
                                       withinKCounts,
                                       withinKCounts,
                                       num_blocks,
                                       cub::Equality(),
                                       stream);
  }
  // InclusiveSumByKey for kthCounts
  {
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSumByKey(nullptr,
                                       temp_storage_bytes,
                                       slice_idx_iter,
                                       kthCounts,
                                       kthCounts,
                                       num_blocks,
                                       cub::Equality(),
                                       stream);
    auto temp_buf =
        phi::memory_utils::Alloc(place, temp_storage_bytes, phi_stream);
    cub::DeviceScan::InclusiveSumByKey(temp_buf->ptr(),
                                       temp_storage_bytes,
                                       slice_idx_iter,
                                       kthCounts,
                                       kthCounts,
                                       num_blocks,
                                       cub::Equality(),
                                       stream);
  }

  gatherTopK<T, IndexType, Dim>
      <<<grid, block, 0, stream>>>(input,
                                   inputSliceSize,
                                   outputSliceSize,
                                   largest,
                                   numInputSlices,
                                   inputWithinSliceStride,
                                   topK,
                                   topKWithinSliceStride,
                                   indices,
                                   indicesWithinSliceStride,
                                   items_per_thread,
                                   blocks_per_slice,
                                   kthValues,
                                   withinKCounts,
                                   kthCounts,
                                   num_blocks);
#else
  // Fallback: use single-block gatherTopK with kthValues
  {
    dim3 grid2;
    bool ok2 = getGridFromTiles(numInputSlices, &grid2);
    assert(ok2);
    (void)ok2;
    int warp_size = TOPK_WARP_SIZE;
    dim3 block2(
        std::min(topk_ceil_div((int64_t)inputSliceSize, (int64_t)warp_size) *
                     (int64_t)warp_size,
                 (int64_t)1024));
    sbtopk::gatherTopK<T, IndexType, Dim, /*WithKthValues=*/true>
        <<<grid2, block2, 0, stream>>>(input,
                                       inputSliceSize,
                                       outputSliceSize,
                                       largest,
                                       numInputSlices,
                                       inputWithinSliceStride,
                                       topK,
                                       topKWithinSliceStride,
                                       indices,
                                       indicesWithinSliceStride,
                                       kthValues);
  }
#endif
}

}  // namespace mbtopk

bool should_use_multiblock(int64_t num_slices, int64_t slice_size) {
  if (num_slices > std::numeric_limits<uint32_t>::max() ||
      slice_size > std::numeric_limits<uint32_t>::max())
    return false;
#if TOPK_CUB_SUPPORTS_SCAN_BY_KEY()
  return (num_slices <= 20 && slice_size >= 20000) ||
         (num_slices > 20 && num_slices <= 40 && slice_size >= 10000) ||
         (num_slices > 40 && num_slices <= 80 && slice_size >= 8000) ||
         (num_slices > 80 && num_slices < 200 && slice_size >= 5000) ||
         (num_slices >= 200 && num_slices < 800 && slice_size >= 3000) ||
         (num_slices >= 800 && num_slices <= 4000 && slice_size >= 800) ||
         (num_slices > 4000 && slice_size >= 400);
#else
  return (num_slices <= 400 && slice_size >= 5000) ||
         (num_slices > 400 && num_slices < 4000 && slice_size >= 1000) ||
         (num_slices >= 4000 && slice_size >= 300);
#endif
}

// canUse32BitIndexMath: check if tensor indexing fits in 32-bit integers
bool canUse32BitIndexMath(
    const phi::DenseTensor& t,
    int64_t max_elem = std::numeric_limits<int32_t>::max()) {
  int64_t elements = t.numel();
  if (elements >= max_elem) {
    return false;
  }
  if (elements == 0) {
    return max_elem > 0;
  }

  int64_t offset = 0;
  int64_t linearId = elements - 1;

  for (int i = t.dims().size() - 1; i >= 0; --i) {
    int64_t curDimIndex = linearId % t.dims()[i];
    int64_t curDimOffset = curDimIndex * t.strides()[i];
    offset += curDimOffset;
    linearId /= t.dims()[i];
  }

  if (offset >= max_elem) {
    return false;
  }

  return true;
}

}  // namespace topk_impl
#endif  // PADDLE_PHI_KERNELS_FUNCS_TOP_K_CUDA_KERNEL_H_
