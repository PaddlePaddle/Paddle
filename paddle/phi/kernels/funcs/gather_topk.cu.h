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

#pragma once

#include <limits>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/top_k_function_cuda.h"

namespace phi {
namespace funcs {

#define MAX_TENSORINFO_DIMS 25

template <typename T>
inline std::pair<int64_t, int64_t> collapse_dims(T* sizes,
                                                 T* strides,
                                                 int64_t dims,
                                                 const int excludeDim = -1) {
  int64_t stopDim = (excludeDim == -1) ? dims : excludeDim;
  int64_t newIndex = -1;
  int64_t oldIndex = 0;
  int64_t remappedExcludedDim = -1;

  while (oldIndex < dims) {
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

    if (oldIndex != dims) {
      ++newIndex;
      sizes[newIndex] = sizes[oldIndex];
      strides[newIndex] = strides[oldIndex];
      remappedExcludedDim = newIndex;
      ++oldIndex;
      stopDim = dims;
    }
  }

  if (newIndex == -1 || (newIndex == 0 && sizes[0] == 1)) {
    // dims = 1; sizes[0] = 1; strides[0] = 1;
    return std::pair<int64_t, int64_t>(0, 1);
  }

  return std::pair<int64_t, int64_t>(remappedExcludedDim, newIndex + 1);
}

template <typename T, typename IndexType>
struct TensorInfo {
  T* data;
  IndexType sizes[MAX_TENSORINFO_DIMS];
  IndexType strides[MAX_TENSORINFO_DIMS];
  int dims;

  TensorInfo() {
    data = nullptr;
    dims = 0;
  }

  TensorInfo(T* p,
             int dim,
             IndexType sz[MAX_TENSORINFO_DIMS],
             IndexType st[MAX_TENSORINFO_DIMS]) {
    data = p;
    dims = dim;
    for (int i = 0; i < dim; ++i) {
      sizes[i] = sz[i];
      strides[i] = st[i];
    }
  }

  bool isContiguous() const {
    // Basic check: strides[i] == sizes[i+1]*strides[i+1]
    // Not strictly sufficient but for collapse logic it helps
    if (dims == 0) return true;
    IndexType z = 1;
    for (int i = dims - 1; i >= 0; i--) {
      if (sizes[i] != 1) {
        if (strides[i] != z) return false;
        z *= sizes[i];
      }
    }
    return true;
  }

  int collapseDims(const int excludeDim = -1) {
    auto result =
        collapse_dims(sizes, strides, static_cast<int64_t>(dims), excludeDim);
    dims = static_cast<int>(std::get<1>(result));
    return static_cast<int>(std::get<0>(result));
  }
};

template <typename T, typename IndexType, int Dims>
struct IndexToOffset {
  static __host__ __device__ IndexType
  get(IndexType linearId, const TensorInfo<T, IndexType>& info) {
    IndexType offset = 0;
    for (int i = Dims - 1; i > 0; --i) {
      IndexType curDimIndex = linearId % info.sizes[i];
      IndexType curDimOffset = curDimIndex * info.strides[i];
      offset += curDimOffset;
      linearId /= info.sizes[i];
    }
    return offset + linearId * info.strides[0];
  }
};

template <typename T, typename IndexType>
struct IndexToOffset<T, IndexType, -1> {
  static inline __host__ __device__ IndexType
  get(IndexType linearId, const TensorInfo<T, IndexType>& info) {
    IndexType offset = 0;
    for (int i = info.dims - 1; i > 0; --i) {
      IndexType curDimIndex = linearId % info.sizes[i];
      IndexType curDimOffset = curDimIndex * info.strides[i];
      offset += curDimOffset;
      linearId /= info.sizes[i];
    }
    return offset + linearId * info.strides[0];
  }
};

template <typename T, typename IndexType>
struct IndexToOffset<T, IndexType, -2> {  // Contiguous
  static inline __host__ __device__ IndexType
  get(IndexType linearId, const TensorInfo<T, IndexType>& info) {
    return linearId;
  }
};

template <typename T, typename IndexType>
TensorInfo<T, IndexType> getTensorInfo(DenseTensor& t) {  // NOLINT
  IndexType sizes[MAX_TENSORINFO_DIMS];
  IndexType strides[MAX_TENSORINFO_DIMS];
  int dims = t.dims().size();
  if (dims == 0) dims = 1;

  auto ddim = t.dims();
  auto dstride = t.strides();

  if (t.dims().size() == 0) {
    sizes[0] = 1;
    strides[0] = 1;
  } else {
    for (int i = 0; i < dims; ++i) {
      sizes[i] = ddim[i];
      strides[i] = dstride[i];
    }
  }
  using NonConstT = typename std::remove_const<T>::type;
  return TensorInfo<T, IndexType>(t.data<NonConstT>(), dims, sizes, strides);
}

template <typename T, typename IndexType>
TensorInfo<T, IndexType> getTensorInfo(const DenseTensor& t) {
  IndexType sizes[MAX_TENSORINFO_DIMS];
  IndexType strides[MAX_TENSORINFO_DIMS];
  int dims = t.dims().size();
  if (dims == 0) dims = 1;

  auto ddim = t.dims();
  auto dstride = t.strides();

  if (t.dims().size() == 0) {
    sizes[0] = 1;
    strides[0] = 1;
  } else {
    for (int i = 0; i < dims; ++i) {
      sizes[i] = ddim[i];
      strides[i] = dstride[i];
    }
  }
  using NonConstT = typename std::remove_const<T>::type;
  return TensorInfo<T, IndexType>(t.data<NonConstT>(), dims, sizes, strides);
}

// --- Bitonic Sort Logic ---

template <typename T>
struct LTOp {
  __device__ bool operator()(const T& a, const T& b) const {
    // Handle NaN: NaN is treated as largest value (sorted to the end)
    // This ensures ascending sort places NaN after all normal values
    if (isnan(static_cast<float>(a)))
      return false;  // a is NaN, a is not less than b
    if (isnan(static_cast<float>(b)))
      return true;  // b is NaN, a is less than b
    return a < b;
  }
};

template <typename T>
struct GTOp {
  __device__ bool operator()(const T& a, const T& b) const {
    // Handle NaN: NaN is treated as smallest value (sorted to the end)
    // This ensures descending sort places NaN after all normal values
    if (isnan(static_cast<float>(a)))
      return false;  // a is NaN, a is not greater than b
    if (isnan(static_cast<float>(b)))
      return true;  // b is NaN, a is greater than b
    return a > b;
  }
};

template <typename T>
__device__ inline void swapVars(T& t1, T& t2) {  // NOLINT
  T tmp = t1;
  t1 = t2;
  t2 = tmp;
}

template <typename Comparator, typename K, typename V>
__device__ inline void bitonicSwap(K& kA,         // NOLINT
                                   V& vA,         // NOLINT
                                   bool& validA,  // NOLINT
                                   K& kB,         // NOLINT
                                   V& vB,         // NOLINT
                                   bool& validB,  // NOLINT
                                   bool dir,
                                   const Comparator& comp) {
  // Invalid entries always sort to the end
  bool swap = (comp(kA, kB) && validA) || !validB;
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
      bitonicSwap<Comparator, K, V>(keys[pos],
                                    values[pos],
                                    valid[pos],
                                    keys[pos + stride],
                                    values[pos + stride],
                                    valid[pos + stride],
                                    flag,
                                    comp);
    }
  }

#pragma unroll
  for (unsigned int stride = Power2SortSize / 2; stride > 0; stride /= 2) {
    __syncthreads();

    unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
    bitonicSwap<Comparator, K, V>(keys[pos],
                                  values[pos],
                                  valid[pos],
                                  keys[pos + stride],
                                  values[pos + stride],
                                  valid[pos + stride],
                                  false,
                                  comp);
  }

  __syncthreads();
}

template <typename T>
__device__ inline int get_linear_block_id() {
  return blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x +
         blockIdx.x;
}

template <int KeyDims,
          int ValueDims,
          int block_dim_x,
          int max_block_dim_y,
          typename K,
          typename V,
          typename Comparator,
          typename IndexType>
__global__ void bitonicSortKVInPlace(TensorInfo<K, IndexType> keys,
                                     IndexType keySlices,
                                     IndexType keySliceSize,
                                     IndexType keySliceStride,
                                     TensorInfo<V, IndexType> values,
                                     IndexType valueSliceStride,
                                     Comparator comp) {
  // Find the slice of the tensor that we are sorting
  // NOTE: blockDim.y may be less max_block_dim_y
  const IndexType blockIndex = get_linear_block_id<IndexType>();
  const IndexType linearIndex = blockIndex * blockDim.y + threadIdx.y;

  // If the entire block is out of bounds exit early
  if (blockIndex * blockDim.y >= keySlices) {
    return;
  }
  // It's also possible for some rows of a block to be out of bounds
  // but all thread need to run for __syncthreads to work.
  const bool row_valid = linearIndex < keySlices;

  constexpr int items_per_thread = 2;
  constexpr int Power2SortSize = block_dim_x * items_per_thread;

  // Storage for max_block_dim_y sorts performed in parallel
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

// Load 2 values per thread into the shared workspace
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

  // Sort!
  bitonicSort<Power2SortSize, IndexType>(
      sharedKeys, sharedValues, sharedValid, comp);

  if (!row_valid) {
    return;
  }

// Store outputs
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

// --- Radix Sort with Stride Support ---

// Adapted from RadixCountUsingMask in top_k_function_cuda.h
template <typename T,
          typename RadixType,
          typename IndexType,
          int RadixSize,
          int RadixBits>
__device__ void RadixCountUsingMaskStrided(const T* input,
                                           IndexType counts[RadixSize],
                                           IndexType* shared_mem,
                                           RadixType desired,
                                           RadixType desired_mask,
                                           int radix_digit_pos,
                                           IndexType slice_size,
                                           IndexType within_slice_stride) {
#pragma unroll
  for (int i = 0; i < RadixSize; ++i) {
    counts[i] = 0;
  }

  if (threadIdx.x < RadixSize) {
    shared_mem[threadIdx.x] = 0;
  }
  __syncthreads();

  for (IndexType i = threadIdx.x; i < slice_size; i += blockDim.x) {
    RadixType val = RadixTypeConfig<T>::Convert(input[i * within_slice_stride]);

    bool has_val = ((val & desired_mask) == desired);
    RadixType digit_in_radix =
        Bitfield<RadixType>::GetBitfield(val, radix_digit_pos, RadixBits);

#pragma unroll
    for (uint32_t j = 0; j < RadixSize; ++j) {
      bool vote = has_val && (digit_in_radix == j);
      counts[j] += __popc(__ballot_sync(__activemask(), vote));
    }
  }

  if (GetLaneId() == 0) {
#pragma unroll
    for (uint32_t i = 0; i < RadixSize; ++i) {
      phi::CudaAtomicAdd(&shared_mem[i], counts[i]);
    }
  }

  __syncthreads();

#pragma unroll
  for (uint32_t i = 0; i < RadixSize; ++i) {
    counts[i] = shared_mem[i];
  }

  __syncthreads();
}

// Adapted from FindPattern in top_k_function_cuda.h
template <typename T, typename RadixType, typename IndexType>
__device__ T FindPatternStrided(const T* input,
                                T* shared_mem,
                                IndexType slice_size,
                                IndexType within_slice_stride,
                                RadixType desired,
                                RadixType desired_mask) {
  if (threadIdx.x < 2) {
    shared_mem[threadIdx.x] = static_cast<T>(0);
  }
  __syncthreads();

  IndexType block_dim = static_cast<IndexType>(blockDim.x);
  IndexType loop = ((slice_size + block_dim - 1) / block_dim * block_dim);
  for (IndexType i = threadIdx.x; i < loop; i += blockDim.x) {
    bool valid = (i < slice_size);
    T v = valid ? input[i * within_slice_stride] : static_cast<T>(0);

    if (valid && ((RadixTypeConfig<T>::Convert(v) & desired_mask) == desired)) {
      shared_mem[0] = static_cast<T>(1);
      shared_mem[1] = v;
    }

    __syncthreads();

    T found = shared_mem[0];
    T val = shared_mem[1];

    __syncthreads();

    if (found != static_cast<T>(0)) {
      return val;
    }
  }

  // assert(false);
  return static_cast<T>(0);
}

// Adapted from RadixSearch in top_k_function_cuda.h
template <typename T, typename RadixType, typename IndexType, bool Largest>
__device__ void RadixSelectStrided(const T* input,
                                   IndexType k,
                                   IndexType slice_size,
                                   IndexType within_slice_stride,
                                   void* shared_mem,
                                   T* kth_value) {
  IndexType counts[RADIX_SIZE];
  IndexType k_left = k;
  RadixType desired = 0;
  RadixType desired_mask = 0;

#pragma unroll
  for (int digit_pos = sizeof(T) * 8 - RADIX_BITS; digit_pos >= 0;
       digit_pos -= RADIX_BITS) {
    RadixCountUsingMaskStrided<T, RadixType, IndexType, RADIX_SIZE, RADIX_BITS>(
        input,
        counts,
        static_cast<IndexType*>(shared_mem),
        desired,
        desired_mask,
        digit_pos,
        slice_size,
        within_slice_stride);

    auto found_unique = [&](int i, IndexType count) -> bool {
      if (count == 1 && k_left == 1) {
        desired =
            Bitfield<RadixType>::SetBitfield(desired, i, digit_pos, RADIX_BITS);
        desired_mask = Bitfield<RadixType>::SetBitfield(
            desired_mask, RADIX_MASK, digit_pos, RADIX_BITS);

        *kth_value = FindPatternStrided<T, RadixType, IndexType>(
            input,
            static_cast<T*>(shared_mem),
            slice_size,
            within_slice_stride,
            desired,
            desired_mask);
        return true;
      }
      return false;
    };
    auto found_non_unique = [&](int i, IndexType count) -> bool {
      if (count >= k_left) {
        desired =
            Bitfield<RadixType>::SetBitfield(desired, i, digit_pos, RADIX_BITS);
        desired_mask = Bitfield<RadixType>::SetBitfield(
            desired_mask, RADIX_MASK, digit_pos, RADIX_BITS);

        return true;
      }
      k_left -= count;
      return false;
    };

    if (Largest) {
      // Descending order
#pragma unroll
      for (int i = RADIX_SIZE - 1; i >= 0; --i) {
        IndexType count = counts[i];
        if (found_unique(i, count)) {
          return;
        }
        if (found_non_unique(i, count)) {
          break;
        }
      }
    } else {
      // Ascending order
#pragma unroll
      for (int i = 0; i < RADIX_SIZE; ++i) {
        IndexType count = counts[i];
        if (found_unique(i, count)) {
          return;
        }
        if (found_non_unique(i, count)) {
          break;
        }
      }
    }
  }

  *kth_value = RadixTypeConfig<T>::Deconvert(desired);
}

// --- Main Kernel Logic ---

template <typename T, typename IndexType, int Dim>
__global__ void gatherTopK(TensorInfo<const T, IndexType> input,
                           IndexType inputSliceSize,
                           IndexType outputSliceSize,  // aka `k`
                           bool largest,
                           IndexType numInputSlices,
                           IndexType inputWithinSliceStride,
                           TensorInfo<T, IndexType> topK,
                           IndexType topKWithinSliceStride,
                           TensorInfo<int64_t, IndexType> indices,
                           IndexType indicesWithinSliceStride) {
  // Shared memory for radix selection
  extern __shared__ char smem_char[];
  void* smem_void = static_cast<void*>(smem_char);

  IndexType slice =
      static_cast<IndexType>(blockIdx.x) * static_cast<IndexType>(blockDim.y) +
      static_cast<IndexType>(threadIdx.y);
  // Handle 3D grid for large batch
  IndexType linearBlockId =
      static_cast<IndexType>(blockIdx.z) * static_cast<IndexType>(gridDim.y) *
          static_cast<IndexType>(gridDim.x) +
      static_cast<IndexType>(blockIdx.y) * static_cast<IndexType>(gridDim.x) +
      static_cast<IndexType>(blockIdx.x);
  if (linearBlockId >= numInputSlices) return;

  slice = linearBlockId;

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
  T topKValue = static_cast<T>(0);
  if (largest) {
    RadixSelectStrided<T,
                       typename RadixTypeConfig<T>::RadixType,
                       IndexType,
                       true>(inputSliceStart,
                             outputSliceSize,
                             inputSliceSize,
                             inputWithinSliceStride,
                             smem_void,
                             &topKValue);
  } else {
    RadixSelectStrided<T,
                       typename RadixTypeConfig<T>::RadixType,
                       IndexType,
                       false>(inputSliceStart,
                              outputSliceSize,
                              inputSliceSize,
                              inputWithinSliceStride,
                              smem_void,
                              &topKValue);
  }

  const auto converted_kth_value = RadixTypeConfig<T>::Convert(topKValue);

  int* scan_smem = static_cast<int*>(smem_void);

  int block_dim = static_cast<int>(blockDim.x);
  IndexType loop = ((inputSliceSize + block_dim - 1) / block_dim * block_dim);
  IndexType write_start = 0;

  for (IndexType i = threadIdx.x; i < loop; i += blockDim.x) {
    bool valid = i < inputSliceSize;
    T v =
        valid ? inputSliceStart[i * inputWithinSliceStride] : static_cast<T>(0);
    const auto convertd_v = RadixTypeConfig<T>::Convert(v);
    bool is_top_k;
    if (largest) {
      is_top_k = valid && (convertd_v > converted_kth_value);
    } else {
      is_top_k = valid && (convertd_v < converted_kth_value);
    }

    int index;
    int carry;
    ExclusiveBinaryPrefixScan<int, true, kps::AddFunctor<int>>(
        scan_smem, is_top_k, &index, &carry, kps::AddFunctor<int>());

    if (is_top_k) {
      IndexType write_index = write_start + index;
      if (write_index < outputSliceSize) {
        topKSliceStart[write_index * topKWithinSliceStride] = v;
        indicesSliceStart[write_index * indicesWithinSliceStride] = i;
      }
    }
    write_start += carry;
  }

  // 3. Fill the rest with value == kth_value
  IndexType remain = outputSliceSize - write_start;
  for (IndexType i = threadIdx.x; i < loop; i += blockDim.x) {
    bool valid = i < inputSliceSize;
    T v =
        valid ? inputSliceStart[i * inputWithinSliceStride] : static_cast<T>(0);
    const auto convertd_v = RadixTypeConfig<T>::Convert(v);
    bool is_top_k = valid && (convertd_v == converted_kth_value);

    int index;
    int carry;
    ExclusiveBinaryPrefixScan<int, true, kps::AddFunctor<int>>(
        scan_smem, is_top_k, &index, &carry, kps::AddFunctor<int>());

    if (is_top_k && index < remain) {
      IndexType write_index = write_start + index;
      if (write_index < outputSliceSize) {
        topKSliceStart[write_index * topKWithinSliceStride] = v;
        indicesSliceStart[write_index * indicesWithinSliceStride] = i;
      }
    }

    if (carry >= remain) {
      break;
    }

    remain -= carry;
    write_start += carry;
  }
}

// Launcher
template <typename T, typename IndexType>
void LaunchGatherTopK(const phi::GPUContext& dev_ctx,
                      const DenseTensor& input,
                      int64_t k,
                      int dim,
                      bool largest,
                      DenseTensor* values,
                      DenseTensor* indices) {
  auto inputInfo = getTensorInfo<const T, IndexType>(input);
  auto topKInfo = getTensorInfo<T, IndexType>(*values);
  auto indicesInfo = getTensorInfo<int64_t, IndexType>(*indices);

  // Collapse dims
  inputInfo.sizes[dim] = 1;
  topKInfo.sizes[dim] = 1;
  indicesInfo.sizes[dim] = 1;

  auto strideTopK = topKInfo.strides[dim];
  auto strideIndices = indicesInfo.strides[dim];

  int collapseInputDim = inputInfo.collapseDims(dim);
  int collapseTopKDim = topKInfo.collapseDims(dim);
  int collapseIndicesDim = indicesInfo.collapseDims(dim);

  topKInfo.strides[collapseTopKDim] = strideTopK;
  indicesInfo.strides[collapseIndicesDim] = strideIndices;

  IndexType numInputSlices = 1;
  for (int i = 0; i < inputInfo.dims; ++i) {
    numInputSlices *= inputInfo.sizes[i];
  }

  IndexType sliceSize = input.dims()[dim];

  dim3 grid;
  const int max_grid_dim = 65535;  // Safe limit
  if (numInputSlices <= max_grid_dim) {
    grid.x = numInputSlices;
    grid.y = 1;
    grid.z = 1;
  } else {
    grid.x = max_grid_dim;
    grid.y = (numInputSlices + max_grid_dim - 1) / max_grid_dim;
    grid.z = 1;
  }

  int block_size = 256;  // Default safe block size
  if (sliceSize > 512)
    block_size = 1024;
  else if (sliceSize > 256)
    block_size = 512;

  size_t shm_size = RADIX_SIZE * sizeof(IndexType);
  if (2 * sizeof(T) > shm_size) shm_size = 2 * sizeof(T);
  size_t scan_shm_size = (block_size / 32) * sizeof(int);
  if (scan_shm_size > shm_size) shm_size = scan_shm_size;

  int allDims = inputInfo.dims;
  if (topKInfo.dims != allDims || indicesInfo.dims != allDims) {
    allDims = -1;
  }

#define RUN_DIM(D)                                        \
  gatherTopK<T, IndexType, D>                             \
      <<<grid, block_size, shm_size, dev_ctx.stream()>>>( \
          inputInfo,                                      \
          sliceSize,                                      \
          static_cast<IndexType>(k),                      \
          largest,                                        \
          numInputSlices,                                 \
          inputInfo.strides[collapseInputDim],            \
          topKInfo,                                       \
          topKInfo.strides[collapseTopKDim],              \
          indicesInfo,                                    \
          indicesInfo.strides[collapseIndicesDim]);

  if (allDims == 1) {
    RUN_DIM(1);
  } else if (allDims == 2) {
    RUN_DIM(2);
  } else if (allDims == 3) {
    RUN_DIM(3);
  } else {
    RUN_DIM(-1);
  }
#undef RUN_DIM
}

template <typename T>
void LaunchGatherTopK(const phi::GPUContext& dev_ctx,
                      const DenseTensor& input,
                      int64_t k,
                      int dim,
                      bool largest,
                      DenseTensor* values,
                      DenseTensor* indices) {
  if (input.numel() > 0) {
    LaunchGatherTopK<T, int64_t>(
        dev_ctx, input, k, dim, largest, values, indices);
  }
}

template <typename T>
void SortGatheredTopK(const phi::GPUContext& dev_ctx,
                      DenseTensor* values,
                      DenseTensor* indices,
                      int axis,
                      bool largest) {
  using IndexType = int64_t;  // Default to int64 for safety

  int64_t k = values->dims()[axis];

  // Check if we can use Bitonic Sort (k <= 32)
  if (k <= 32) {
    // Prepare TensorInfo with collapse logic to handle arbitrary axis
    auto keyInfo = getTensorInfo<T, IndexType>(*values);
    auto valueInfo = getTensorInfo<int64_t, IndexType>(*indices);

    // Stash stride
    auto strideKey = keyInfo.strides[axis];
    keyInfo.sizes[axis] = 1;
    int collapseKeyDim = keyInfo.collapseDims(axis);
    keyInfo.strides[collapseKeyDim] = strideKey;

    auto strideValue = valueInfo.strides[axis];
    valueInfo.sizes[axis] = 1;
    int collapseValueDim = valueInfo.collapseDims(axis);
    valueInfo.strides[collapseValueDim] = strideValue;

    IndexType keySlices = 1;
    for (int i = 0; i < keyInfo.dims; ++i) {
      keySlices *= keyInfo.sizes[i];
    }

    // Launch config
    constexpr int sort_size = 32;
    constexpr int max_block_y = 16;
    constexpr int items_per_thread = 2;
    constexpr int block_x = sort_size / items_per_thread;

    const int min_grid = 1;  // Simplify occupancy check
    const auto max_batch = std::max(IndexType{1}, keySlices / min_grid);
    const int block_y = std::min(IndexType(max_block_y), max_batch);
    dim3 block(block_x, block_y);
    dim3 grid((keySlices + block_y - 1) / block_y);

#define RUN_BITONIC_DIM(D)                                  \
  if (largest) {                                            \
    bitonicSortKVInPlace<D, D, block_x, max_block_y>        \
        <<<grid, block, 0, dev_ctx.stream()>>>(keyInfo,     \
                                               keySlices,   \
                                               k,           \
                                               strideKey,   \
                                               valueInfo,   \
                                               strideValue, \
                                               GTOp<T>());  \
  } else {                                                  \
    bitonicSortKVInPlace<D, D, block_x, max_block_y>        \
        <<<grid, block, 0, dev_ctx.stream()>>>(keyInfo,     \
                                               keySlices,   \
                                               k,           \
                                               strideKey,   \
                                               valueInfo,   \
                                               strideValue, \
                                               LTOp<T>());  \
  }

    // Determine dims for IndexToOffset
    // If collapse result dims is 2 (e.g. batch, dim), use 2. Else -1.
    // Actually collapseDims modifies keyInfo.dims.
    if (keyInfo.dims == 2) {
      RUN_BITONIC_DIM(2);
    } else {
      RUN_BITONIC_DIM(-1);
    }
#undef RUN_BITONIC_DIM
    return;
  }

  // Fallback to CUB Radix Sort (Stable) for larger K

  // 1. Prepare for sorting
  DenseTensor* sort_values = values;
  DenseTensor* sort_indices = indices;
  DenseTensor trans_values, trans_indices;

  // If axis is not the last dimension, transpose to make it last
  bool need_transpose = (axis != values->dims().size() - 1);
  std::vector<int> trans_perm;

  if (need_transpose) {
    // Construct permutation: [0, 1, ..., axis-1, axis+1, ..., last, axis]
    for (int i = 0; i < values->dims().size(); ++i) {
      if (i != axis) trans_perm.push_back(i);
    }
    trans_perm.push_back(axis);

    // Resize transposed tensors
    DDim trans_dims = values->dims();
    for (int i = 0; i < trans_perm.size(); ++i) {
      trans_dims[i] = values->dims()[trans_perm[i]];
    }
    trans_values.Resize(trans_dims);
    trans_indices.Resize(trans_dims);
    dev_ctx.template Alloc<T>(&trans_values);
    dev_ctx.template Alloc<int64_t>(&trans_indices);

    // Transpose
    TransCompute<phi::GPUContext, T>(
        values->dims().size(), dev_ctx, *values, &trans_values, trans_perm);
    TransCompute<phi::GPUContext, int64_t>(
        indices->dims().size(), dev_ctx, *indices, &trans_indices, trans_perm);

    sort_values = &trans_values;
    sort_indices = &trans_indices;
  }

  // 2. Perform CUB Segmented Sort
  int64_t num_items = sort_values->numel();
  int64_t num_segments = num_items / k;

  // Create segment offsets
  cub::CountingInputIterator<int64_t> counting_iter(0);
  SegmentOffsetIter segment_offset_iter(k);  // k is stride between segments
  cub::TransformInputIterator<int64_t,
                              SegmentOffsetIter,
                              cub::CountingInputIterator<int64_t>>
      segment_offsets(counting_iter, segment_offset_iter);

  size_t temp_storage_bytes = 0;

  // Allocate temp storage for output
  DenseTensor temp_values_out, temp_indices_out;
  temp_values_out.Resize(sort_values->dims());
  temp_indices_out.Resize(sort_indices->dims());
  T* d_values_out = dev_ctx.template Alloc<T>(&temp_values_out);
  int64_t* d_indices_out = dev_ctx.template Alloc<int64_t>(&temp_indices_out);

  const T* d_keys_in = sort_values->data<T>();
  const int64_t* d_items_in = sort_indices->data<int64_t>();

  // Determine temp storage size
  if (largest) {
    cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr,
                                                       temp_storage_bytes,
                                                       d_keys_in,
                                                       d_values_out,
                                                       d_items_in,
                                                       d_indices_out,
                                                       num_items,
                                                       num_segments,
                                                       segment_offsets,
                                                       segment_offsets + 1,
                                                       0,
                                                       sizeof(T) * 8,
                                                       dev_ctx.stream());
  } else {
    cub::DeviceSegmentedRadixSort::SortPairs(nullptr,
                                             temp_storage_bytes,
                                             d_keys_in,
                                             d_values_out,
                                             d_items_in,
                                             d_indices_out,
                                             num_items,
                                             num_segments,
                                             segment_offsets,
                                             segment_offsets + 1,
                                             0,
                                             sizeof(T) * 8,
                                             dev_ctx.stream());
  }

  DenseTensor temp_storage;
  dev_ctx.template Alloc<uint8_t>(&temp_storage, temp_storage_bytes);

  if (largest) {
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        temp_storage.data<uint8_t>(),
        temp_storage_bytes,
        d_keys_in,
        d_values_out,
        d_items_in,
        d_indices_out,
        num_items,
        num_segments,
        segment_offsets,
        segment_offsets + 1,
        0,
        sizeof(T) * 8,
        dev_ctx.stream());
  } else {
    cub::DeviceSegmentedRadixSort::SortPairs(temp_storage.data<uint8_t>(),
                                             temp_storage_bytes,
                                             d_keys_in,
                                             d_values_out,
                                             d_items_in,
                                             d_indices_out,
                                             num_items,
                                             num_segments,
                                             segment_offsets,
                                             segment_offsets + 1,
                                             0,
                                             sizeof(T) * 8,
                                             dev_ctx.stream());
  }

  // Copy back
  phi::Copy(dev_ctx, temp_values_out, dev_ctx.GetPlace(), false, sort_values);
  phi::Copy(dev_ctx, temp_indices_out, dev_ctx.GetPlace(), false, sort_indices);

  // 3. Transpose back if needed
  if (need_transpose) {
    std::vector<int> inv_perm;
    int ndims = values->dims().size();
    for (int i = 0; i < ndims; ++i) {
      if (i < axis)
        inv_perm.push_back(i);
      else if (i == axis)
        inv_perm.push_back(ndims - 1);
      else
        inv_perm.push_back(i - 1);
    }

    TransCompute<phi::GPUContext, T>(
        ndims, dev_ctx, *sort_values, values, inv_perm);
    TransCompute<phi::GPUContext, int64_t>(
        ndims, dev_ctx, *sort_indices, indices, inv_perm);
  }
}

}  // namespace funcs
}  // namespace phi
