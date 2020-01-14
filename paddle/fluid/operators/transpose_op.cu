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
#include <algorithm>
#include <limits>
#include <utility>

#include "paddle/fluid/operators/transpose_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/platform/gpu_launch_param_config.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

struct EqualTo {
  constexpr bool operator()(int a, int b) const { return a == b; }
};

struct GreaterThan {
  constexpr bool operator()(int a, int b) const { return a > b; }
};

// Value can be decided in compile time.
template <typename FUN, int INT_32 = 32>
constexpr bool CheckProperTileSize(int tile_long, int tile_short, int size_T,
                                   FUN op) {
  return (size_T == 16 && ((tile_long == INT_32 && op(tile_short, 4)) ||
                           (tile_long == 2 * INT_32 && op(tile_short, 4)) ||
                           (tile_long == 4 * INT_32 && op(tile_short, 4)) ||
                           (tile_long == 8 * INT_32 && op(tile_short, 2)))) ||
         (size_T == 8 && ((tile_long == INT_32 && op(tile_short, 15)) ||
                          (tile_long == 2 * INT_32 && op(tile_short, 15)) ||
                          (tile_long == 4 * INT_32 && op(tile_short, 8)) ||
                          (tile_long == 8 * INT_32 && op(tile_short, 4)) ||
                          (tile_long == 16 * INT_32 && op(tile_short, 2)))) ||
         ((size_T == 4 || size_T == 2 || size_T == 1) &&
          ((tile_long == INT_32 && op(tile_short, 15)) ||
           (tile_long == 2 * INT_32 && op(tile_short, 15)) ||
           (tile_long == 4 * INT_32 && op(tile_short, 8)) ||
           (tile_long == 8 * INT_32 && op(tile_short, 4)) ||
           (tile_long == 16 * INT_32 && op(tile_short, 2)) ||
           (tile_long == 16 * INT_32 && op(tile_short, 2))));
}

constexpr bool CheckLongTileSize(int tile_long, int tile_short, int size_T) {
  return CheckProperTileSize(tile_long, tile_short, size_T, EqualTo());
}

constexpr bool CheckOutsideTileSize(int tile_long, int tile_short, int size_T) {
  return CheckProperTileSize(tile_long, tile_short, size_T, GreaterThan());
}

constexpr bool CheckNonLongTileSize(int tile_long, int tile_short, int size_T) {
  return !CheckOutsideTileSize(tile_long, tile_short, size_T) &&
         (CheckOutsideTileSize(tile_long * 2, tile_short, size_T) ||
          CheckOutsideTileSize(tile_long, tile_short + 1, size_T)) &&
         !CheckLongTileSize(tile_long, tile_short, size_T);
}

template <typename T, T DefaultValue>
struct Array3 {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T& operator[](int index) {
    return data[index];
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const T& operator[](int index) const {
    return data[index];
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Array3() {
    for (int i = 0; i < 3; i++) {
      data[i] = DefaultValue;
    }
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Array3(T d0, T d1, T d2) {
    data[0] = d0;
    data[1] = d1;
    data[2] = d2;
  }
  EIGEN_STRONG_INLINE Array3(const std::array<T, 3>& array) {
    for (int i = 0; i < 3; i++) {
      data[i] = array[i];
    }
  }
  T data[3];
};

struct Dim3 : Array3<int, 1> {
  typedef Array3<int, 1> Base;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Dim3() : Base() {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Dim3(int a0, int a1, int a2)
      : Base(a0, a1, a2) {}
  EIGEN_STRONG_INLINE Dim3(const std::array<int, 3>& array) : Base(array) {}
};

struct Index3 : Array3<int, 0> {
  typedef Array3<int, 0> Base;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index3() : Base() {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index3(int a0, int a1, int a2)
      : Base(a0, a1, a2) {}
};

// Flat index with real domension
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int FlatTensorIndex(const Index3& index,
                                                          const Dim3& dims) {
  int flat_index = index[0];
  for (int i = 1; i < 3; i++) {
    flat_index = flat_index * dims[i] + index[i];
  }
  return flat_index;
}

// Convert index to tensor index with dimension..
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index3
ConvertTensorIndex(int index, const Dim3& dims) {
  Index3 tensor_index;
  for (int i = 2; i >= 0; i--) {
    int new_index = index / dims[i];
    tensor_index[i] = index - dims[i] * new_index;
    index = new_index;
  }
  return tensor_index;
}

// Use SM to do data transfer, load a tile into SM then store out.

template <typename T, int NumThreads, int TileX, int TileY>
__global__ void TilingSwapDim1And2(const T* __restrict__ input, Dim3 input_dims,
                                   T* __restrict__ output) {
  assert(blockDim.x == NumThreads);
  assert(blockDim.y == 1);
  assert(blockDim.z == 1);
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);

  constexpr int BlockReadRows = NumThreads / TileY;
  constexpr int BlockWriteRows = NumThreads / TileX;

  // One extra line in the inner dimension to avoid share memory bank conflict.
  __shared__ __align__(
      alignof(T)) char share_mem_ptr[TileX * (TileY + 1) * sizeof(T)];
  typedef T(*ShareMemory)[TileY + 1];

  ShareMemory tile_sm = reinterpret_cast<ShareMemory>(share_mem_ptr);

  int x = threadIdx.x;

  Dim3 output_dims = {
      input_dims[0], input_dims[2], input_dims[1],
  };

  // Align dim to Tiles
  Dim3 tile_aligned_input_dim = {
      input_dims[0], (input_dims[1] + TileX - 1) / TileX,
      (input_dims[2] + TileY - 1) / TileY,
  };

  // Converts block idx to tile index, each block process a tile
  Index3 input_block_tile_index =
      ConvertTensorIndex(blockIdx.x, tile_aligned_input_dim);

  // Compute real index align to tile:0, 32, 64...
  Index3 block_tile_index_in_input = {
      input_block_tile_index[0], input_block_tile_index[1] * TileX,
      input_block_tile_index[2] * TileY,
  };

  // Compute block flat index against input dims.
  int input_origin_block_flat_index =
      FlatTensorIndex(block_tile_index_in_input, input_dims);

  bool full_tile = true;
  int tile_width = TileY;

  // Last row is not full.
  if (input_block_tile_index[2] == tile_aligned_input_dim[2] - 1) {
    tile_width = input_dims[2] - (tile_aligned_input_dim[2] - 1) * TileY;
    full_tile &= false;
  }

  int tile_height = TileX;

  if (input_block_tile_index[1] == tile_aligned_input_dim[1] - 1) {
    tile_height = input_dims[1] - (tile_aligned_input_dim[1] - 1) * TileX;
    full_tile &= false;
  }

  constexpr int in_effective_thread_num = NumThreads / TileY * TileY;

  if (x < in_effective_thread_num) {
    // Read a tile from input using block.
    int x_i = x / TileY;
    int x_j = x % TileY;
    int input_ind = input_origin_block_flat_index + x_i * input_dims[2] + x_j;
    int input_inc = BlockReadRows * input_dims[2];

    if (full_tile) {
#pragma unroll
      for (int ind_i = x_i; ind_i < (TileX); ind_i += BlockReadRows) {
        tile_sm[ind_i][x_j] = input[input_ind];
        input_ind += input_inc;
      }
    } else {
      if (x_j < tile_width) {
        for (int ind_i = x_i; ind_i < (tile_height); ind_i += BlockReadRows) {
          tile_sm[ind_i][x_j] = input[input_ind];
          input_ind += input_inc;
        }
      }
    }
  }

  __syncthreads();

  // Store sm value back to out
  Index3 output_block_tile_index = {
      input_block_tile_index[0], input_block_tile_index[2],
      input_block_tile_index[1],
  };

  Index3 block_tile_index_in_output = {
      output_block_tile_index[0], output_block_tile_index[1] * TileY,
      output_block_tile_index[2] * TileX,
  };

  int output_origin_block_flat_index =
      FlatTensorIndex(block_tile_index_in_output, output_dims);

  constexpr int out_effective_thread_num = NumThreads / TileX * TileX;

  if (x < out_effective_thread_num) {
    int x_i = x / TileX;
    int x_j = x % TileX;
    int output_ind =
        output_origin_block_flat_index + x_i * output_dims[2] + x_j;
    int output_inc = BlockWriteRows * output_dims[2];

    if (full_tile) {
#pragma unroll
      for (int ind_i = x_i; ind_i < (TileY); ind_i += BlockWriteRows) {
        output[output_ind] = tile_sm[x_j][ind_i];
        output_ind += output_inc;
      }
    } else {
      if (x_j < tile_height) {
        for (int ind_i = x_i; ind_i < (tile_width); ind_i += BlockWriteRows) {
          output[output_ind] = tile_sm[x_j][ind_i];
          output_ind += output_inc;
        }
      }
    }
  }
}

template <int TSIZE>
const std::vector<std::pair<int, int>>& SelectProperTileSize() {
  PADDLE_ENFORCE_LE(
      TSIZE, 16,
      "Currently, only data types of sizes 16 bytes or less are supported.");
  PADDLE_ENFORCE_EQ((TSIZE & (TSIZE - 1)), 0,
                    "Data types must have sizes that are powers of 2.");
  auto frontier = std::vector<std::pair<int, int>>();
  const int kMaxLongSideLen = 1024;
  const int kMaxShortSideLen = 15;

  for (int long_side = 32; long_side <= kMaxLongSideLen; long_side *= 2) {
    for (int short_side = 2; short_side <= kMaxShortSideLen; short_side += 1) {
      if (CheckLongTileSize(long_side, short_side, TSIZE)) {
        frontier.push_back(std::make_pair(long_side, short_side));

        if (short_side == 2) return frontier;

        break;
      }
    }
  }
  LOG(FATAL) << "The corresponding short side length of the largest long side "
                "length has to be 2.";
  return frontier;
}

// Get Transpose Type
template <int ElemBytes>
struct TransposeElemType;
template <>
struct TransposeElemType<1> {
  using type = uint8_t;
};
template <>
struct TransposeElemType<2> {
  using type = uint16_t;
};
template <>
struct TransposeElemType<4> {
  using type = uint32_t;
};
template <>
struct TransposeElemType<8> {
  using type = uint64_t;
};
template <>
struct TransposeElemType<16> {
  using type = float4;
};

template <typename IntegralType, bool ceil>
IntegralType CeilOrFloorOfRatio(IntegralType numerator,
                                IntegralType denominator) {
  PADDLE_ENFORCE_NE(0, denominator, "Division by zero is not supported.");

  const IntegralType rounded_toward_zero = numerator / denominator;
  const IntegralType intermediate_product = rounded_toward_zero * denominator;

  if (ceil) {
    const bool needs_adjustment =
        (rounded_toward_zero >= 0) &&
        ((denominator > 0 && numerator > intermediate_product) ||
         (denominator < 0 && numerator < intermediate_product));
    const IntegralType adjustment = static_cast<IntegralType>(needs_adjustment);
    const IntegralType ceil_of_ratio = rounded_toward_zero + adjustment;
    return ceil_of_ratio;
  } else {
    const bool needs_adjustment =
        (rounded_toward_zero <= 0) &&
        ((denominator > 0 && numerator < intermediate_product) ||
         (denominator < 0 && numerator > intermediate_product));
    const IntegralType adjustment = static_cast<IntegralType>(needs_adjustment);
    const IntegralType floor_of_ratio = rounded_toward_zero - adjustment;
    return floor_of_ratio;
  }
}

template <typename T, int tile_long, int tile_short>
void LaunchNarrowDims2TransposeKernel(const platform::CUDADeviceContext& d,
                                      int tile_size_i, int tile_size_j,
                                      int total_tiles_count, const T* input,
                                      const Dim3& input_dims, T* output) {
  constexpr int NumThreads = tile_long;
  if (tile_size_i <= tile_long && tile_size_j <= tile_short) {
    TilingSwapDim1And2<
        T, NumThreads, tile_long,
        tile_short><<<total_tiles_count, NumThreads, 0, d.stream()>>>(
        input, input_dims, output);
  } else {
    TilingSwapDim1And2<
        T, NumThreads, tile_short,
        tile_long><<<total_tiles_count, NumThreads, 0, d.stream()>>>(
        input, input_dims, output);
  }
}

template <typename T, int tile_long, int tile_short, typename dummy = void>
struct NarrowDims2TransposeDispatch {
  static void DoIt(const platform::CUDADeviceContext& d, int tile_size_i,
                   int tile_size_j, int total_tiles_count, const T* input,
                   const Dim3& input_dims, T* output) {
    PADDLE_ENFORCE_EQ(
        (tile_long & (tile_long - 1)), 0,
        "The length of the longer side of the tile is always a power of 2.");
    bool request_satisfied = std::max(tile_size_i, tile_size_j) <= tile_long &&
                             std::min(tile_size_i, tile_size_j) <= tile_short;

    if (request_satisfied) {
      LaunchNarrowDims2TransposeKernel<T, tile_long, tile_short>(
          d, tile_size_i, tile_size_j, total_tiles_count, input, input_dims,
          output);
      return;
    }

    const bool long_side_request_not_satisfied =
        std::max(tile_size_i, tile_size_j) > tile_long;

    if (long_side_request_not_satisfied) {
      NarrowDims2TransposeDispatch<T, tile_long * 2, tile_short>::DoIt(
          d, tile_size_i, tile_size_j, total_tiles_count, input, input_dims,
          output);
    } else {
      NarrowDims2TransposeDispatch<T, tile_long, tile_short + 1>::DoIt(
          d, tile_size_i, tile_size_j, total_tiles_count, input, input_dims,
          output);
    }
  }
};

template <typename T, int tile_long, int tile_short>
struct NarrowDims2TransposeDispatch<
    T, tile_long, tile_short,
    typename std::enable_if<
        CheckNonLongTileSize(tile_long, tile_short, sizeof(T)), void>::type> {
  static void DoIt(const platform::CUDADeviceContext& d, int tile_size_i,
                   int tile_size_j, int total_tiles_count, const T* input,
                   const Dim3& input_dims, T* output) {
    PADDLE_ENFORCE_EQ(
        (tile_long & (tile_long - 1)), 0,
        "The length of the longer side of the tile is always a power of 2.");
    bool request_satisfied = std::max(tile_size_i, tile_size_j) <= tile_long &&
                             std::min(tile_size_i, tile_size_j) <= tile_short;

    if (request_satisfied) {
      LaunchNarrowDims2TransposeKernel<T, tile_long, tile_short>(
          d, tile_size_i, tile_size_j, total_tiles_count, input, input_dims,
          output);
      return;
    }

    NarrowDims2TransposeDispatch<T, tile_long, tile_short + 1>::DoIt(
        d, tile_size_i, tile_size_j, total_tiles_count, input, input_dims,
        output);
  }
};

template <typename T, int tile_long, int tile_short>
struct NarrowDims2TransposeDispatch<
    T, tile_long, tile_short,
    typename std::enable_if<CheckLongTileSize(tile_long, tile_short, sizeof(T)),
                            void>::type> {
  static void DoIt(const platform::CUDADeviceContext& d, int tile_size_i,
                   int tile_size_j, int total_tiles_count, const T* input,
                   const Dim3& input_dims, T* output) {
    PADDLE_ENFORCE_EQ(
        (tile_long & (tile_long - 1)), 0,
        "The length of the longer side of the tile is always a power of 2.");

    LaunchNarrowDims2TransposeKernel<T, tile_long, tile_short>(
        d, tile_size_i, tile_size_j, total_tiles_count, input, input_dims,
        output);
  }
};

template <typename T, bool conjugate = false>
void SwapDim1And2InNarrow(const platform::CUDADeviceContext& d, const T* input,
                          const Dim3& input_dims, T* output,
                          const int kMinTileSize) {
  // Get available tile sizes here for the data type requested:
  const auto& tile_spec = SelectProperTileSize<sizeof(T)>();

  int tile_long_edge = 0;
  int tile_short_edge = 0;
  float lowest_cost = std::numeric_limits<float>::max();
  int input_long_edge = std::max(input_dims[1], input_dims[2]);

  for (auto tile_size_pair : tile_spec) {
    int proposed_tile_long_edge = tile_size_pair.first;
    // data may not aligned to tile, so some threads wasted
    int num_wasted_threads = input_long_edge -
                             CeilOrFloorOfRatio<int, false>(
                                 input_long_edge, proposed_tile_long_edge) *
                                 proposed_tile_long_edge;

    int num_full_tiles = CeilOrFloorOfRatio<int, false>(
        input_long_edge, proposed_tile_long_edge);

    float cost = 0;

    if (num_full_tiles <= 1) cost = num_wasted_threads;

    // Find least weasted threads.
    if (cost <= lowest_cost) {
      tile_long_edge = proposed_tile_long_edge;
      tile_short_edge = tile_size_pair.second;
      lowest_cost = cost;
    }
  }

  // The tile size we select should be match with input dim, long side to long
  // short side to short.
  // First set long side  as i if dim1 > Tile min size, then set dim2 as j.
  int select_tile_size_i =
      input_dims[1] >= kMinTileSize ? tile_long_edge : input_dims[1];
  int select_tile_size_j =
      input_dims[1] >= kMinTileSize ? input_dims[2] : tile_long_edge;

  // Check if i is long edge, if not set i as short.
  select_tile_size_i = select_tile_size_i == tile_long_edge
                           ? tile_long_edge
                           : std::min(select_tile_size_i, tile_short_edge);

  // Check if j is long edge, if not set j as short.
  select_tile_size_j = select_tile_size_j == tile_long_edge
                           ? tile_long_edge
                           : std::min(select_tile_size_j, tile_short_edge);

  Dim3 input_dims_aligned = {
      input_dims[0],
      CeilOrFloorOfRatio<int, true>(input_dims[1], select_tile_size_i),
      CeilOrFloorOfRatio<int, true>(input_dims[2], select_tile_size_j),
  };

  int total_tiles_count =
      input_dims_aligned[0] * input_dims_aligned[1] * input_dims_aligned[2];

  using ElemType = typename TransposeElemType<sizeof(T)>::type;
  static_assert(alignof(T) >= alignof(ElemType), "Unexpected data alignment.");
  NarrowDims2TransposeDispatch<ElemType, 32, 2>::DoIt(
      d, select_tile_size_i, select_tile_size_j, total_tiles_count,
      reinterpret_cast<const ElemType*>(input), input_dims,
      reinterpret_cast<ElemType*>(output));
}

template <typename T, int pos0, int pos1, int pos2>
__global__ void TransposeSimpleKernel(int nthreads, const T* __restrict__ input,
                                      Dim3 input_dims, T* __restrict__ output) {
  Dim3 output_dims;
  output_dims[pos0] = input_dims[0];
  output_dims[pos1] = input_dims[1];
  output_dims[pos2] = input_dims[2];

  CUDA_1D_KERNEL_LOOP(output_index, nthreads) {
    Index3 output_tensor_index = ConvertTensorIndex(output_index, output_dims);

    Index3 input_tensor_index;
    input_tensor_index[0] = output_tensor_index[pos0];
    input_tensor_index[1] = output_tensor_index[pos1];
    input_tensor_index[2] = output_tensor_index[pos2];

    int input_index = FlatTensorIndex(input_tensor_index, input_dims);

    output[output_index] = input[input_index];
  }
}

// Here suppose convert all tensor to dim3, so just change dim1 and 2.
template <typename T>
void RunSwapDim1And2InTranspose(const platform::CUDADeviceContext& d,
                                const T* input, const Dim3 input_dims,
                                T* output) {
  // Suppose tile size > 16
  static const int kMinTileSize = 16;
  static const int kMinNarrowTileSize = 96;

  bool large_tile =
      input_dims[1] >= kMinTileSize && input_dims[2] >= kMinTileSize;
  bool narrow_tile = input_dims[1] >= kMinNarrowTileSize ||
                     input_dims[2] >= kMinNarrowTileSize;
  if (large_tile) {
    // suppose 32 X 32 gives best performance, and 8 warp in block.
    constexpr int kTileSize = 32;
    constexpr int kNumThreads = 256;

    Dim3 input_dims_aligned = {
        input_dims[0], CeilOrFloorOfRatio<int, true>(input_dims[1], kTileSize),
        CeilOrFloorOfRatio<int, true>(input_dims[2], kTileSize),
    };

    int total_tiles_count =
        input_dims_aligned[0] * input_dims_aligned[1] * input_dims_aligned[2];

    TilingSwapDim1And2<
        T, kNumThreads, kTileSize,
        kTileSize><<<total_tiles_count, kNumThreads, 0, d.stream()>>>(
        input, input_dims, output);

  } else if (narrow_tile) {
    SwapDim1And2InNarrow<T>(d, input, input_dims, output, kMinTileSize);
  } else {
    int total_elements = input_dims[0] * input_dims[1] * input_dims[2];
    auto config = GetGpuLaunchConfig1D(d, total_elements);
    TransposeSimpleKernel<T, 0, 2, 1><<<
        config.block_per_grid.x, config.thread_per_block.x, 0, d.stream()>>>(
        total_elements, input, input_dims, output);
  }
}

template <typename T>
struct SwapDim1And2InTranspose {
  typedef platform::CUDADeviceContext Device;
  void operator()(const Device& d, const T* in,
                  const std::vector<int>& combined_dims, T* out) {
    Dim3 input_dims = {static_cast<int>(combined_dims[0]),
                       static_cast<int>(combined_dims[1]),
                       static_cast<int>(combined_dims[2])};
    RunSwapDim1And2InTranspose<T>(d, in, input_dims, out);
  }
};

template <typename T>
struct SwapDim0And2InTranspose {
  typedef platform::CUDADeviceContext Device;
  void operator()(const Device& d, const T* in,
                  const std::vector<int>& combined_dims, T* out) {
    Dim3 input_dims = {static_cast<int>(combined_dims[0]),
                       static_cast<int>(combined_dims[1]),
                       static_cast<int>(combined_dims[2])};

    size_t total_size = combined_dims[0] * combined_dims[1] * combined_dims[2];
    auto config = GetGpuLaunchConfig1D(d, total_size);

    TransposeSimpleKernel<T, 2, 1, 0><<<
        config.block_per_grid.x, config.thread_per_block.x, 0, d.stream()>>>(
        total_size, in, input_dims, out);
  }
};

inline void CombineTransposeDim3(const framework::DDim& shape,
                                 const std::vector<int>& perm,
                                 std::vector<int>* new_perm,
                                 framework::DDim* new_dims) {
  PADDLE_ENFORCE_EQ(shape.size(), perm.size(),
                    " shape should have the save dim with");
  std::vector<int> dim_vec;
  if (shape.size() == 1) {
    // If input dimension is already 1, n<<dim_idx;o need to reduce dimension.
    new_perm->resize(1);
    (*new_perm)[0] = perm[0];
    dim_vec.push_back(shape[0]);
    *new_dims = framework::make_ddim(dim_vec);
    return;
  }
  std::vector<int> new_dim_pos(shape.size(), -1);
  std::vector<int> combined_dims(shape.size(), 0);
  int cur_head = perm[0];
  new_dim_pos[cur_head] = 0;
  combined_dims[0] = shape[cur_head];
  int dim_idx = 0;
  for (int perm_idx = 1; perm_idx < shape.size(); ++perm_idx) {
    // combine consecutive dimensions.
    if (cur_head + 1 == perm[perm_idx]) {
      cur_head = perm[perm_idx];
      combined_dims[dim_idx] *= shape[cur_head];
    } else {
      // Else start a new dimension.
      cur_head = perm[perm_idx];
      dim_idx++;
      new_dim_pos[cur_head] = dim_idx;
      combined_dims[dim_idx] = shape[cur_head];
    }
  }

  new_perm->resize(dim_idx + 1);

  dim_idx = 0;
  for (int i = 0; i < new_dim_pos.size(); ++i) {
    if (new_dim_pos[i] >= 0) {
      int new_perm_idx = new_dim_pos[i];
      (*new_perm)[dim_idx] = new_perm_idx;
      dim_vec.push_back(combined_dims[new_perm_idx]);
      dim_idx++;
    }
  }

  *new_dims = framework::make_ddim(dim_vec);
}

template <typename T>
struct TransposeSimple {
  static bool run(const platform::CUDADeviceContext& ctx, const Tensor& in,
                  const std::vector<int32_t> perm, Tensor* out) {
    // First try to reduce the dimensions of the input tensor.
    std::vector<int> new_perm;
    framework::DDim new_dims;
    CombineTransposeDim3(in.dims(), perm, &new_perm, &new_dims);

    // Only use special GPU kernel when dimension is 2 or 3.
    int dims = new_dims.size();
    std::vector<int> new_dim_vec = framework::vectorize<int>(new_dims);
    if (dims < 2 || dims > 3) return false;
    auto in_data = in.data<T>();
    auto out_data = out->data<T>();

    switch (dims) {
      case 2:
        if (new_perm[0] == 1 && new_perm[1] == 0) {
          // Add the first dimension size as 1.
          new_dim_vec.insert(new_dim_vec.begin(), 1);
          SwapDim1And2InTranspose<T>()(ctx, in_data, new_dim_vec, out_data);
          return true;
        }
        break;
      case 3:
        if (new_perm == std::vector<int>({0, 2, 1})) {
          SwapDim1And2InTranspose<T>()(ctx, in_data, new_dim_vec, out_data);
          return true;
        } else if (new_perm == std::vector<int>({2, 1, 0})) {
          SwapDim0And2InTranspose<T>()(ctx, in_data, new_dim_vec, out_data);
          return true;
        } else {
          return false;
        }
        break;
      default:
        return false;
    }
    return false;
  }
};

template <typename DeviceContext, typename T>
class TransposeGPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<framework::Tensor>("X");
    auto* out = context.Output<framework::Tensor>("Out");
    out->mutable_data<T>(context.GetPlace());

    std::vector<int> axis = context.Attr<std::vector<int>>("axis");
    int ndims = axis.size();
    const auto& dev_ctx = context.template device_context<DeviceContext>();
    auto ret = TransposeSimple<T>::run(dev_ctx, *x, axis, out);
    if (!ret) {
      TransCompute<DeviceContext, T>(ndims, dev_ctx, *x, out, axis);
    }
  }
};
template <typename DeviceContext, typename T>
class TransposeGradGPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* out_grad =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* x_grad =
        context.Output<framework::Tensor>(framework::GradVarName("X"));
    if (!x_grad) return;

    x_grad->mutable_data<T>(context.GetPlace());
    std::vector<int> axis = context.Attr<std::vector<int>>("axis");
    std::vector<int> reversed_axis(axis);

    for (size_t i = 0; i < axis.size(); i++) {
      reversed_axis[axis[i]] = i;
    }

    int ndims = axis.size();
    const auto& dev_ctx = context.template device_context<DeviceContext>();
    auto ret =
        TransposeSimple<T>::run(dev_ctx, *out_grad, reversed_axis, x_grad);
    if (!ret) {
      TransCompute<DeviceContext, T>(ndims, dev_ctx, *out_grad, x_grad,
                                     reversed_axis);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(
    transpose,
    ops::TransposeGPUKernel<paddle::platform::CUDADeviceContext, float>,
    ops::TransposeGPUKernel<paddle::platform::CUDADeviceContext, double>,
    ops::TransposeGPUKernel<paddle::platform::CUDADeviceContext,
                            plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    transpose_grad,
    ops::TransposeGradGPUKernel<paddle::platform::CUDADeviceContext, float>,
    ops::TransposeGradGPUKernel<paddle::platform::CUDADeviceContext, double>,
    ops::TransposeGradGPUKernel<paddle::platform::CUDADeviceContext,
                                plat::float16>);

REGISTER_OP_CUDA_KERNEL(
    transpose2,
    ops::TransposeGPUKernel<paddle::platform::CUDADeviceContext, int32_t>,
    ops::TransposeGPUKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::TransposeGPUKernel<paddle::platform::CUDADeviceContext, float>,
    ops::TransposeGPUKernel<paddle::platform::CUDADeviceContext, double>,
    ops::TransposeGPUKernel<paddle::platform::CUDADeviceContext,
                            plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    transpose2_grad,
    ops::TransposeGradGPUKernel<paddle::platform::CUDADeviceContext, int32_t>,
    ops::TransposeGradGPUKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::TransposeGradGPUKernel<paddle::platform::CUDADeviceContext, float>,
    ops::TransposeGradGPUKernel<paddle::platform::CUDADeviceContext, double>,
    ops::TransposeGradGPUKernel<paddle::platform::CUDADeviceContext,
                                plat::float16>);
