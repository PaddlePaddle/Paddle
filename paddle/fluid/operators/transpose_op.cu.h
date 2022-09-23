/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/framework/gpu_utils.h"
#include "paddle/fluid/operators/transpose_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/fluid/platform/fast_divmod.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/autotune/auto_tune_base.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using Dim3 = framework::Dim3;
using Index3 = framework::Index3;

struct EqualTo {
  constexpr bool operator()(int a, int b) const { return a == b; }
};

struct GreaterThan {
  constexpr bool operator()(int a, int b) const { return a > b; }
};

// Value can be decided in compile time.
template <typename FUN, int INT_32 = 32>
constexpr bool CheckProperTileSize(int tile_long,
                                   int tile_short,
                                   int size_T,
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

// Use SM to do data transfer, load a tile into SM then store out.
// All tile read and write are colascing, so can speedup memory copy
template <typename T,
          int NumThreads,
          int TileX,
          int TileY,
          typename IndexType = int>
__global__ void TilingSwapDim1And2(const T* __restrict__ input,
                                   Dim3 input_dims,
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
      input_dims[0],
      input_dims[2],
      input_dims[1],
  };

  // Align dim to Tiles
  Dim3 tile_aligned_input_dim = {
      input_dims[0],
      (input_dims[1] + TileX - 1) / TileX,
      (input_dims[2] + TileY - 1) / TileY,
  };

  // Converts block idx to tile index, each block process a tile
  Index3 input_block_tile_index = framework::ConvertTensorIndex<IndexType>(
      blockIdx.x, tile_aligned_input_dim);

  // Compute real index align to tile:0, 32, 64...
  Index3 block_tile_index_in_input = {
      input_block_tile_index[0],
      input_block_tile_index[1] * TileX,
      input_block_tile_index[2] * TileY,
  };

  // Compute block flat index against input dims.
  IndexType input_origin_block_flat_index =
      framework::FlatTensorIndex<IndexType>(block_tile_index_in_input,
                                            input_dims);

  bool full_tile = true;
  IndexType tile_width = TileY;

  // Last row is not full.
  if (input_block_tile_index[2] == tile_aligned_input_dim[2] - 1) {
    tile_width = input_dims[2] - (tile_aligned_input_dim[2] - 1) * TileY;
    full_tile &= false;
  }

  IndexType tile_height = TileX;

  if (input_block_tile_index[1] == tile_aligned_input_dim[1] - 1) {
    tile_height = input_dims[1] - (tile_aligned_input_dim[1] - 1) * TileX;
    full_tile &= false;
  }

  constexpr IndexType in_effective_thread_num = NumThreads / TileY * TileY;

  if (x < in_effective_thread_num) {
    // Read a tile from input using block.
    int x_i = x / TileY;
    int x_j = x % TileY;
    IndexType input_ind =
        input_origin_block_flat_index + x_i * input_dims[2] + x_j;
    IndexType input_inc = BlockReadRows * input_dims[2];

    if (full_tile) {
#pragma unroll
      for (int ind_i = x_i; ind_i < (TileX); ind_i += BlockReadRows) {
        tile_sm[ind_i][x_j] = input[input_ind];
        input_ind += input_inc;
      }
    } else {
      if (x_j < tile_width) {
#pragma unroll
        for (IndexType ind_i = x_i; ind_i < (tile_height);
             ind_i += BlockReadRows) {
          tile_sm[ind_i][x_j] = input[input_ind];
          input_ind += input_inc;
        }
      }
    }
  }

  __syncthreads();

  // Store sm value back to out
  Index3 output_block_tile_index = {
      input_block_tile_index[0],
      input_block_tile_index[2],
      input_block_tile_index[1],
  };

  Index3 block_tile_index_in_output = {
      output_block_tile_index[0],
      output_block_tile_index[1] * TileY,
      output_block_tile_index[2] * TileX,
  };

  IndexType output_origin_block_flat_index =
      framework::FlatTensorIndex<IndexType>(block_tile_index_in_output,
                                            output_dims);

  constexpr IndexType out_effective_thread_num = NumThreads / TileX * TileX;

  if (x < out_effective_thread_num) {
    int x_i = x / TileX;
    int x_j = x % TileX;
    IndexType output_ind =
        output_origin_block_flat_index + x_i * output_dims[2] + x_j;
    IndexType output_inc = BlockWriteRows * output_dims[2];

    if (full_tile) {
#pragma unroll
      for (int ind_i = x_i; ind_i < (TileY); ind_i += BlockWriteRows) {
        output[output_ind] = tile_sm[x_j][ind_i];
        output_ind += output_inc;
      }
    } else {
      if (x_j < tile_height) {
#pragma unroll
        for (IndexType ind_i = x_i; ind_i < (tile_width);
             ind_i += BlockWriteRows) {
          output[output_ind] = tile_sm[x_j][ind_i];
          output_ind += output_inc;
        }
      }
    }
  }
}

// This function will find combination of long_side X short_side in backups
template <int TSIZE>
bool SelectProperTileSize(std::vector<std::pair<int, int>>* tiles) {
  PADDLE_ENFORCE_LE(
      TSIZE,
      16,
      platform::errors::InvalidArgument(
          "The tile size should smaller than 16, but received is:%d.", TSIZE));

  PADDLE_ENFORCE_EQ(
      (TSIZE & (TSIZE - 1)),
      0,
      platform::errors::InvalidArgument(
          "Data types should be powers of 2, but reived size is:%d.", TSIZE));

  const int kMaxLongSideLen = 1024;
  const int kMaxShortSideLen = 15;

  for (int long_side = 32; long_side <= kMaxLongSideLen; long_side *= 2) {
    for (int short_side = 2; short_side <= kMaxShortSideLen; short_side += 1) {
      if (CheckLongTileSize(long_side, short_side, TSIZE)) {
        tiles->push_back(std::make_pair(long_side, short_side));

        if (short_side == 2) return true;

        break;
      }
    }
  }
  return false;
}

// Use system built in type
template <int ByteSize>
struct SystemElemType;
template <>
struct SystemElemType<1> {
  using type = uint8_t;
};
template <>
struct SystemElemType<2> {
  using type = uint16_t;
};
template <>
struct SystemElemType<4> {
  using type = uint32_t;
};
template <>
struct SystemElemType<8> {
  using type = uint64_t;
};
template <>
struct SystemElemType<16> {
  using type = float4;
};

template <typename T, int tile_long, int tile_short, typename IndexType = int>
void LaunchNarrowDims2TransposeKernel(const phi::GPUContext& d,
                                      int tile_size_i,
                                      int tile_size_j,
                                      IndexType total_tiles_count,
                                      const T* input,
                                      const Dim3& input_dims,
                                      T* output) {
  constexpr int NumThreads = tile_long;
  if (tile_size_i <= tile_long && tile_size_j <= tile_short) {
    TilingSwapDim1And2<T, NumThreads, tile_long, tile_short, IndexType>
        <<<total_tiles_count, NumThreads, 0, d.stream()>>>(
            input, input_dims, output);
  } else {
    TilingSwapDim1And2<T, NumThreads, tile_short, tile_long, IndexType>
        <<<total_tiles_count, NumThreads, 0, d.stream()>>>(
            input, input_dims, output);
  }
}

template <typename T,
          int tile_long,
          int tile_short,
          typename IndexType = int,
          typename dummy = void>
struct NarrowDims2TransposeDispatch {
  static void DoTranspose(const phi::GPUContext& d,
                          int tile_size_i,
                          int tile_size_j,
                          IndexType total_tiles_count,
                          const T* input,
                          const Dim3& input_dims,
                          T* output) {
    PADDLE_ENFORCE_EQ(
        (tile_long & (tile_long - 1)),
        0,
        platform::errors::InvalidArgument(
            "The length of the longer side of the tile should be power of 2."
            " But received value is:%d.",
            tile_long));

    bool request_satisfied = std::max(tile_size_i, tile_size_j) <= tile_long &&
                             std::min(tile_size_i, tile_size_j) <= tile_short;

    if (request_satisfied) {
      LaunchNarrowDims2TransposeKernel<T, tile_long, tile_short, IndexType>(
          d,
          tile_size_i,
          tile_size_j,
          total_tiles_count,
          input,
          input_dims,
          output);
      return;
    }

    const bool long_side_request_not_satisfied =
        std::max(tile_size_i, tile_size_j) > tile_long;

    if (long_side_request_not_satisfied) {
      NarrowDims2TransposeDispatch<T, tile_long * 2, tile_short, IndexType>::
          DoTranspose(d,
                      tile_size_i,
                      tile_size_j,
                      total_tiles_count,
                      input,
                      input_dims,
                      output);
    } else {
      NarrowDims2TransposeDispatch<T, tile_long, tile_short + 1, IndexType>::
          DoTranspose(d,
                      tile_size_i,
                      tile_size_j,
                      total_tiles_count,
                      input,
                      input_dims,
                      output);
    }
  }
};

// If Not long tile size, goto this function when compile.
template <typename T, int tile_long, int tile_short, typename IndexType>
struct NarrowDims2TransposeDispatch<
    T,
    tile_long,
    tile_short,
    IndexType,
    typename std::enable_if<CheckNonLongTileSize(
                                tile_long, tile_short, sizeof(T)),
                            void>::type> {
  static void DoTranspose(const phi::GPUContext& d,
                          int tile_size_i,
                          int tile_size_j,
                          IndexType total_tiles_count,
                          const T* input,
                          const Dim3& input_dims,
                          T* output) {
    PADDLE_ENFORCE_EQ(
        (tile_long & (tile_long - 1)),
        0,
        platform::errors::InvalidArgument(
            "The length of the longer side of the tile should be power of 2."
            " But received value is:%d.",
            tile_long));

    bool request_satisfied = std::max(tile_size_i, tile_size_j) <= tile_long &&
                             std::min(tile_size_i, tile_size_j) <= tile_short;

    if (request_satisfied) {
      LaunchNarrowDims2TransposeKernel<T, tile_long, tile_short, IndexType>(
          d,
          tile_size_i,
          tile_size_j,
          total_tiles_count,
          input,
          input_dims,
          output);
      return;
    }

    NarrowDims2TransposeDispatch<T, tile_long, tile_short + 1, IndexType>::
        DoTranspose(d,
                    tile_size_i,
                    tile_size_j,
                    total_tiles_count,
                    input,
                    input_dims,
                    output);
  }
};

// If long tile size, goto this function when compile.
template <typename T, int tile_long, int tile_short, typename IndexType>
struct NarrowDims2TransposeDispatch<
    T,
    tile_long,
    tile_short,
    IndexType,
    typename std::enable_if<CheckLongTileSize(tile_long, tile_short, sizeof(T)),
                            void>::type> {
  static void DoTranspose(const phi::GPUContext& d,
                          int tile_size_i,
                          int tile_size_j,
                          IndexType total_tiles_count,
                          const T* input,
                          const Dim3& input_dims,
                          T* output) {
    PADDLE_ENFORCE_EQ(
        (tile_long & (tile_long - 1)),
        0,
        platform::errors::InvalidArgument(
            "The length of the longer side of the tile should be power of 2,"
            " but received is:%d.",
            tile_long));

    LaunchNarrowDims2TransposeKernel<T, tile_long, tile_short, IndexType>(
        d,
        tile_size_i,
        tile_size_j,
        total_tiles_count,
        input,
        input_dims,
        output);
  }
};

template <typename T, bool conjugate = false, typename IndexType = int>
void SwapDim1And2InNarrow(const phi::GPUContext& d,
                          const T* input,
                          const Dim3& input_dims,
                          T* output,
                          const int kMinTileSize) {
  // First get available tile sizes for the data type requested as backups
  std::vector<std::pair<int, int>> tile_sele;
  auto ret = SelectProperTileSize<sizeof(T)>(&tile_sele);
  PADDLE_ENFORCE_EQ(
      ret,
      true,
      platform::errors::InvalidArgument(
          "SelectProperTileSize should return true, but return value is:%d.",
          ret));

  int tile_long_edge = 0;
  int tile_short_edge = 0;
  float lowest_cost = std::numeric_limits<float>::max();
  int input_long_edge = std::max(input_dims[1], input_dims[2]);

  // Find the tile size that best suit in  inputs.
  for (auto tile_size_pair : tile_sele) {
    int proposed_tile_long_edge = tile_size_pair.first;
    // data may not aligned to tile, so some threads wasted, we need
    // to find least wasted threads, which means we need to find tile
    // can split input properly, in another words: num_wasted_threads=0.
    int num_wasted_threads =
        input_long_edge - framework::CeilOrFloor<int, false>(
                              input_long_edge, proposed_tile_long_edge) *
                              proposed_tile_long_edge;

    int num_full_tiles = framework::CeilOrFloor<int, false>(
        input_long_edge, proposed_tile_long_edge);

    float cost = num_wasted_threads;

    if (cost <= lowest_cost) {
      tile_long_edge = proposed_tile_long_edge;
      tile_short_edge = tile_size_pair.second;
      lowest_cost = cost;
    }
    // break as we already find best tile size.
    if (cost == 0) break;
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

  // Here finally get proper long X short tile size.
  Dim3 input_dims_aligned = {
      input_dims[0],
      framework::CeilOrFloor<int, true>(input_dims[1], select_tile_size_i),
      framework::CeilOrFloor<int, true>(input_dims[2], select_tile_size_j),
  };

  IndexType total_tiles_count = input_dims_aligned[0];
  total_tiles_count *= input_dims_aligned[1];
  total_tiles_count *= input_dims_aligned[2];

  // Suppose T can be replaced by system builtin types
  using ElemType = typename SystemElemType<sizeof(T)>::type;

  NarrowDims2TransposeDispatch<ElemType, 32, 2, IndexType>::DoTranspose(
      d,
      select_tile_size_i,
      select_tile_size_j,
      total_tiles_count,
      reinterpret_cast<const ElemType*>(input),
      input_dims,
      reinterpret_cast<ElemType*>(output));
}

// This is for case that cannot do coalescing read and write.
// Or input is too small to split into tiles.
template <typename T, int pos0, int pos1, int pos2, typename IndexType = int>
__global__ void TransposeSimpleKernel(IndexType nthreads,
                                      const T* __restrict__ input,
                                      Dim3 input_dims,
                                      T* __restrict__ output) {
  Dim3 output_dims;
  output_dims[pos0] = input_dims[0];
  output_dims[pos1] = input_dims[1];
  output_dims[pos2] = input_dims[2];

  CUDA_KERNEL_LOOP_TYPE(output_index, nthreads, IndexType) {
    Index3 output_tensor_index =
        framework::ConvertTensorIndex<IndexType>(output_index, output_dims);

    Index3 input_tensor_index;
    input_tensor_index[0] = output_tensor_index[pos0];
    input_tensor_index[1] = output_tensor_index[pos1];
    input_tensor_index[2] = output_tensor_index[pos2];

    IndexType input_index =
        framework::FlatTensorIndex<IndexType>(input_tensor_index, input_dims);

    output[output_index] = input[input_index];
  }
}

// Here suppose convert all tensor to dim3, so just change dim1 and 2.
template <typename T, typename IndexType = int>
void SendSwapDim1And2InTranspose(const phi::GPUContext& d,
                                 const T* input,
                                 const Dim3& input_dims,
                                 T* output) {
  // Suppose tile size > 16
  static const int kMinTileSize = 16;
  static const int kMinNarrowTileSize = 96;

  bool large_tile =
      input_dims[1] >= kMinTileSize && input_dims[2] >= kMinTileSize;
  bool narrow_tile = input_dims[1] >= kMinNarrowTileSize ||
                     input_dims[2] >= kMinNarrowTileSize;
  if (large_tile) {
    // If input is large square, such as 32X32, use SM to do copy.
    // suppose 32 X 32 gives best performance, and 8 warp in block.
    constexpr int kTileSize = 32;
    constexpr int kNumThreads = 256;

    Dim3 input_dims_aligned = {
        input_dims[0],
        framework::CeilOrFloor<int, true>(input_dims[1], kTileSize),
        framework::CeilOrFloor<int, true>(input_dims[2], kTileSize),
    };

    IndexType total_tiles_count = input_dims_aligned[0];
    total_tiles_count *= input_dims_aligned[1];
    total_tiles_count *= input_dims_aligned[2];

    TilingSwapDim1And2<T, kNumThreads, kTileSize, kTileSize, IndexType>
        <<<total_tiles_count, kNumThreads, 0, d.stream()>>>(
            input, input_dims, output);

  } else if (narrow_tile) {
    // If input shape is like Rect, such as 2X100, use Narrow tile size.
    // It makes things complicated, because need to find a tile can coverr
    // input and also reach best coalescing.
    SwapDim1And2InNarrow<T, false, IndexType>(
        d, input, input_dims, output, kMinTileSize);
  } else {
    // If input shape is small, such as 8X8, just do simple copy
    IndexType total_elements = input_dims[0];
    total_elements *= input_dims[1];
    total_elements *= input_dims[2];
    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(d, total_elements);
    TransposeSimpleKernel<T, 0, 2, 1, IndexType>
        <<<config.block_per_grid.x, config.thread_per_block.x, 0, d.stream()>>>(
            total_elements, input, input_dims, output);
  }
}

template <typename T, typename IndexType = int>
struct SwapDim1And2InTranspose {
  typedef phi::GPUContext Device;
  void operator()(const Device& d,
                  const T* in,
                  const std::vector<int>& combined_dims,
                  T* out) {
    Dim3 input_dims = {static_cast<int>(combined_dims[0]),
                       static_cast<int>(combined_dims[1]),
                       static_cast<int>(combined_dims[2])};
    SendSwapDim1And2InTranspose<T, IndexType>(d, in, input_dims, out);
  }
};

template <typename T, typename IndexType = int>
struct SwapDim0And2InTranspose {
  typedef phi::GPUContext Device;
  void operator()(const Device& d,
                  const T* in,
                  const std::vector<int>& combined_dims,
                  T* out) {
    Dim3 input_dims = {static_cast<int>(combined_dims[0]),
                       static_cast<int>(combined_dims[1]),
                       static_cast<int>(combined_dims[2])};

    IndexType total_size = combined_dims[0];
    total_size *= combined_dims[1];
    total_size *= combined_dims[2];
    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(d, total_size);

    TransposeSimpleKernel<T, 2, 1, 0, IndexType>
        <<<config.block_per_grid.x, config.thread_per_block.x, 0, d.stream()>>>(
            total_size, in, input_dims, out);
  }
};

// This function is to combine dimension. fox example:
// (0, 1, 3, 2) --> (0, 2, 1)
inline void CombineTransposeDim3(const framework::DDim& shape,
                                 const std::vector<int>& perm,
                                 std::vector<int>* new_perm,
                                 framework::DDim* new_dims) {
  PADDLE_ENFORCE_EQ(shape.size(),
                    perm.size(),
                    platform::errors::InvalidArgument(
                        " shape should have the save dim with perm, but"
                        " received shape size is:%d, perm size is:%d.",
                        shape.size(),
                        perm.size()));

  std::vector<int> dim_vec;
  if (shape.size() == 1) {
    // If input dimension is already 1, no need to combine dim.
    new_perm->resize(1);
    (*new_perm)[0] = perm[0];
    dim_vec.push_back(shape[0]);
    *new_dims = phi::make_ddim(dim_vec);
    return;
  }
  std::vector<int> new_dim_pos(shape.size(), -1);
  std::vector<int64_t> combined_dims(shape.size(), 0);
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

  *new_dims = phi::make_ddim(dim_vec);
}

template <typename T, typename IndexType = int>
struct TransposeSimple {
  static bool run(const phi::GPUContext& ctx,
                  const Tensor& in,
                  const std::vector<int32_t> perm,
                  Tensor* out) {
    // First reduce the dimensions of the input tensor if possible.
    std::vector<int> new_perm;
    framework::DDim new_dims;
    CombineTransposeDim3(in.dims(), perm, &new_perm, &new_dims);

    // Only use tile copy GPU kernel when dimension is 2 or 3.
    int dims = new_dims.size();
    std::vector<int> new_dim_vec = phi::vectorize<int>(new_dims);
    if (dims < 2 || dims > 3) return false;
    auto in_data = in.data<T>();
    auto out_data = out->data<T>();
    // In most cases, dim will not greater than 3 after combine.
    switch (dims) {
      case 2:
        if (new_perm[0] == 1 && new_perm[1] == 0) {
          // Add the first dimension size as 1.
          new_dim_vec.insert(new_dim_vec.begin(), 1);
          SwapDim1And2InTranspose<T, IndexType>()(
              ctx, in_data, new_dim_vec, out_data);
          return true;
        }
        break;
      case 3:
        // In this case, suppose we can do coalescing read and write in tile.
        if (new_perm == std::vector<int>({0, 2, 1})) {
          SwapDim1And2InTranspose<T, IndexType>()(
              ctx, in_data, new_dim_vec, out_data);
          return true;
        } else if (new_perm == std::vector<int>({2, 1, 0})) {
          // Maybe can optimize later, find a way to do coalescing memory copy.
          // But I think it depends on the data size. If span is not large,
          // maybe
          // can do coalescing.
          SwapDim0And2InTranspose<T, IndexType>()(
              ctx, in_data, new_dim_vec, out_data);
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

template <int N, typename T>
class IdxHelper {
 public:
  IdxHelper() {}
  explicit IdxHelper(const T* dims) {
    for (int i = N - 1; i >= 0; --i) {
      stride_[i] = i < (N - 1) ? dims[i + 1] * stride_[i + 1] : 1;
    }
  }

  __device__ inline T GetStride(int idx) const { return stride_[idx]; }

  __device__ inline void GetIndexFromOffset(T offset, T* index) const {
    T remaining = offset;
#pragma unroll
    for (int i = 0; i < N - 1; ++i) {
      const T idx = remaining / stride_[i];
      remaining -= idx * stride_[i];
      index[i] = idx;
    }
    index[N - 1] = remaining;
  }

 private:
  T stride_[N];
};

template <int N>
class IdxHelper<N, uint32_t> {
 public:
  IdxHelper() {}
  explicit IdxHelper(const uint32_t* dims) {
    for (int i = N - 1; i >= 0; --i) {
      uint32_t value = i < (N - 1) ? dims[i + 1] * stride_[i + 1] : 1;
      divmoder_[i] = paddle::platform::FastDivMod(value);
      stride_[i] = value;
    }
  }

  __device__ inline uint32_t GetStride(int idx) const { return stride_[idx]; }

  __device__ inline void GetIndexFromOffset(uint32_t offset,
                                            uint32_t* index) const {
    uint32_t remaining = offset;
#pragma unroll
    for (int i = 0; i < N - 1; ++i) {
      uint32_t idx = divmoder_[i].Div(remaining);
      index[i] = idx;
      remaining -= idx * stride_[i];
    }
    index[N - 1] = remaining;
  }

 private:
  uint32_t stride_[N];
  paddle::platform::FastDivMod divmoder_[N];
};

// Transform index between memory offset and shape coodinate.
template <typename T, int N>
class IdxAndOffsetHelper {
 public:
  IdxAndOffsetHelper() {}
  ~IdxAndOffsetHelper() = default;

  explicit IdxAndOffsetHelper(const T* dims) {
    index_helper = IdxHelper<N, T>(dims);
  }

  template <typename U>
  explicit IdxAndOffsetHelper(const U* dims) {
    T temp_dims[N];
    for (int i = 0; i < N; ++i) {
      temp_dims[i] = static_cast<T>(dims[i]);
    }
    index_helper = IdxHelper<N, T>(temp_dims);
  }

  __device__ inline T IndexToOffset(const T* index) const {
    T offset = 0;
#pragma unroll
    for (int i = 0; i < N - 1; ++i) {
      offset += index[i] * index_helper.GetStride(i);
    }
    offset += index[N - 1];
    return offset;
  }

  __device__ inline void OffsetToIndex(T offset, T* index) const {
    index_helper.GetIndexFromOffset(offset, index);
  }

 private:
  IdxHelper<N, T> index_helper;
};

template <size_t Rank, typename IndexT>
struct PermuteParams {
 public:
  IdxAndOffsetHelper<IndexT, Rank> src_index_helper;
  IdxAndOffsetHelper<IndexT, Rank> dst_index_helper;
  int perm[Rank]{};

  explicit PermuteParams(const std::vector<size_t>& dims,
                         const std::vector<int>& perm_) {
    size_t dst_dims[Rank];
    for (size_t i = 0; i < Rank; ++i) {
      dst_dims[i] = dims[perm_[i]];
      perm[i] = perm_[i];
    }
    dst_index_helper = IdxAndOffsetHelper<IndexT, Rank>(dst_dims);
    src_index_helper = IdxAndOffsetHelper<IndexT, Rank>(dims.data());
  }
};

// A special kernel for target case, both vectorized read and write supported.
template <typename T, typename IndexT, int VecSize, int Rank>
__global__ void VectorizedPermuteKernel(PermuteParams<Rank, IndexT> params,
                                        const size_t count,
                                        const T* __restrict__ src_data,
                                        T* dst_data) {
  using VecT = phi::AlignedVector<T, VecSize>;
  IndexT src_index[Rank];
  IndexT dst_index[Rank];

  const VecT* __restrict__ src =
      reinterpret_cast<const VecT* __restrict__>(src_data);
  VecT* dst = reinterpret_cast<VecT*>(dst_data);

  IndexT tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (IndexT i = tid; i < count; i += blockDim.x * gridDim.x) {
    params.dst_index_helper.OffsetToIndex(i, dst_index);

#pragma unroll
    for (int j = 0; j < Rank; ++j) {
      src_index[params.perm[j]] = dst_index[j];
    }
    IndexT src_offset = params.src_index_helper.IndexToOffset(src_index);
    dst[i] = src[src_offset];
  }
}

// A general kernel for normal case, only support vectorized write.
template <typename T, typename IndexT, int VecSize, int Rank>
__global__ void GeneralPermuteKernel(PermuteParams<Rank, IndexT> params,
                                     const T* __restrict__ src,
                                     T* dst,
                                     const size_t main_cnt,
                                     const size_t tail_cnt,
                                     const size_t offset) {
  using VecT = phi::AlignedVector<T, VecSize>;
  VecT* vec_dst = reinterpret_cast<VecT*>(dst);

  IndexT src_index[VecSize][Rank];
  IndexT dst_index[VecSize][Rank];

  // Avoid read perm data both in 2 load process.
  __shared__ int perm[Rank];
  if (threadIdx.x < Rank) {
    perm[threadIdx.x] = params.perm[threadIdx.x];
  }
  __syncthreads();

  // Vectorized load data.
  IndexT tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (IndexT idx = tid; idx < main_cnt; idx += blockDim.x * gridDim.x) {
    VecT vec_data;
    IndexT vec_idx = idx * VecSize;

#pragma unroll
    for (int i = 0; i < VecSize; ++i) {
      params.dst_index_helper.OffsetToIndex(vec_idx + i, dst_index[i]);

#pragma unroll
      for (int j = 0; j < Rank; ++j) {
        src_index[i][perm[j]] = dst_index[i][j];
      }
      IndexT src_offset = params.src_index_helper.IndexToOffset(src_index[i]);
      vec_data[i] = src[src_offset];
    }
    vec_dst[idx] = vec_data;
  }

  // Singularized load data.
  if (tid < tail_cnt) {
    IndexT idx = tid + offset;
    params.dst_index_helper.OffsetToIndex(idx, dst_index[0]);

#pragma unroll
    for (int j = 0; j < Rank; ++j) {
      src_index[0][perm[j]] = dst_index[0][j];
    }
    IndexT src_offset = params.src_index_helper.IndexToOffset(src_index[0]);
    dst[idx] = src[src_offset];
  }
}

// A Gerneral permute method that drectly find the dst data
// coordinate in the source data.
template <typename T, typename IndexT, int VecSize, int Rank>
inline void LaunchPermuteKernel(const phi::GPUContext& ctx,
                                const IndexT count,
                                const PermuteType perm_type,
                                const std::vector<size_t>& dims,
                                const std::vector<int>& perm,
                                const T* src,
                                T* dst) {
  size_t main_count = count / VecSize;
  auto params = PermuteParams<Rank, IndexT>(dims, perm);
  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(ctx, main_count);

  if (perm_type == PermuteType::kNormalPermute) {
    size_t tail_count = count - main_count * VecSize;
    size_t offset = count - tail_count;
    GeneralPermuteKernel<T, IndexT, VecSize, Rank>
        <<<config.GetGridSize(), config.GetBlockSize(), 0, ctx.stream()>>>(
            params, src, dst, main_count, tail_count, offset);
  } else {
    VectorizedPermuteKernel<T, IndexT, VecSize, Rank>
        <<<config.GetGridSize(), config.GetBlockSize(), 0, ctx.stream()>>>(
            params, main_count, src, dst);
  }
}

template <typename T, typename IndexT, int VecSize>
inline void LaunchPermuteRankDispatch(const phi::GPUContext& ctx,
                                      const IndexT count,
                                      const PermuteType perm_type,
                                      const std::vector<size_t>& dims,
                                      const std::vector<int>& perm,
                                      const T* src,
                                      T* dst) {
#define CALL_DISPATCH_RANK(rank)                      \
  case rank: {                                        \
    LaunchPermuteKernel<T, IndexT, VecSize, rank>(    \
        ctx, count, perm_type, dims, perm, src, dst); \
    break;                                            \
  }

  switch (dims.size()) {
    CALL_DISPATCH_RANK(1);
    CALL_DISPATCH_RANK(2);
    CALL_DISPATCH_RANK(3);
    CALL_DISPATCH_RANK(4);
    CALL_DISPATCH_RANK(5);
    CALL_DISPATCH_RANK(6);
    CALL_DISPATCH_RANK(7);
    CALL_DISPATCH_RANK(8);
    CALL_DISPATCH_RANK(9);
  }
#undef CALL_DISPATCH_RANK
}

// Aim at transposing the last 2 dimensions. Refer from
// https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
template <typename T, typename IndexT, int VecSize>
__global__ void BatchTransposeKernel(const T* __restrict__ src_data,
                                     T* dst_data,
                                     IndexT rows,
                                     IndexT cols) {
  using VecT = phi::AlignedVector<T, VecSize>;

  __shared__ VecT tile[kTileSize][kShareCol];
  T* single_tile = reinterpret_cast<T*>(tile);

  IndexT col_in_matrix = blockIdx.x * kTileSize + threadIdx.x;
  IndexT offset = blockIdx.z * rows * cols;

  // Vectorized load data from src into shared memory. [rows, cols]
  const VecT* __restrict__ src =
      reinterpret_cast<const VecT* __restrict__>(src_data);

  for (IndexT tile_y = threadIdx.y; tile_y < kTileSize; tile_y += kBlockRows) {
    IndexT row_in_matrix = tile_y + blockIdx.y * kTileSize;

    if (col_in_matrix < cols && row_in_matrix < rows) {
      tile[tile_y][threadIdx.x] =
          src[offset + row_in_matrix * cols + col_in_matrix];
    }
  }

  // Singularized load data from shared memory into dst.
  // and dst_cols = rows, dst_rows = cols, [cols * Vecsize, rows]
  col_in_matrix = blockIdx.y * kTileSize + threadIdx.x;
  offset = offset * VecSize + col_in_matrix;
  IndexT tile_x_idx = threadIdx.x * (kShareCol * VecSize);

  __syncthreads();

  for (IndexT tile_y = threadIdx.y; tile_y < kTileSize; tile_y += kBlockRows) {
    IndexT row_in_matrix = tile_y + blockIdx.x * kTileSize;
    IndexT dst_idx = offset + row_in_matrix * VecSize * rows;
    IndexT tile_idx = tile_x_idx + tile_y * VecSize;
    if (col_in_matrix < /*dst_cols=*/rows &&
        row_in_matrix < /*dst_rows=*/cols) {
#pragma unroll
      for (auto i = 0; i < VecSize; ++i) {
        dst_data[dst_idx + i * rows] = single_tile[tile_idx + i];
      }
    }
  }
}

// With the byte limitation of shared_memory, the VecSize shall be restricted
// for the type whose byte-size is less than 8.
template <typename T,
          typename IndexT,
          int Size,
          int VecSize = (sizeof(T) > 8 ? 1 : Size)>
inline void LaunchTransposeKernel(const phi::GPUContext& ctx,
                                  const std::vector<size_t>& dims,
                                  const T* src,
                                  T* dst) {
  auto rank = dims.size();
  IndexT num_batches = (rank == 2) ? 1 : dims[0];
  IndexT rows = dims[rank - 2];
  IndexT cols = dims[rank - 1];
  IndexT num_tile_rows = (rows + kTileSize - 1) / kTileSize;
  IndexT num_tile_cols = (cols + kTileSize - 1) / kTileSize;

  dim3 blocks(num_tile_cols, num_tile_rows, num_batches);
  dim3 threads(kTileSize, kBlockRows, 1);

  BatchTransposeKernel<T, IndexT, VecSize>
      <<<blocks, threads, 0, ctx.stream()>>>(src, dst, rows, cols);
}

template <typename T, typename IndexT>
inline void LaunchWithDispatchVecSize(const phi::GPUContext& ctx,
                                      const int vec_size,
                                      const PermuteType perm_type,
                                      const std::vector<size_t>& dims,
                                      const std::vector<int>& perm,
                                      const T* src,
                                      T* dst,
                                      IndexT count) {
#define CALL_DISPATCH_VEC_SIZE(vec_size)                               \
  case vec_size: {                                                     \
    if (perm_type == PermuteType::kTranspose) {                        \
      LaunchTransposeKernel<T, IndexT, vec_size>(ctx, dims, src, dst); \
    } else {                                                           \
      LaunchPermuteRankDispatch<T, IndexT, vec_size>(                  \
          ctx, count, perm_type, dims, perm, src, dst);                \
    }                                                                  \
    break;                                                             \
  }

  switch (vec_size) {
    CALL_DISPATCH_VEC_SIZE(1);
    CALL_DISPATCH_VEC_SIZE(2);
    CALL_DISPATCH_VEC_SIZE(4);
    default: {
      PADDLE_THROW(phi::errors::Unimplemented(
          "Unsupported vectorized size: %d !", vec_size));
      break;
    }
  }
#undef CALL_DISPATCH_VEC_SIZE
}

template <typename T>
inline void LaunchWithDispatchIndex(const phi::GPUContext& ctx,
                                    const size_t count,
                                    const int vec_size,
                                    const PermuteType perm_type,
                                    const std::vector<size_t>& dims,
                                    const std::vector<int>& perm,
                                    const T* src,
                                    T* dst) {
  if (count < std::numeric_limits<uint32_t>::max()) {
    LaunchWithDispatchVecSize<T, uint32_t>(ctx,
                                           vec_size,
                                           perm_type,
                                           dims,
                                           perm,
                                           src,
                                           dst,
                                           static_cast<uint32_t>(count));
  } else {
    int64_t cnt = static_cast<int64_t>(count);
    LaunchWithDispatchVecSize<T, int64_t>(ctx,
                                          vec_size,
                                          perm_type,
                                          dims,
                                          perm,
                                          src,
                                          dst,
                                          static_cast<int64_t>(count));
  }
}

template <typename DeviceContext, typename T>
inline void SimplifyThenLaunch(const int rank,
                               const DeviceContext& ctx,
                               const Tensor& in,
                               Tensor* out,
                               const std::vector<int32_t>& perm) {
  int sm_count = ctx.GetSMCount();
  auto src_dims = phi::vectorize<size_t>(in.dims());
  auto simplifier = DimsSimplifier<T>(
      sm_count, rank, perm, src_dims, in.data<T>(), out->data<T>());

  if (simplifier.GetPermType() == PermuteType::kCopy) {
    // If perm is [0,1,2,3], then just operate a DtoD copy.
    phi::Copy(ctx, in, ctx.GetPlace(), false, out);
  } else {
    LaunchWithDispatchIndex<T>(ctx,
                               simplifier.GetCount(),
                               simplifier.GetVecSize(),
                               simplifier.GetPermType(),
                               simplifier.GetDims(),
                               simplifier.GetPerm(),
                               in.data<T>(),
                               out->data<T>());
  }
}

template <typename T>
void TransposeGPUKernelDriver(const phi::GPUContext& ctx,
                              const Tensor& in,
                              const std::vector<int32_t>& perm,
                              Tensor* out) {
  const int rank = perm.size();
  int64_t numel = in.numel();
  bool ret{false};
  if (numel >= std::numeric_limits<int32_t>::max()) {
    ret = TransposeSimple<T, int64_t>::run(ctx, in, perm, out);
  } else {
    ret = TransposeSimple<T>::run(ctx, in, perm, out);
  }
  if (!ret) {
    auto* tuner =
        phi::autotune::MakeTransposeTuner<T>(TransCompute<phi::GPUContext, T>);
    tuner->AddCallBack(
        phi::autotune::MakeCallback<T>(SimplifyThenLaunch<phi::GPUContext, T>));

    size_t key = phi::autotune::TransposeKey(
        phi::vectorize(in.dims()),
        perm,
        paddle::experimental::CppTypeToDataType<T>::Type());

    tuner->Run(ctx,
               phi::autotune::AlgorithmType::kTranspose,
               key,
               rank,
               ctx,
               in,
               out,
               perm);
  }
}

}  // namespace operators
}  // namespace paddle
