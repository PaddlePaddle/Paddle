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
#include "paddle/phi/kernels/autotune/auto_tune_base.h"
#include "paddle/phi/kernels/autotune/cache.h"
#include "paddle/phi/kernels/copy_kernel.h"

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

// Use SM to do data transfer, load a tile into SM then store out.
// All tile read and write are colascing, so can speedup memory copy
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
#pragma unroll
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
#pragma unroll
        for (int ind_i = x_i; ind_i < (tile_width); ind_i += BlockWriteRows) {
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
      TSIZE, 16,
      platform::errors::InvalidArgument(
          "The tile size should smaller than 16, but received is:%d.", TSIZE));

  PADDLE_ENFORCE_EQ(
      (TSIZE & (TSIZE - 1)), 0,
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

template <typename T, int tile_long, int tile_short>
void LaunchNarrowDims2TransposeKernel(const phi::GPUContext& d, int tile_size_i,
                                      int tile_size_j, int total_tiles_count,
                                      const T* input, const Dim3& input_dims,
                                      T* output) {
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
  static void DoTranspose(const phi::GPUContext& d, int tile_size_i,
                          int tile_size_j, int total_tiles_count,
                          const T* input, const Dim3& input_dims, T* output) {
    PADDLE_ENFORCE_EQ(
        (tile_long & (tile_long - 1)), 0,
        platform::errors::InvalidArgument(
            "The length of the longer side of the tile should be power of 2."
            " But received value is:%d.",
            tile_long));

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
      NarrowDims2TransposeDispatch<T, tile_long * 2, tile_short>::DoTranspose(
          d, tile_size_i, tile_size_j, total_tiles_count, input, input_dims,
          output);
    } else {
      NarrowDims2TransposeDispatch<T, tile_long, tile_short + 1>::DoTranspose(
          d, tile_size_i, tile_size_j, total_tiles_count, input, input_dims,
          output);
    }
  }
};

// If Not long tile size, goto this function when compile.
template <typename T, int tile_long, int tile_short>
struct NarrowDims2TransposeDispatch<
    T, tile_long, tile_short,
    typename std::enable_if<
        CheckNonLongTileSize(tile_long, tile_short, sizeof(T)), void>::type> {
  static void DoTranspose(const phi::GPUContext& d, int tile_size_i,
                          int tile_size_j, int total_tiles_count,
                          const T* input, const Dim3& input_dims, T* output) {
    PADDLE_ENFORCE_EQ(
        (tile_long & (tile_long - 1)), 0,
        platform::errors::InvalidArgument(
            "The length of the longer side of the tile should be power of 2."
            " But received value is:%d.",
            tile_long));

    bool request_satisfied = std::max(tile_size_i, tile_size_j) <= tile_long &&
                             std::min(tile_size_i, tile_size_j) <= tile_short;

    if (request_satisfied) {
      LaunchNarrowDims2TransposeKernel<T, tile_long, tile_short>(
          d, tile_size_i, tile_size_j, total_tiles_count, input, input_dims,
          output);
      return;
    }

    NarrowDims2TransposeDispatch<T, tile_long, tile_short + 1>::DoTranspose(
        d, tile_size_i, tile_size_j, total_tiles_count, input, input_dims,
        output);
  }
};

// If long tile size, goto this function when compile.
template <typename T, int tile_long, int tile_short>
struct NarrowDims2TransposeDispatch<
    T, tile_long, tile_short,
    typename std::enable_if<CheckLongTileSize(tile_long, tile_short, sizeof(T)),
                            void>::type> {
  static void DoTranspose(const phi::GPUContext& d, int tile_size_i,
                          int tile_size_j, int total_tiles_count,
                          const T* input, const Dim3& input_dims, T* output) {
    PADDLE_ENFORCE_EQ(
        (tile_long & (tile_long - 1)), 0,
        platform::errors::InvalidArgument(
            "The length of the longer side of the tile should be power of 2,"
            " but received is:%d.",
            tile_long));

    LaunchNarrowDims2TransposeKernel<T, tile_long, tile_short>(
        d, tile_size_i, tile_size_j, total_tiles_count, input, input_dims,
        output);
  }
};

template <typename T, bool conjugate = false>
void SwapDim1And2InNarrow(const phi::GPUContext& d, const T* input,
                          const Dim3& input_dims, T* output,
                          const int kMinTileSize) {
  // First get available tile sizes for the data type requested as backups
  std::vector<std::pair<int, int>> tile_sele;
  auto ret = SelectProperTileSize<sizeof(T)>(&tile_sele);
  PADDLE_ENFORCE_EQ(
      ret, true,
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
    int num_wasted_threads = input_long_edge -
                             framework::CeilOrFloor<int, false>(
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

  int total_tiles_count =
      input_dims_aligned[0] * input_dims_aligned[1] * input_dims_aligned[2];

  // Suppose T can be replaced by system builtin types
  using ElemType = typename SystemElemType<sizeof(T)>::type;

  NarrowDims2TransposeDispatch<ElemType, 32, 2>::DoTranspose(
      d, select_tile_size_i, select_tile_size_j, total_tiles_count,
      reinterpret_cast<const ElemType*>(input), input_dims,
      reinterpret_cast<ElemType*>(output));
}

// This is for case that cannot do coalescing read and write.
// Or input is too small to split into tiles.
template <typename T, int pos0, int pos1, int pos2>
__global__ void TransposeSimpleKernel(int nthreads, const T* __restrict__ input,
                                      Dim3 input_dims, T* __restrict__ output) {
  Dim3 output_dims;
  output_dims[pos0] = input_dims[0];
  output_dims[pos1] = input_dims[1];
  output_dims[pos2] = input_dims[2];

  CUDA_KERNEL_LOOP(output_index, nthreads) {
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
void SendSwapDim1And2InTranspose(const phi::GPUContext& d, const T* input,
                                 const Dim3& input_dims, T* output) {
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

    int total_tiles_count =
        input_dims_aligned[0] * input_dims_aligned[1] * input_dims_aligned[2];

    TilingSwapDim1And2<
        T, kNumThreads, kTileSize,
        kTileSize><<<total_tiles_count, kNumThreads, 0, d.stream()>>>(
        input, input_dims, output);

  } else if (narrow_tile) {
    // If input shape is like Rect, such as 2X100, use Narrow tile size.
    // It makes things complicated, because need to find a tile can coverr
    // input and also reach best coalescing.
    SwapDim1And2InNarrow<T>(d, input, input_dims, output, kMinTileSize);
  } else {
    // If input shape is small, such as 8X8, just do simple copy
    int total_elements = input_dims[0] * input_dims[1] * input_dims[2];
    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(d, total_elements);
    TransposeSimpleKernel<T, 0, 2, 1><<<
        config.block_per_grid.x, config.thread_per_block.x, 0, d.stream()>>>(
        total_elements, input, input_dims, output);
  }
}

template <typename T>
struct SwapDim1And2InTranspose {
  typedef phi::GPUContext Device;
  void operator()(const Device& d, const T* in,
                  const std::vector<int>& combined_dims, T* out) {
    Dim3 input_dims = {static_cast<int>(combined_dims[0]),
                       static_cast<int>(combined_dims[1]),
                       static_cast<int>(combined_dims[2])};
    SendSwapDim1And2InTranspose<T>(d, in, input_dims, out);
  }
};

template <typename T>
struct SwapDim0And2InTranspose {
  typedef phi::GPUContext Device;
  void operator()(const Device& d, const T* in,
                  const std::vector<int>& combined_dims, T* out) {
    Dim3 input_dims = {static_cast<int>(combined_dims[0]),
                       static_cast<int>(combined_dims[1]),
                       static_cast<int>(combined_dims[2])};

    size_t total_size = combined_dims[0] * combined_dims[1] * combined_dims[2];
    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(d, total_size);

    TransposeSimpleKernel<T, 2, 1, 0><<<
        config.block_per_grid.x, config.thread_per_block.x, 0, d.stream()>>>(
        total_size, in, input_dims, out);
  }
};

// This function is to combine dimension. fox example:
// (0, 1, 3, 2) --> (0, 2, 1)
inline void CombineTransposeDim3(const framework::DDim& shape,
                                 const std::vector<int>& perm,
                                 std::vector<int>* new_perm,
                                 framework::DDim* new_dims) {
  PADDLE_ENFORCE_EQ(shape.size(), perm.size(),
                    platform::errors::InvalidArgument(
                        " shape should have the save dim with perm, but"
                        " received shape size is:%d, perm size is:%d.",
                        shape.size(), perm.size()));

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

  *new_dims = phi::make_ddim(dim_vec);
}

template <typename T>
struct TransposeSimple {
  static bool run(const phi::GPUContext& ctx, const Tensor& in,
                  const std::vector<int32_t> perm, Tensor* out) {
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
          SwapDim1And2InTranspose<T>()(ctx, in_data, new_dim_vec, out_data);
          return true;
        }
        break;
      case 3:
        // In this case, suppose we can do coalescing read and write in tile.
        if (new_perm == std::vector<int>({0, 2, 1})) {
          SwapDim1And2InTranspose<T>()(ctx, in_data, new_dim_vec, out_data);
          return true;
        } else if (new_perm == std::vector<int>({2, 1, 0})) {
          // Maybe can optimize later, find a way to do coalescing memory copy.
          // But I think it depends on the data size. If span is not large,
          // maybe
          // can do coalescing.
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

constexpr int32_t kMov4TileSize = 32;
constexpr int32_t kBlockRows = 8;

template <int N, typename T>
class IndexHelper {
 public:
  IndexHelper() {}
  explicit IndexHelper(const T* dims) {
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
class IndexHelper<N, uint32_t> {
 public:
  IndexHelper() {}
  explicit IndexHelper(const uint32_t* dims) {
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

template <typename T, int N>
class NdIdxAndOffset {
 public:
  NdIdxAndOffset() {}
  ~NdIdxAndOffset() = default;

  explicit NdIdxAndOffset(const T* dims) {
    index_helper = IndexHelper<N, T>(dims);
  }

  template <typename U>
  explicit NdIdxAndOffset(const U* dims) {
    T temp_dims[N];
    for (int i = 0; i < N; ++i) {
      temp_dims[i] = static_cast<T>(dims[i]);
    }
    index_helper = IndexHelper<N, T>(temp_dims);
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
  IndexHelper<N, T> index_helper;
};

template <size_t Rank, typename IndexT>
struct PermuteParams {
  NdIdxAndOffset<IndexT, Rank> src_index_helper;
  NdIdxAndOffset<IndexT, Rank> dst_index_helper;
  int32_t __restrict__ perm[Rank]{};
  IndexT count{};

  explicit PermuteParams(const std::vector<size_t>& in_dims,
                         const std::vector<int>& perm_, IndexT count_)
      : count(count_) {
    size_t dst_dims[Rank];
    for (size_t i = 0; i < Rank; ++i) {
      dst_dims[i] = in_dims[perm_[i]];
    }
    dst_index_helper = NdIdxAndOffset<IndexT, Rank>(dst_dims);
    src_index_helper = NdIdxAndOffset<IndexT, Rank>(in_dims.data());

    for (size_t i = 0; i < Rank; ++i) {
      perm[i] = perm_[i];
    }
  }
};

template <typename T, typename IndexT, size_t Rank, size_t MovementSize>
__global__ void GeneralPermuteKernel(PermuteParams<Rank, IndexT> params,
                                     const T* __restrict__ src_data,
                                     T* dst_data) {
  IndexT src_index[Rank];
  IndexT dst_index[Rank];
  int perm[Rank];
  IndexT tid = blockIdx.x * blockDim.x + threadIdx.x;

  using Type = typename std::aligned_storage<MovementSize, MovementSize>::type;
  const Type* src = reinterpret_cast<const Type* __restrict__>(src_data);
  Type* dst = reinterpret_cast<Type*>(dst_data);

  for (IndexT i = tid; i < params.count; i += blockDim.x * gridDim.x) {
    params.dst_index_helper.OffsetToIndex(i, dst_index);

#pragma unroll
    for (int j = 0; j < Rank; ++j) {
      src_index[params.perm[j]] = dst_index[j];
    }
    IndexT src_offset = params.src_index_helper.IndexToOffset(src_index);
    dst[i] = src[src_offset];
  }
}

// Refer from
// https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
template <typename T, size_t MovementSize, typename IndexT>
__global__ void BatchTransposeKernel(const T* __restrict__ src_data,
                                     T* dst_data, IndexT rows, IndexT cols,
                                     IndexT num_tile_rows, IndexT num_tile_cols,
                                     int32_t block_nums) {
  const IndexT src_rows = rows;
  const IndexT src_cols = cols;
  const IndexT dst_rows = cols;
  const IndexT dst_cols = rows;

  using Type = typename std::aligned_storage<MovementSize, MovementSize>::type;
  __shared__ Type tile[kMov4TileSize]
                      [kMov4TileSize + 1];  // To avoid bank conflict.

  const Type* src = reinterpret_cast<const Type* __restrict__>(src_data);
  Type* dst = reinterpret_cast<Type*>(dst_data);

  IndexT batch_num_tile = num_tile_rows * num_tile_cols;

  for (int i = blockIdx.x; i < block_nums; i += gridDim.x) {
    const IndexT batch_index = i / batch_num_tile;
    const IndexT tile_index = i - batch_index * batch_num_tile;

    const IndexT tile_row_index = tile_index / num_tile_cols;
    const IndexT tile_col_index = tile_index - tile_row_index * num_tile_cols;

    const IndexT offset = batch_index * src_rows * src_cols;

    {
      IndexT col_in_tile = threadIdx.x;
      IndexT col_in_matrix = tile_col_index * kMov4TileSize + threadIdx.x;

#pragma unroll
      for (IndexT row_in_tile = threadIdx.y; row_in_tile < kMov4TileSize;
           row_in_tile += kBlockRows) {
        IndexT row_in_matrix = row_in_tile + tile_row_index * kMov4TileSize;
        if (col_in_matrix < src_cols && row_in_matrix < src_rows) {
          tile[row_in_tile][col_in_tile] =
              src[offset + row_in_matrix * src_cols + col_in_matrix];
        }
      }
    }
    __syncthreads();
    {
      IndexT col_in_tile = threadIdx.x;
      IndexT col_in_matrix = tile_row_index * kMov4TileSize + threadIdx.x;

#pragma unroll
      for (IndexT row_in_tile = threadIdx.y; row_in_tile < kMov4TileSize;
           row_in_tile += kBlockRows) {
        IndexT row_in_matrix = row_in_tile + tile_col_index * kMov4TileSize;
        if (col_in_matrix < dst_cols && row_in_matrix < dst_rows) {
          dst[offset + row_in_matrix * dst_cols + col_in_matrix] =
              tile[col_in_tile][row_in_tile];
        }
      }
    }
    __syncthreads();
  }
}

template <typename T, size_t Rank, size_t MovementSize, typename IndexT>
inline void LaunchBatchTransposeKernel(
    const phi::GPUContext& ctx, const PermuteParams<Rank, IndexT>& params,
    const IndexT& num_batches, const IndexT& rows, const IndexT& cols,
    const T* src, T* dst) {
  IndexT num_tile_rows = (rows + kMov4TileSize - 1) / kMov4TileSize;
  IndexT num_tile_cols = (cols + kMov4TileSize - 1) / kMov4TileSize;
  constexpr int32_t kCudaMaxBlocksNum = 8192;
  const int32_t block_nums = num_batches * num_tile_rows * num_tile_cols;
  int32_t blocks = std::min(block_nums, kCudaMaxBlocksNum);

  BatchTransposeKernel<
      T, MovementSize,
      IndexT><<<blocks, dim3(kMov4TileSize, kBlockRows), 0, ctx.stream()>>>(
      src, dst, rows, cols, num_tile_rows, num_tile_cols, block_nums);
}

template <size_t Rank, typename IndexT>
inline bool CheckLaunchBatchTranspose(const int* perm,
                                      const IndexT& num_batches,
                                      const IndexT& rows, const IndexT& cols) {
  bool result = false;
  bool greater_than_tile =
      (rows < kMov4TileSize || cols < kMov4TileSize) ? false : true;

  if (greater_than_tile) {
    if (num_batches == 1 && perm[1] == 0 && perm[0] == 1) {
      // 2D tensor case: (0, 1) -> (1, 0)
      result = true;
    } else if (Rank == 3 && perm[2] == 1 && perm[1] == 2) {
      // 3D tensor case: (0, 1, 2) -> (0, 2, 1)
      return true;
    }
  }
  return result;
}

template <typename T, size_t Rank, size_t MovementSize, typename IndexT>
inline void LaunchTransposeKernel(const phi::GPUContext& ctx,
                                  const std::vector<size_t>& in_dims,
                                  const std::vector<int>& perm, const T* src,
                                  T* dst, IndexT count) {
  auto params = PermuteParams<Rank, IndexT>(in_dims, perm, count);
  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(ctx, count);

  if (Rank == 2 || Rank == 3) {
    IndexT num_batches = (Rank == 2) ? 1 : in_dims[0];
    IndexT rows = (Rank == 2) ? in_dims[0] : in_dims[1];
    IndexT cols = (Rank == 2) ? in_dims[1] : in_dims[2];

    if (CheckLaunchBatchTranspose<Rank, IndexT>(params.perm, num_batches, rows,
                                                cols)) {
      LaunchBatchTransposeKernel<T, Rank, MovementSize, IndexT>(
          ctx, params, num_batches, rows, cols, src, dst);
    } else {
      GeneralPermuteKernel<T, IndexT, Rank, MovementSize><<<
          config.GetGridSize(), config.GetBlockSize(), 0, ctx.stream()>>>(
          params, src, dst);
    }
  } else {
    GeneralPermuteKernel<T, IndexT, Rank, MovementSize><<<
        config.GetGridSize(), config.GetBlockSize(), 0, ctx.stream()>>>(
        params, src, dst);
  }
}

template <typename T, size_t Rank, size_t MovementSize>
inline void DispatchIndexType(const phi::GPUContext& ctx,
                              const std::vector<size_t>& in_dims,
                              const std::vector<int>& perm, const T* src,
                              T* dst) {
  size_t count = std::accumulate(in_dims.begin(), in_dims.begin() + Rank,
                                 size_t{1}, std::multiplies<size_t>());

  if (count < std::numeric_limits<uint32_t>::max()) {
    LaunchTransposeKernel<T, Rank, MovementSize, uint32_t>(
        ctx, in_dims, perm, src, dst, static_cast<uint32_t>(count));
  } else {
    int64_t cnt = static_cast<int64_t>(count);
    LaunchTransposeKernel<T, Rank, MovementSize, int64_t>(
        ctx, in_dims, perm, src, dst, static_cast<int64_t>(count));
  }
}

template <typename T, size_t Rank>
inline void DispatchMovementSize(const phi::GPUContext& ctx,
                                 size_t movement_size,
                                 const const std::vector<size_t>& in_dims,
                                 const std::vector<int>& perm, const T* src,
                                 T* dst) {
#define CALL_DISPATCH_INDEX_TYPE_FUNC(size)                         \
  case size: {                                                      \
    DispatchIndexType<T, Rank, size>(ctx, in_dims, perm, src, dst); \
    break;                                                          \
  }

  switch (movement_size) {
    CALL_DISPATCH_INDEX_TYPE_FUNC(1);
    CALL_DISPATCH_INDEX_TYPE_FUNC(2);
    CALL_DISPATCH_INDEX_TYPE_FUNC(4);
    CALL_DISPATCH_INDEX_TYPE_FUNC(8);
    CALL_DISPATCH_INDEX_TYPE_FUNC(16);
  }
#undef CALL_DISPATCH_INDEX_TYPE_FUNC
}

template <typename T>
inline void LaunchWithSimplifiedDispatch(const phi::GPUContext& ctx,
                                         const size_t rank,
                                         const size_t movement_size,
                                         const std::vector<size_t>& in_dims,
                                         const std::vector<int>& perm,
                                         const T* src, T* dst) {
#define CALL_DISPATCH_MOVEMENT_SIZE_FUNC(rank)                            \
  case rank: {                                                            \
    DispatchMovementSize<T, rank>(ctx, movement_size, in_dims, perm, src, \
                                  dst);                                   \
    break;                                                                \
  }

  switch (rank) {
    CALL_DISPATCH_MOVEMENT_SIZE_FUNC(1);
    CALL_DISPATCH_MOVEMENT_SIZE_FUNC(2);
    CALL_DISPATCH_MOVEMENT_SIZE_FUNC(3);
    CALL_DISPATCH_MOVEMENT_SIZE_FUNC(4);
    CALL_DISPATCH_MOVEMENT_SIZE_FUNC(5);
    CALL_DISPATCH_MOVEMENT_SIZE_FUNC(6);
    CALL_DISPATCH_MOVEMENT_SIZE_FUNC(7);
    CALL_DISPATCH_MOVEMENT_SIZE_FUNC(8);
    CALL_DISPATCH_MOVEMENT_SIZE_FUNC(9);
  }
#undef CALL_DISPATCH_MOVEMENT_SIZE_FUNC
}

template <typename DeviceContext, typename T>
inline void SimplifyThenLaunch(const int rank, const DeviceContext& ctx,
                               const Tensor& in, Tensor* out,
                               const std::vector<int32_t>& perm) {
  auto src_dims = phi::vectorize<size_t>(in.dims());

  // kMaxMovementSize is for vectorized load, for fp32, the vec_size
  // is (kMaxMovementSize)/sizeof(fp32) = 4, for fp16 is 8.
  auto simplifier = DimsAndPermSimplifier<T, /*kMaxMovementSize=*/16>(
      rank, sizeof(T), perm, &src_dims, in.data<T>(), out->data<T>());

  if (simplifier.IsSequential()) {
    // If perm is [0,1,2,3], then just operate a DtoD copy.
    phi::Copy(ctx, in, ctx.GetPlace(), true, out);
  } else {
    LaunchWithSimplifiedDispatch<T>(ctx, simplifier.GetRank(),
                                    simplifier.GetMovementSize(),
                                    simplifier.GetDims(), simplifier.GetPerm(),
                                    in.data<T>(), out->data<T>());
  }
}

template <typename T>
size_t GetTransposeKey(const int rank, const Tensor& in,
                       const std::vector<int32_t>& perm) {
  auto in_shape = phi::vectorize(in.dims());
  return phi::autotune::GetKey(
      in_shape, perm, rank, paddle::experimental::CppTypeToDataType<T>::Type());
}

template <typename T>
void TransposeGPUKernelDriver(const phi::GPUContext& dev_ctx, const int rank,
                              const Tensor& in,
                              const std::vector<int32_t>& perm, Tensor* out) {
  PADDLE_ENFORCE_LT(
      rank, phi::DDim::kMaxRank,
      platform::errors::OutOfRange(
          "The maximum dimension rank of "
          "tensor is expected to be less than %d, but here is %d.",
          phi::DDim::kMaxRank, rank));

  auto ret = TransposeSimple<T>::run(dev_ctx, in, perm, out);
  if (!ret) {
    SimplifyThenLaunch<phi::GPUContext, T>(rank, dev_ctx, in, out, perm);
    // auto* tuner =
    //     phi::autotune::MakeTransposeTuner<T>(TransCompute<phi::GPUContext,
    //     T>);
    // if (!tuner->CheckInit()) {
    //   tuner->AddCallBack(phi::autotune::MakeCallback<T>(
    //       SimplifyThenLaunch<phi::GPUContext, T>));
    //   tuner->FinishInit();
    // }

    // auto key = GetTransposeKey<T>(rank, in, perm);
    // auto& cache = phi::autotune::AutoTuneCache::Instance().GetTranspose();
    // if (cache.Find(key)) {
    //   auto index = cache.Get(key);
    //   tuner->RunBestKernel(index, rank, dev_ctx, in, out, perm);
    // } else {
    //   // All avaliable kernels have ran while picking the best kernel, so
    //   // there may be no need for another RunBestKernel.
    //   auto index = tuner->PickBestKernel(dev_ctx, rank, dev_ctx, in, out,
    //   perm);
    //   cache.Set(key, index);
    // }
  }
}

}  // namespace operators
}  // namespace paddle
