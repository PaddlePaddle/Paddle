// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/cum_kernel.h"

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/reverse.h>
#include <thrust/scan.h>
#ifdef __NVCC__
#include <cub/cub.cuh>
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include "paddle/common/hostdevice.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename InT, typename OutT>
__global__ void MatrixRowReverse(const InT* matrix_data,
                                 OutT* reverse_data,
                                 int64_t grid_size,
                                 int64_t reverse_size) {
  int item_per_block = 1024;
  for (int64_t bx = blockIdx.x; bx < grid_size; bx += gridDim.x) {
    for (int64_t block_offset = 0; block_offset < reverse_size;
         block_offset += item_per_block) {
      int64_t reverse_offset = block_offset + threadIdx.x;
      int64_t src_offset = bx * reverse_size + reverse_offset;
      int64_t dst_offset =
          bx * reverse_size + (reverse_size - reverse_offset - 1);
      if (reverse_offset < reverse_size) {
        reverse_data[dst_offset] = static_cast<OutT>(matrix_data[src_offset]);
      }
    }
  }
}

template <typename T, typename Op>
struct BlockPrefixCallbackOp {
  // Running prefix
  T running_total_;
  Op op_;

  __device__ BlockPrefixCallbackOp(T running_total, Op op)
      : running_total_(running_total), op_(op) {}

  // Callback operator to be entered by the first warp of threads in the block.
  // tid 0 is responsible for returning a value for seeding the block-wide scan.
  __device__ T operator()(T block_aggregate) {
    T old_prefix = running_total_;
    running_total_ = op_(old_prefix, block_aggregate);
    return old_prefix;
  }
};

// No bank-conflict transpose
template <typename InT, typename OutT, int TILE_DIM, int BLOCK_ROWS>
__global__ void MatrixTranspose(OutT* odata,
                                const InT* idata,
                                size_t height,
                                size_t width) {
  __shared__ InT tile[TILE_DIM][TILE_DIM + 1];

  int64_t wblocks = (width + TILE_DIM - 1) / TILE_DIM;
  int64_t hblocks = (height + TILE_DIM - 1) / TILE_DIM;

  int64_t block_i = blockIdx.x;
  for (; block_i < wblocks * hblocks; block_i += gridDim.x) {
    int64_t block_y = block_i / wblocks;
    int64_t block_x = block_i % wblocks;
    int64_t x = block_x * TILE_DIM + threadIdx.x;
    int64_t y = block_y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
      if (x < width && (y + j) < height) {
        tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];
      }
    }
    __syncthreads();

    x = block_y * TILE_DIM + threadIdx.x;  // transpose block offset
    y = block_x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
      if (x < height && (y + j) < width) {
        odata[(y + j) * height + x] =
            static_cast<OutT>(tile[threadIdx.x][threadIdx.y + j]);
      }
    }
  }
}

struct LogAddExp {
  template <typename T>
  __host__ __device__ __forceinline__ T operator()(const T& a,
                                                   const T& b) const {
    return std::log(1 + std::exp(std::min(a, b) - std::max(a, b))) +
           std::max(a, b);
  }
};

struct ComplexSum {
  template <typename T>
  __host__ __device__ __forceinline__ T operator()(const T& a,
                                                   const T& b) const {
    return a + b;
  }
};

template <typename T, typename op>
struct Identity;

template <typename T>
struct Identity<T, cub::Sum> {
  static constexpr T value = 0;
};

template <typename T>
struct Identity<T, LogAddExp> {
  static constexpr T value = std::numeric_limits<T>::lowest();
};

template <typename T>
struct Identity<T, ComplexSum> {
  static constexpr T value = {0, 0};
};

template <typename InT,
          typename OutT,
          int BLOCK_THREADS,
          int ITEMS_PER_THREAD,
          typename Op>
__global__ void BlockScanKernel(OutT* d_out,
                                const InT* d_in,
                                int64_t grid_size,
                                int64_t scan_size,
                                bool exclusive,
                                Op op) {
  // Logical accumulation type, based on InT, consistent with Op.
  using AccLogicT = typename phi::dtype::MPTypeTrait<InT>::Type;

  // Specialize BlockLoad for InT, BlockStore for OutT.
  // cub::BlockScan and its temporary storage will be for AccLogicT.
  typedef cub::
      BlockLoad<InT, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_TRANSPOSE>
          BlockLoadIn;
  typedef cub::BlockStore<OutT,
                          BLOCK_THREADS,
                          ITEMS_PER_THREAD,
                          cub::BLOCK_STORE_TRANSPOSE>
      BlockStoreOut;
  typedef cub::BlockScan<AccLogicT, BLOCK_THREADS> BlockScanInternal;

  __shared__ union {
    typename BlockLoadIn::TempStorage load;
    typename BlockStoreOut::TempStorage store;
    typename BlockScanInternal::TempStorage scan;
  } temp_storage;

  int64_t item_per_block = BLOCK_THREADS * ITEMS_PER_THREAD;
  for (int64_t bx = blockIdx.x; bx < grid_size; bx += gridDim.x) {
    BlockPrefixCallbackOp<AccLogicT, Op> prefix_op(
        Identity<AccLogicT, Op>::value, op);
    for (int64_t block_offset = 0; block_offset < scan_size;
         block_offset += item_per_block) {
      int64_t valid_item = (scan_size - block_offset > item_per_block)
                               ? item_per_block
                               : (scan_size - block_offset);
      if (scan_size < item_per_block) {
        valid_item = scan_size;
      }

      int64_t current_item_offset_in_sequence = bx * scan_size + block_offset;

      InT thread_input_keys_in[ITEMS_PER_THREAD];
      BlockLoadIn(temp_storage.load)
          .Load(d_in + current_item_offset_in_sequence,
                thread_input_keys_in,
                valid_item,
                static_cast<InT>(0));

      AccLogicT thread_keys_for_scan[ITEMS_PER_THREAD];
#pragma unroll
      for (int64_t i = 0; i < ITEMS_PER_THREAD; ++i) {
        thread_keys_for_scan[i] =
            static_cast<AccLogicT>(thread_input_keys_in[i]);
      }

      __syncthreads();
      if (exclusive) {
        BlockScanInternal(temp_storage.scan)
            .ExclusiveScan(
                thread_keys_for_scan, thread_keys_for_scan, op, prefix_op);
      } else {
        BlockScanInternal(temp_storage.scan)
            .InclusiveScan(
                thread_keys_for_scan, thread_keys_for_scan, op, prefix_op);
      }
      __syncthreads();

      // Cast AccLogicT result back to OutT for final storage
      OutT thread_output_keys_out[ITEMS_PER_THREAD];
#pragma unroll
      for (int64_t i = 0; i < ITEMS_PER_THREAD; ++i) {
        thread_output_keys_out[i] = static_cast<OutT>(thread_keys_for_scan[i]);
      }

      BlockStoreOut(temp_storage.store)
          .Store(d_out + current_item_offset_in_sequence,
                 thread_output_keys_out,
                 valid_item);
    }
  }
}

template <typename Context, typename T>
typename std::enable_if<!std::is_same<T, phi::dtype::float16>::value &&
                        !std::is_same<T, phi::dtype::bfloat16>::value>::type
ThrustCumsumKernel(const Context& dev_ctx,
                   const T* in_data,
                   T* out_data,
                   int64_t size,
                   bool reverse,
                   bool exclusive) {
#ifdef __HIPCC__
  const auto& policy = thrust::hip::par.on(dev_ctx.stream());
#else
  phi::memory_utils::ThrustAllocator<cudaStream_t> allocator(dev_ctx.GetPlace(),
                                                             dev_ctx.stream());
  const auto& policy = thrust::cuda::par(allocator).on(dev_ctx.stream());
#endif
  if (reverse) {
    thrust::reverse_iterator<thrust::device_ptr<const T>> reversed_in(
        thrust::device_pointer_cast(in_data) + size);
    thrust::reverse_iterator<thrust::device_ptr<T>> reversed_out(
        thrust::device_pointer_cast(out_data) + size);
    if (exclusive) {
      thrust::exclusive_scan(
          policy, reversed_in, reversed_in + size, reversed_out);
    } else {
      thrust::inclusive_scan(
          policy, reversed_in, reversed_in + size, reversed_out);
    }
  } else {
    if (exclusive) {
      thrust::exclusive_scan(policy, in_data, in_data + size, out_data);
    } else {
      thrust::inclusive_scan(policy, in_data, in_data + size, out_data);
    }
  }
}

template <typename Context, typename T>
typename std::enable_if<std::is_same<T, phi::dtype::float16>::value>::type
ThrustCumsumKernel(const Context& dev_ctx,
                   const phi::dtype::float16* in_data,
                   phi::dtype::float16* out_data,
                   int64_t size,
                   bool reverse,
                   bool exclusive) {}

template <typename Context, typename T>
typename std::enable_if<std::is_same<T, phi::dtype::bfloat16>::value>::type
ThrustCumsumKernel(const Context& dev_ctx,
                   const phi::dtype::bfloat16* in_data,
                   phi::dtype::bfloat16* out_data,
                   int64_t size,
                   bool reverse,
                   bool exclusive) {}

template <typename InT,
          typename OutT,
          typename Context,
          typename OpForInT,
          typename OpForOutT>
void ScanKernel(const Context& dev_ctx,
                const DenseTensor& x,
                int axis,
                bool flatten,
                bool exclusive,
                bool reverse,
                OpForInT opInT,
                OpForOutT opOutT,
                DenseTensor* out) {
  OutT* out_data = dev_ctx.template Alloc<OutT>(out);
  if (out->numel() == 0) {
    return;
  }
  // For 0D Tensor
  if (out->numel() == 1) {
    auto raw_dims = out->dims();
    phi::Copy<Context>(dev_ctx, x, dev_ctx.GetPlace(), false, out);
    out->Resize(raw_dims);
    return;
  }

  auto out_dims = out->dims();
  auto size = x.numel();

  PADDLE_ENFORCE_EQ(
      axis < out_dims.size() && axis >= (0 - out_dims.size()),
      true,
      common::errors::OutOfRange(
          "Attr(axis) is out of range, It's expected "
          "to be in range of [-%d, %d]. But received Attr(axis) = %d.",
          out_dims.size(),
          out_dims.size() - 1,
          axis));
  if (axis < 0) {
    axis += out_dims.size();
  }

  const InT* in_data = x.data<InT>();

  // Use thrust for parallel acceleration when the input size is equal to the
  // length of the 'axis' dimension.
  if constexpr (std::is_same<InT, OutT>::value) {
    if (!std::is_same<InT, phi::dtype::float16>::value &&
        !std::is_same<InT, phi::dtype::bfloat16>::value &&
        std::is_same<OpForInT, cub::Sum>::value && size == out_dims[axis]) {
      ThrustCumsumKernel<Context, InT>(
          dev_ctx, in_data, out_data, size, reverse, exclusive);
      return;
    }
  }

  size_t height = 1;
  size_t width = 1;
  for (size_t i = 0; i <= axis; i++) {
    height *= out_dims[i];
  }

  for (size_t i = axis + 1; i < out_dims.size(); i++) {
    width *= out_dims[i];
  }
  int scan_size = out_dims[axis];
  bool transpose = (axis != out_dims.size() - 1);

  DenseTensor tmp_tensor;
  tmp_tensor.Resize(out_dims);
  auto* tmp_data = dev_ctx.template Alloc<OutT>(&tmp_tensor);

  OutT* next_in_data = out_data;
  OutT* next_out_data = tmp_data;
  int64_t max_grid_x = dev_ctx.GetCUDAMaxGridDimSize()[0];

  // Do pre-process transpose
  int tile_size = 32;
  dim3 blocks(32, 8);
  int64_t transpose_grids = ((width + tile_size - 1) / tile_size) *
                            ((height + tile_size - 1) / tile_size);
  transpose_grids = std::min(transpose_grids, max_grid_x);
  if (transpose) {
    MatrixTranspose<InT, OutT, 32, 8>
        <<<transpose_grids, blocks, 0, dev_ctx.stream()>>>(
            out_data, in_data, height, width);
    next_in_data = out_data;
    next_out_data = tmp_data;
  }
  auto swap_ptr = [](OutT*& ptr1, OutT*& ptr2) {  // which
    OutT* tmp = ptr2;
    ptr2 = ptr1;
    ptr1 = tmp;
  };

  // Do pre-process reverse
  int64_t outer_size = height / scan_size;
  int64_t inner_size = width;
  int64_t grid_size = outer_size * inner_size;
  int64_t scan_grid = std::min(grid_size, max_grid_x);

  if (reverse) {
    if (transpose) {
      MatrixRowReverse<OutT, OutT><<<scan_grid, 1024, 0, dev_ctx.stream()>>>(
          next_in_data, next_out_data, grid_size, scan_size);
      if (!transpose) next_in_data = tmp_data;
      swap_ptr(next_in_data, next_out_data);
    } else {
      MatrixRowReverse<InT, OutT><<<scan_grid, 1024, 0, dev_ctx.stream()>>>(
          in_data, out_data, grid_size, scan_size);
    }
  }

  // Do scan
  if (!transpose && !reverse) {
    BlockScanKernel<InT, OutT, 128, 4, OpForInT>
        <<<scan_grid, 128, 0, dev_ctx.stream()>>>(
            out_data, in_data, grid_size, scan_size, exclusive, opInT);

  } else {
    BlockScanKernel<OutT, OutT, 128, 4, OpForOutT>
        <<<scan_grid, 128, 0, dev_ctx.stream()>>>(next_out_data,
                                                  next_in_data,
                                                  grid_size,
                                                  scan_size,
                                                  exclusive,
                                                  opOutT);
  }
  swap_ptr(next_in_data, next_out_data);

  // Do post-process reverse and transpose
  if (reverse) {
    MatrixRowReverse<OutT, OutT><<<scan_grid, 1024, 0, dev_ctx.stream()>>>(
        next_in_data, next_out_data, grid_size, scan_size);
    swap_ptr(next_in_data, next_out_data);
  }
  if (transpose) {
    MatrixTranspose<OutT, OutT, 32, 8>
        <<<transpose_grids, blocks, 0, dev_ctx.stream()>>>(
            next_out_data, next_in_data, width, height);
  }
}

template <typename DeviceContext, typename InT>
struct CumsumKernelVisitor {
  const DeviceContext& dev_ctx_;
  const DenseTensor& x_;
  int axis_scalar_;
  bool flatten_;
  bool exclusive_;
  bool reverse_;
  DenseTensor* out_;

  CumsumKernelVisitor(const DeviceContext& dev_ctx,
                      const DenseTensor& x,
                      int axis,
                      bool flatten,
                      bool exclusive,
                      bool reverse,
                      DenseTensor* out)
      : dev_ctx_(dev_ctx),
        x_(x),
        axis_scalar_(axis),
        flatten_(flatten),
        exclusive_(exclusive),
        reverse_(reverse),
        out_(out) {}

  template <typename OutT>
  void apply() const {
    using OpForInT = typename std::conditional<
        std::is_same<InT, phi::dtype::complex<float>>::value ||
            std::is_same<InT, phi::dtype::complex<double>>::value,
        ComplexSum,
        cub::Sum>::type;

    using OpForOutT = typename std::conditional<
        std::is_same<OutT, phi::dtype::complex<float>>::value ||
            std::is_same<OutT, phi::dtype::complex<double>>::value,
        ComplexSum,
        cub::Sum>::type;
    auto opInT = OpForInT();
    auto opOutT = OpForOutT();
    ScanKernel<InT, OutT, DeviceContext, OpForInT, OpForOutT>(dev_ctx_,
                                                              x_,
                                                              axis_scalar_,
                                                              flatten_,
                                                              exclusive_,
                                                              reverse_,
                                                              opInT,
                                                              opOutT,
                                                              out_);
  }
};

template <typename T, typename Context>
void CumsumKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const Scalar& axis,
                  bool flatten,
                  bool exclusive,
                  bool reverse,
                  DataType dtype,
                  DenseTensor* out) {
  phi::VisitDataType(
      out->dtype(),
      CumsumKernelVisitor<Context, T>{
          dev_ctx, x, axis.to<int>(), flatten, exclusive, reverse, out});
}

template <typename T, typename Context>
void LogcumsumexpKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        int axis,
                        bool flatten,
                        bool exclusive,
                        bool reverse,
                        DenseTensor* out) {
  using Op = LogAddExp;
  auto op = Op();
  ScanKernel<T, T, Context, Op, Op>(
      dev_ctx, x, axis, flatten, exclusive, reverse, op, op, out);
}

}  // namespace phi

#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(cumsum,
                   GPU,
                   ALL_LAYOUT,
                   phi::CumsumKernel,
                   float,
                   phi::dtype::float16,
                   double,
                   int16_t,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(
    logcumsumexp, GPU, ALL_LAYOUT, phi::LogcumsumexpKernel, float, double) {}
#else
PD_REGISTER_KERNEL(cumsum,
                   GPU,
                   ALL_LAYOUT,
                   phi::CumsumKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(logcumsumexp,
                   GPU,
                   ALL_LAYOUT,
                   phi::LogcumsumexpKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#endif
