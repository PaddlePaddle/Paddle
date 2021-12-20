/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

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
#include "paddle/fluid/operators/cum_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"

using Tensor = paddle::framework::Tensor;
using LoDTensor = paddle::framework::LoDTensor;

namespace paddle {
namespace operators {

template <typename T, int BLOCK_SIZE>
__device__ void BlockReverse(const T* idata, T* odata, int src_base,
                             int dst_base, int valid_item) {
  __shared__ T sh_mem[BLOCK_SIZE];
  int tx = threadIdx.x;

  int offset = tx;
  int in_index = src_base + offset;
  if (offset >= valid_item) {
    sh_mem[offset] = 0;
  } else {
    int sh_mem_index = BLOCK_SIZE - offset - 1;
    T data = idata[in_index];
    sh_mem[sh_mem_index] = data;
  }

  __syncthreads();
  int out_index = dst_base - offset;
  if (offset < valid_item) {
    int sh_mem_index = BLOCK_SIZE - offset - 1;
    odata[out_index] = sh_mem[sh_mem_index];
  }
}

template <typename T>
__global__ void MatrixRowReverse(const T* matrix_data, T* reverse_data,
                                 int reverse_size, int outer_size,
                                 int inner_size) {
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int item_per_block = 1024;

  for (int block_offset = 0; block_offset < reverse_size;
       block_offset += item_per_block) {
    int valid_item = (reverse_size - block_offset > item_per_block)
                         ? item_per_block
                         : reverse_size - block_offset;
    int src_offset =
        bx * reverse_size + block_offset + by * (inner_size * reverse_size);
    int dst_offset = bx * reverse_size + by * (inner_size * reverse_size) +
                     reverse_size - 1 - block_offset;
    if (reverse_size < item_per_block) {
      valid_item = reverse_size;
    }

    BlockReverse<T, 1024>(matrix_data, reverse_data, src_offset, dst_offset,
                          valid_item);
  }
}

template <typename T>
struct BlockPrefixCallbackOp {
  // Running prefix
  T running_total;
  // Constructor
  __device__ BlockPrefixCallbackOp(T running_total)
      : running_total(running_total) {}
  // Callback operator to be entered by the first warp of threads in the block.
  // Thread-0 is responsible for returning a value for seeding the block-wide
  // scan.
  __device__ T operator()(T block_aggregate) {
    T old_prefix = running_total;
    running_total = old_prefix + block_aggregate;
    return old_prefix;
  }
};

// No bank-conflict transpose
template <typename T, int TILE_DIM, int BLOCK_ROWS>
__global__ void MatrixTranspose(T* odata, const T* idata, size_t height,
                                size_t width) {
  __shared__ T tile[TILE_DIM][TILE_DIM + 1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    if (x < width && (y + j) < height) {
      tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];
    } else {
      tile[threadIdx.y + j][threadIdx.x] = 0;
    }
  }

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    if (x < height && (y + j) < width) {
      odata[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
    }
  }
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void BlockScanKernel(T* d_out, const T* d_in, int inner_size,
                                int outer_size, int scan_size, bool exclusive) {
  // Specialize BlockLoad, BlockStore, and BlockRadixSort collective types
  typedef cub::BlockLoad<T, BLOCK_THREADS, ITEMS_PER_THREAD,
                         cub::BLOCK_LOAD_TRANSPOSE>
      BlockLoadT;
  typedef cub::BlockStore<T, BLOCK_THREADS, ITEMS_PER_THREAD,
                          cub::BLOCK_STORE_TRANSPOSE>
      BlockStoreT;
  typedef cub::BlockScan<T, BLOCK_THREADS> BlockScanT;
  // Allocate type-safe, repurposable shared memory for collectives
  __shared__ union {
    typename BlockLoadT::TempStorage load;
    typename BlockStoreT::TempStorage store;
    typename BlockScanT::TempStorage scan;
  } temp_storage;

  int bx = blockIdx.x;
  int by = blockIdx.y;

  BlockPrefixCallbackOp<T> prefix_op(0);
  T block_aggregate = static_cast<T>(0);

  // Obtain this block's segment of consecutive keys (blocked across threads)
  int item_per_block = BLOCK_THREADS * ITEMS_PER_THREAD;
  for (int block_offset = 0; block_offset < scan_size;
       block_offset += BLOCK_THREADS * ITEMS_PER_THREAD) {
    int valid_item = (scan_size - block_offset > item_per_block)
                         ? item_per_block
                         : (scan_size - block_offset);
    if (scan_size < item_per_block) {
      valid_item = scan_size;
    }

    int offset = bx * scan_size + block_offset + by * (inner_size * scan_size);

    T thread_keys[ITEMS_PER_THREAD];
    BlockLoadT(temp_storage.load)
        .Load(d_in + offset, thread_keys, valid_item, 0);

    __syncthreads();
    if (exclusive) {
      T init_value = static_cast<T>(0);
      BlockScanT(temp_storage.scan)
          .ExclusiveScan(thread_keys, thread_keys, cub::Sum(), prefix_op);
    } else {
      BlockScanT(temp_storage.scan)
          .InclusiveScan(thread_keys, thread_keys, cub::Sum(), prefix_op);
    }
    __syncthreads();

    BlockStoreT(temp_storage.store)
        .Store(d_out + offset, thread_keys, valid_item);
  }
}

template <typename DeviceContext, typename T>
class CumCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<framework::Tensor>("X");
    auto* out = context.Output<framework::Tensor>("Out");

    int axis = context.Attr<int>("axis");
    bool exclusive = context.Attr<bool>("exclusive");
    bool reverse = context.Attr<bool>("reverse");
    auto out_dims = out->dims();
    auto size = in->numel();

    PADDLE_ENFORCE_EQ(
        axis < out_dims.size() && axis >= (0 - out_dims.size()), true,
        platform::errors::OutOfRange(
            "Attr(axis) is out of range, It's expected "
            "to be in range of [-%d, %d]. But received Attr(axis) = %d.",
            out_dims.size(), out_dims.size() - 1, axis));
    if (axis < 0) {
      axis += out_dims.size();
    }

    T* out_data = out->mutable_data<T>(context.GetPlace());
    const T* in_data = in->data<T>();

    // Use thrust for parallel acceleration when the input size is equal to the
    // length of the ‘axis’ dimension.
    if (size == out_dims[axis]) {
      if (reverse) {
        thrust::device_ptr<const T> dev_ptr =
            thrust::device_pointer_cast(in_data);
        thrust::device_vector<T> vec(dev_ptr, dev_ptr + size);
        if (exclusive) {
          thrust::exclusive_scan(thrust::device, vec.rbegin(), vec.rend(),
                                 out_data);
        } else {
          thrust::inclusive_scan(thrust::device, vec.rbegin(), vec.rend(),
                                 out_data);
        }
        thrust::reverse(thrust::device, out_data, out_data + size);
      } else {
        if (exclusive) {
          thrust::exclusive_scan(thrust::device, in_data, in_data + size,
                                 out_data);
        } else {
          thrust::inclusive_scan(thrust::device, in_data, in_data + size,
                                 out_data);
        }
      }
      return;
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

    int tile_size = 32;
    dim3 blocks(32, 8);
    dim3 transpose_grids((width + tile_size - 1) / tile_size,
                         (height + tile_size - 1) / tile_size);
    auto& dev_ctx = context.template device_context<DeviceContext>();
    framework::Tensor tmp;
    tmp.Resize(out_dims);
    auto* tmp_data = tmp.mutable_data<T>(context.GetPlace());
    T* next_in_data = out_data;
    T* next_out_data = tmp_data;
    if (transpose) {
      MatrixTranspose<T, 32,
                      8><<<transpose_grids, blocks, 0, dev_ctx.stream()>>>(
          out_data, in_data, height, width);
      next_in_data = out_data;
      next_out_data = tmp_data;
    }
    auto swap_ptr = [](T*& ptr1, T*& ptr2) {
      T* tmp = ptr2;
      ptr2 = ptr1;
      ptr1 = tmp;
    };
    int outer_size = height / scan_size;
    int inner_size = width;
    // Consider the size of shared memory, here block size is 128
    dim3 scan_grid(outer_size, inner_size);
    dim3 reverse_grid = scan_grid;
    if (reverse) {
      if (transpose) {
        reverse_grid.x = scan_grid.y;
        reverse_grid.y = scan_grid.x;
        MatrixRowReverse<T><<<reverse_grid, 1024, 0, dev_ctx.stream()>>>(
            next_in_data, next_out_data, scan_size, outer_size, inner_size);
        if (!transpose) next_in_data = tmp_data;
        swap_ptr(next_in_data, next_out_data);
      } else {
        MatrixRowReverse<T><<<reverse_grid, 1024, 0, dev_ctx.stream()>>>(
            in_data, out_data, scan_size, outer_size, inner_size);
      }
    }
    if (!transpose && !reverse) {
      BlockScanKernel<T, 128, 4><<<scan_grid, 128, 0, dev_ctx.stream()>>>(
          out_data, in_data, outer_size, inner_size, scan_size, exclusive);

    } else {
      BlockScanKernel<T, 128, 4><<<scan_grid, 128, 0, dev_ctx.stream()>>>(
          next_out_data, next_in_data, outer_size, inner_size, scan_size,
          exclusive);
    }
    swap_ptr(next_in_data, next_out_data);
    if (reverse) {
      MatrixRowReverse<T><<<reverse_grid, 1024, 0, dev_ctx.stream()>>>(
          next_in_data, next_out_data, scan_size, outer_size, inner_size);
      swap_ptr(next_in_data, next_out_data);
    }
    if (transpose) {
      transpose_grids.x = (height + tile_size - 1) / tile_size;
      transpose_grids.y = (width + tile_size - 1) / tile_size;
      MatrixTranspose<T, 32,
                      8><<<transpose_grids, blocks, 0, dev_ctx.stream()>>>(
          next_out_data, next_in_data, width, height);
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    cumsum, ops::CumCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::CumCUDAKernel<paddle::platform::CUDADeviceContext, double>,
    ops::CumCUDAKernel<paddle::platform::CUDADeviceContext, int>,
    ops::CumCUDAKernel<paddle::platform::CUDADeviceContext, int64_t>);
