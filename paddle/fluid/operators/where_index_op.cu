/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <thrust/scan.h>
#include <algorithm>
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/where_index_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

using CUDADeviceContext = paddle::platform::CUDADeviceContext;

template <typename T>
struct CheckTrue {
  __host__ __device__ bool operator()(const T &val) {
    return static_cast<bool>(val);
  }
};
template <typename T>
__global__ void KeGetTrueNum(const T *cond_data, const int64_t numel,
                             int64_t *true_num_array) {
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int64_t idx = tid; idx < numel; idx += gridDim.x * blockDim.x) {
    true_num_array[idx] = CheckTrue<T>()(cond_data[idx]) ? 1 : 0;
  }
}
template <typename T, int BLOCKDIM>
__global__ void KeSetTrueIndex(int64_t *out_ptr, const T *cond_data,
                               const int64_t numel, const int64_t *ptr_stride,
                               const int64_t rank,
                               const int64_t *true_num_array,
                               const int64_t *block_reduce_sum) {
  const int tid = threadIdx.x;
  constexpr int tile_size = BLOCKDIM * 2;
  const int end_bid = (numel + tile_size - 1) / tile_size;

#pragma unroll
  for (int bid = blockIdx.x; bid < end_bid; bid += gridDim.x) {
    const int64_t tile_beg = bid * tile_size;
    const int64_t index = 2 * tid + tile_beg;

    const int64_t prefix_sum = block_reduce_sum[bid];
#pragma unroll
    for (int64_t idx = index; idx < index + 2 && idx < numel; idx++) {
      if (CheckTrue<T>()(cond_data[idx])) {
        int64_t rank_index = idx;
        const int64_t true_index = true_num_array[idx] + prefix_sum;
        for (int j = 0; j < rank; j++) {
          const int64_t out_index = rank_index / ptr_stride[j];
          out_ptr[true_index * rank + j] = out_index;
          rank_index -= out_index * ptr_stride[j];
        }
      }
    }
  }
}
// From [Parallel Prefix Sum (Scan) with CUDA]
// (https://www.eecs.umich.edu/courses/eecs570/hw/parprefix.pdf)
// TODO(jiangcheng): avoiding bank conflicts
template <typename T, int BLOCKDIM>
__device__ T BlockPrefixSum(T *data, const int64_t index, const int64_t i_end) {
  const int tid = threadIdx.x;
  constexpr int tile_size = BLOCKDIM * 2;
  __shared__ T s_data[tile_size];
  s_data[2 * tid] = 0;
  s_data[2 * tid + 1] = 0;

  if (index < i_end) s_data[2 * tid] = data[index];
  if (index + 1 < i_end) s_data[2 * tid + 1] = data[index + 1];

  int offset = 1;
  T reduce_sum = 0;
  for (int i = tile_size >> 1; i > 0; i >>= 1) {
    __syncthreads();
    if (tid < i) {
      int x = offset * (2 * tid + 1) - 1;
      int y = offset * (2 * tid + 2) - 1;

      s_data[y] += s_data[x];
    }
    offset <<= 1;
  }

  if (tid == 0) {
    reduce_sum = s_data[tile_size - 1];
    s_data[tile_size - 1] = 0;
  }

  for (int i = 1; i < tile_size; i <<= 1) {
    offset >>= 1;
    __syncthreads();

    if (tid < i) {
      int x = offset * (2 * tid + 1) - 1;
      int y = offset * (2 * tid + 2) - 1;

      T tmp = s_data[x];
      s_data[x] = s_data[y];
      s_data[y] += tmp;
    }
  }
  __syncthreads();
  if (index < i_end) data[index] = s_data[2 * tid];
  if (index + 1 < i_end) data[index + 1] = s_data[2 * tid + 1];

  return reduce_sum;
}
template <typename T, int BLOCKDIM>
__global__ void KeScanPrefixSum(T *data, const int64_t numel,
                                T *block_reduce_sum) {
  const int tid = threadIdx.x;
  constexpr int tile_size = BLOCKDIM * 2;
  const int end_bid = (numel + tile_size - 1) / tile_size;

  for (int bid = blockIdx.x; bid < end_bid; bid += gridDim.x) {
    const int64_t tile_beg = bid * tile_size;
    const int64_t tile_end = min(tile_beg + tile_size, numel);
    const int64_t index = 2 * tid + tile_beg;

    T b_sum = BlockPrefixSum<T, BLOCKDIM>(data, index, tile_end);
    if (tid == 0) block_reduce_sum[bid] = b_sum;
  }
}
template <typename T, int BLOCKDIM>
__global__ void KeScanPrefixSum_OneBlock(T *data, const int64_t numel,
                                         T *reduce_sum) {
  if (blockIdx.x == 0) {
    const int tid = threadIdx.x;
    constexpr int tile_size = BLOCKDIM * 2;
    const int end_bid = (numel + tile_size - 1) / tile_size;
    __shared__ T base_sum;
    base_sum = 0;
    __syncthreads();

    for (int bid = 0; bid < end_bid; bid++) {
      const int64_t tile_beg = bid * tile_size;
      const int64_t tile_end = min(tile_beg + tile_size, numel);
      const int64_t index = 2 * tid + tile_beg;

      T b_sum = BlockPrefixSum<T, BLOCKDIM>(data, index, tile_end);

      if (index < tile_end) data[index] += base_sum;
      if (index + 1 < tile_end) data[index + 1] += base_sum;
      __syncthreads();
      if (tid == 0) base_sum += b_sum;
      __syncthreads();
    }
    reduce_sum[0] = base_sum;
  }
}

template <typename T, int BLOCKDIM>
inline void KeWhereIndex(int64_t *out_ptr, int64_t *d_true_num,
                         const T *cond_data, const int64_t numel,
                         const int64_t *h_stride, const int64_t rank,
                         const framework::ExecutionContext &context) {
  auto &dev_ctx = context.template device_context<CUDADeviceContext>();
  constexpr int tile_size = BLOCKDIM * 2;
  const int64_t need_blocks = (numel + tile_size - 1) / tile_size;
  const int64_t grids = std::min(
      static_cast<int64_t>(dev_ctx.GetCUDAMaxGridDimSize().x), need_blocks);

  int64_t tmp_mem_size = rank + numel + need_blocks;
  auto d_tmp_mem = memory::Alloc(dev_ctx, tmp_mem_size * sizeof(int64_t));

  int64_t *ptr_stride = reinterpret_cast<int64_t *>(d_tmp_mem->ptr());
  int64_t *ptr_true_num_array = ptr_stride + rank;
  int64_t *ptr_block_reduce_sum = ptr_true_num_array + numel;

  memory::Copy(BOOST_GET_CONST(platform::CUDAPlace, dev_ctx.GetPlace()),
               ptr_stride, platform::CPUPlace(), h_stride,
               rank * sizeof(int64_t), dev_ctx.stream());
  KeGetTrueNum<T><<<grids, BLOCKDIM, 0, dev_ctx.stream()>>>(cond_data, numel,
                                                            ptr_true_num_array);
  KeScanPrefixSum<int64_t, BLOCKDIM><<<grids, BLOCKDIM, 0, dev_ctx.stream()>>>(
      ptr_true_num_array, numel, ptr_block_reduce_sum);
  KeScanPrefixSum_OneBlock<int64_t,
                           BLOCKDIM><<<1, BLOCKDIM, 0, dev_ctx.stream()>>>(
      ptr_block_reduce_sum, need_blocks, d_true_num);
  KeSetTrueIndex<T, BLOCKDIM><<<grids, BLOCKDIM, 0, dev_ctx.stream()>>>(
      out_ptr, cond_data, numel, ptr_stride, rank, ptr_true_num_array,
      ptr_block_reduce_sum);
}

template <typename T>
class CUDAWhereIndexKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *condition = context.Input<framework::Tensor>("Condition");
    auto *out = context.Output<framework::Tensor>("Out");
    auto &dev_ctx = context.template device_context<CUDADeviceContext>();

    const T *cond_data = condition->data<T>();
    const int64_t numel = condition->numel();
    auto dims = condition->dims();
    const int rank = dims.size();

    auto d_tmp_mem = memory::Alloc(dev_ctx, sizeof(int64_t));
    auto h_tmp_mem =
        memory::Alloc(platform::CPUPlace(), rank * sizeof(int64_t));
    int64_t *ptr_true_num = reinterpret_cast<int64_t *>(d_tmp_mem->ptr());
    int64_t *h_stride = reinterpret_cast<int64_t *>(h_tmp_mem->ptr());

    out->Resize(framework::make_ddim({static_cast<int64_t>(numel), rank}));
    auto out_ptr = out->mutable_data<int64_t>(context.GetPlace());

    h_stride[rank - 1] = 1;
    for (int i = rank - 2; i >= 0; i--) {
      h_stride[i] = h_stride[i + 1] * dims[i + 1];
    }

#define SelectBlockSize(BlockSize)                                    \
  KeWhereIndex<T, BlockSize>(out_ptr, ptr_true_num, cond_data, numel, \
                             h_stride, rank, context);

    if (numel / 1024 >= 2) {
      SelectBlockSize(1024)
    } else if (numel / 256 >= 2) {
      SelectBlockSize(256)
    } else {
      SelectBlockSize(32)
    }
#undef SelectBlockSize

    int64_t true_num = 0;
    memory::Copy(platform::CPUPlace(), &true_num,
                 BOOST_GET_CONST(platform::CUDAPlace, dev_ctx.GetPlace()),
                 ptr_true_num, sizeof(int64_t), dev_ctx.stream());
    dev_ctx.Wait();
    out->Resize(framework::make_ddim({static_cast<int64_t>(true_num), rank}));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(where_index, ops::CUDAWhereIndexKernel<int64_t>,
                        ops::CUDAWhereIndexKernel<int>,
                        ops::CUDAWhereIndexKernel<bool>,
                        ops::CUDAWhereIndexKernel<float>,
                        ops::CUDAWhereIndexKernel<double>);
