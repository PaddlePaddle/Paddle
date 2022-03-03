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

#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include <algorithm>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/where_index_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/fluid/platform/for_range.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/kernels/primitive/kernel_primitives.h"

namespace paddle {
namespace operators {

using CUDADeviceContext = paddle::platform::CUDADeviceContext;
namespace kps = phi::kps;
using Mode = kps::details::ReduceMode;

/*
* Count how many of the data being processed by the current block are true
* 1. Load data from global memory and cast from bool to int64_t
* 2. Get result of this thread according to thread reduce
* 3. Get result of this block according to block reduce
* 4. first block store 0 and current result
*/

template <typename InT, typename OutT, int VecSize, int IsBoundary>
__device__ void GetBlockCountImpl(const InT *in, OutT *out, int num,
                                  int repeat) {
  InT in_data[VecSize];
  OutT temp[VecSize];
  OutT result = static_cast<OutT>(0.0f);
  using Add = kps::AddFunctor<OutT>;
  using Cast = kps::IdentityFunctor<InT>;
  int store_fix = BLOCK_ID_X + repeat * GRID_NUM_X;

  kps::Init<InT, VecSize>(&in_data[0], static_cast<InT>(0.0f));
  kps::ReadData<InT, VecSize, 1, 1, IsBoundary>(&in_data[0], in, num);
  kps::ElementwiseUnary<InT, OutT, VecSize, 1, 1, Cast>(&temp[0], &in_data[0],
                                                        Cast());
  kps::Reduce<OutT, VecSize, 1, 1, Add, Mode::kLocalMode>(&result, &temp[0],
                                                          Add(), true);
  kps::Reduce<OutT, 1, 1, 1, Add, Mode::kGlobalMode>(&result, &result, Add(),
                                                     true);
  if (store_fix == 0) {
    // first block's fix_size = 0;
    OutT tmp = static_cast<OutT>(0.0f);
    kps::WriteData<OutT, 1, 1, 1, true>(out + store_fix, &tmp, 1);
  }
  // store num of this block
  kps::WriteData<OutT, 1, 1, 1, true>(out + store_fix + 1, &result, 1);
}

template <typename InT, typename OutT, int VecSize>
__global__ void GetBlockCountKernel(const InT *in, OutT *out, int numel,
                                    int main_offset) {
  int data_offset = BLOCK_ID_X * BLOCK_NUM_X * VecSize;
  int stride = BLOCK_NUM_X * GRID_NUM_X * VecSize;
  int repeat = 0;
  for (; data_offset < main_offset; data_offset += stride) {
    GetBlockCountImpl<InT, OutT, VecSize, false>(in + data_offset, out,
                                                 BLOCK_NUM_X * VecSize, repeat);
    repeat++;  // to get the real blockIdx
  }

  int num = numel - main_offset;
  if (num > 0) {
    GetBlockCountImpl<InT, OutT, VecSize, true>(in + data_offset, out, num,
                                                repeat);
  }
}

/*
* Get block num prefix us one block, VecSize must be 2
* 1. Each thread load 2 data : threadIdx.x and threadIdx.x + blockDimx.x
* 2. Cumsum limitation is blockDim.x must be less than 512
*/

template <typename InT, typename OutT, typename Functor, int VecSize,
          bool IsBoundary>
__device__ void CumsumImpl(const InT *in, OutT *out, OutT *pre_cumsum, int num,
                           Functor func) {
  __shared__ OutT max_thread_data;
  OutT temp[VecSize];
  InT arg[VecSize];
  OutT result[VecSize];
  // init data_pr
  kps::Init<InT, VecSize>(&arg[0], static_cast<InT>(0.0f));
  // set pre_cumsum
  kps::Init<OutT, VecSize>(&temp[0], *pre_cumsum);
  // load data to arg
  kps::ReadData<InT, InT, VecSize, 1, 1, IsBoundary>(&arg[0], in, num, 1,
                                                     BLOCK_NUM_X, 1);
  // block cumsum
  kps::Cumsum<InT, OutT, 1, Functor>(&result[0], &arg[0], func);
  // result = cumsum_result + pre_cumsum
  kps::ElementwiseBinary<OutT, OutT, VecSize, 1, 1, Functor>(
      &result[0], &result[0], &temp[0], func);
  // get the last prefix sum
  if ((THREAD_ID_X == BLOCK_NUM_X - 1) && !IsBoundary) {
    max_thread_data = result[VecSize - 1];
  }
  __syncthreads();
  // update pre_cumsum
  *pre_cumsum = max_thread_data;
  kps::WriteData<OutT, OutT, VecSize, 1, 1, IsBoundary>(out, &result[0], num, 1,
                                                        BLOCK_NUM_X, 1);
}

template <typename InT, typename OutT, typename Functor, int VecSize>
__global__ void CumsumOneBlock(const InT *in, OutT *out, int numel,
                               int main_offset, Functor func) {
  int stride = BLOCK_NUM_X * VecSize;
  int offset = 0;
  OutT pre_cumsum = static_cast<OutT>(0);
  for (; offset < main_offset; offset += stride) {
    CumsumImpl<InT, OutT, Functor, VecSize, false>(
        in + offset, out + offset, &pre_cumsum, BLOCK_NUM_X * VecSize, func);
  }

  int num = numel - offset;
  if (num > 0) {
    CumsumImpl<InT, OutT, Functor, VecSize, true>(in + offset, out + offset,
                                                  &pre_cumsum, num, func);
  }
}
/**
* Get mask's index if mask == true
*/

template <typename InT, typename MT, typename OutT, typename Functor,
          int VecSize, int MaskData,
          int IsBoundary>  // SelectType = 1 Mask_select else where_index
__device__ void
SelectKernelImpl(OutT *out, const MT *mask, InT *in, Functor func, int num,
                 int data_offset, int store_rank) {
  const int kCVecSize = 2;
  // each thread cumsum 2 data
  using IdT = int64_t;
  // Set index data type
  using Add = kps::AddFunctor<IdT>;       // for cumsum
  using Cast = kps::IdentityFunctor<MT>;  // for mask

  IdT init_idx = static_cast<IdT>(0.0f);
  MT init_mask = static_cast<MT>(0.0f);

  IdT num_thread[kCVecSize];
  IdT cumsum_thread[kCVecSize];

  IdT index_reg[VecSize];
  OutT store_data[VecSize * framework::DDim::kMaxRank];
  InT in_data[VecSize];
  MT mask_data[VecSize];
  IdT mask_idt[VecSize];
  // init
  // init data_pr
  kps::Init<IdT, kCVecSize>(&cumsum_thread[0], init_idx);
  kps::Init<IdT, kCVecSize>(&num_thread[0], init_idx);
  kps::Init<MT, VecSize>(&mask_data[0], init_mask);
  // Load mask
  kps::ReadData<MT, VecSize, 1, 1, IsBoundary>(&mask_data[0], mask, num);
  // Cast from MT to int
  kps::ElementwiseUnary<MT, IdT, VecSize, 1, 1, Cast>(&mask_idt[0],
                                                      &mask_data[0], Cast());
  // Get the num of thread only num_thread[1] has data
  kps::Reduce<IdT, VecSize, 1, 1, Add, Mode::kLocalMode>(
      &num_thread[0], &mask_idt[0], Add(), true);
  // Get cumsum_thread cumsum from 0 to num_thread cumsum_thread[0] is the
  // thread_fix
  kps::Cumsum<IdT, IdT, 1, Add>(&cumsum_thread[0], &num_thread[0], Add());
  // Set data index of global
  kps::InitWithDataIndex<IdT, VecSize, 1, 1>(&index_reg[0], data_offset);
  // Get store data(index) according to mask_idt
  kps::OperatorTernary<MT, IdT, OutT, Functor>(&store_data[0], &mask_data[0],
                                               &index_reg[0], func, VecSize);
  // get thread_fix
  int thread_fix =
      (static_cast<int>(cumsum_thread[0] - num_thread[0]) * store_rank);
  // get how many data need to store
  int store_num = static_cast<int>(num_thread[0]) * store_rank;
  // thread store num data, each thread may has different num
  kps::details::WriteData<OutT>(out + thread_fix, &store_data[0], store_num);
}

template <typename MT, typename InT, typename OutT, typename Functor,
          int VecSize, int MaskData>
__global__ void SelectKernel(int64_t *out, const MT *mask, InT *in, InT *cumsum,
                             Functor func, const int64_t numel, int main_offset,
                             int store_rank) {
  int data_offset = BLOCK_ID_X * BLOCK_NUM_X * VecSize;
  int stride = BLOCK_NUM_X * GRID_NUM_X * VecSize;
  int repeat = 0;
  int size = VecSize * BLOCK_ID_X;
  for (; data_offset < main_offset; data_offset += stride) {
    // Cumsum index
    int idx_cumsum = repeat * GRID_NUM_X + BLOCK_ID_X;
    // niuliling todo: us ReadData API
    int block_store_offset = cumsum[idx_cumsum];
    SelectKernelImpl<InT, MT, OutT, Functor, VecSize, MaskData, false>(
        out + block_store_offset * store_rank, mask + data_offset,
        in + data_offset, func, size, data_offset, store_rank);
    repeat++;
  }

  int num = numel - data_offset;
  if (num > 0) {
    // Cumsum index
    int idx_cumsum = repeat * GRID_NUM_X + BLOCK_ID_X;
    // niuliling todo: us ReadData API
    int block_store_offset = static_cast<int>(cumsum[idx_cumsum]);
    SelectKernelImpl<InT, MT, OutT, Functor, VecSize, MaskData, true>(
        out + block_store_offset * store_rank, mask + data_offset,
        in + data_offset, func, num, data_offset, store_rank);
  }
}

template <typename T1, typename T2, typename OutT>
struct IndexFunctor {
  T2 stride[paddle::framework::DDim::kMaxRank];
  int dims;
  HOSTDEVICE IndexFunctor(const framework::DDim &in_dims) {
    dims = in_dims.size();
    std::vector<T2> strides_in_tmp;
    strides_in_tmp.resize(dims, 1);
    // get strides according to in_dims
    for (T2 i = 1; i < dims; i++) {
      strides_in_tmp[i] = strides_in_tmp[i - 1] * in_dims[dims - i];
    }
    memcpy(stride, strides_in_tmp.data(), dims * sizeof(T2));
  }

  HOSTDEVICE inline void operator()(OutT *out, const T1 *mask, const T2 *index,
                                    const int num) {
    int store_fix = 0;
    for (int idx = 0; idx < num; idx++) {
      if (mask[idx]) {
        T2 data_index = index[idx];
        // get index
        for (int rank_id = dims - 1; rank_id >= 0; --rank_id) {
          out[store_fix] = static_cast<OutT>(data_index / stride[rank_id]);
          data_index = data_index % stride[rank_id];
          store_fix++;
        }
      }
    }
  }
};

template <typename T>
class CUDAWhereIndexKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *condition = context.Input<framework::Tensor>("Condition");
    auto *out = context.Output<framework::Tensor>("Out");
    auto &dev_ctx = context.template device_context<CUDADeviceContext>();

    auto stream = dev_ctx.stream();
    const T *cond_data = condition->data<T>();
    const int64_t numel = condition->numel();
    auto dims = condition->dims();
    int rank = dims.size();
    // alloc for cpu
    auto h_array_mem =
        memory::Alloc(platform::CPUPlace(), (rank + 1) * sizeof(int64_t));
    int64_t *h_stride_array = reinterpret_cast<int64_t *>(h_array_mem->ptr());
    // calculate the inclusive prefix sum of "true_num_array"
    // to get the index of "out" tensor,
    // and the total number of cond_data[i]==true.
    // Example:
    // condition: F T T F F F T T
    // before:    0 1 1 0 0 0 1 1
    // after:     0 1 2 2 2 2 3 4
    // out:       1 2 6 7
    // 1.1 get stored data num of per block
    const int VecSize = 4;
    int block = 256;
    int num_per_block = VecSize * block;
    const int64_t need_grids = (numel + num_per_block - 1) / num_per_block;
    const int grid = std::min(need_grids, static_cast<int64_t>(256));
    int main_offset = (numel) / num_per_block * num_per_block;
    // 1.2 alloc tmp data for CoutBlock
    int size_count_block = need_grids + 1;
    auto count_mem = memory::Alloc(dev_ctx, size_count_block * sizeof(int64_t));
    int64_t *count_data = reinterpret_cast<int64_t *>(count_mem->ptr());
    // 1.3 launch CountKernl
    GetBlockCountKernel<T, int64_t, VecSize><<<grid, block, 0, stream>>>(
        cond_data, count_data, numel, main_offset);
    memory::Copy(platform::CPUPlace(), h_stride_array, dev_ctx.GetPlace(),
                 count_data, sizeof(int64_t), dev_ctx.stream());
    // 2.1 alloc cumsum data for CoutBlock prefix
    auto cumsum_mem =
        memory::Alloc(dev_ctx, size_count_block * sizeof(int64_t));
    int64_t *cumsum_data = reinterpret_cast<int64_t *>(cumsum_mem->ptr());
    // 2.2 get prefix of count_data for real out_index
    int block_c = 256;
    int main_offset_c = size_count_block / (2 * block_c) * (2 * block_c);
    CumsumOneBlock<int64_t, int64_t, kps::AddFunctor<int64_t>,
                   2><<<1, block_c, 0, stream>>>(
        count_data, cumsum_data, size_count_block, main_offset_c,
        kps::AddFunctor<int64_t>());
    // 3.1 set temp ptr for in;
    // 3.1 alloc for out
    // 3.1.1 get true_num for gpu place the last cumsum is the true_num
    memory::Copy(platform::CPUPlace(), h_stride_array + 1, dev_ctx.GetPlace(),
                 cumsum_data + need_grids, sizeof(int64_t), dev_ctx.stream());
    // 3.1.2 allock for out with total_true_num
    out->Resize(
        phi::make_ddim({static_cast<int64_t>(h_stride_array[1]), rank}));
    auto out_data = out->mutable_data<int64_t>(context.GetPlace());
    // 3.2 get true data's index according to cond_data and cumsum_data
    int64_t *tmp_in = nullptr;
    using Functor = IndexFunctor<T, int64_t, int64_t>;
    Functor index_functor = Functor(dims);
    SelectKernel<T, int64_t, int64_t, Functor, VecSize,
                 0><<<grid, block, 0, stream>>>(out_data, cond_data, tmp_in,
                                                cumsum_data, index_functor,
                                                numel, main_offset, rank);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(where_index, ops::CUDAWhereIndexKernel<int64_t>,
                        ops::CUDAWhereIndexKernel<int>,
                        ops::CUDAWhereIndexKernel<int16_t>,
                        ops::CUDAWhereIndexKernel<bool>,
                        ops::CUDAWhereIndexKernel<float>,
                        ops::CUDAWhereIndexKernel<double>);
