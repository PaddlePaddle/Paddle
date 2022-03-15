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
template<typename T>
struct NonZeroFunctor{
  HOSTDEVICE NonZeroFunctor() {}
  HOSTDEVICE inline T operator()(const T in) {
	  if (in) {
	    return static_cast<T>(1.0f);
	  } else {
	    return static_cast<T>(0.0f);
	  }
  }
};

template <typename T>
__global__ void GetTrueNum(const T *cond_data, const int64_t numel,
                           int64_t *true_num_array) {
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int64_t idx = tid; idx < numel; idx += gridDim.x * blockDim.x) {
    true_num_array[idx] =
        static_cast<int64_t>(static_cast<bool>(cond_data[idx]));
  }
}

template <typename T>
__global__ void SetTrueIndex(int64_t *out_ptr, const T *cond_data,
                             const int64_t numel, const int64_t *stride_array,
                             const int64_t rank,
                             const int64_t *true_num_array) {
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int64_t idx = tid; idx < numel; idx += gridDim.x * blockDim.x) {
    // true_num_array is calculated by cub::InclusiveSum,
    // cause the first element of true_num_array is 1,
    // so we need substract 1 to get true index.
    const int64_t true_index = true_num_array[idx] - 1;
    if (static_cast<bool>(cond_data[idx])) {
      int64_t rank_index = idx;
      for (int j = 0; j < rank; j++) {
        const int64_t out_index = rank_index / stride_array[j];
        out_ptr[true_index * rank + j] = out_index;
        rank_index -= out_index * stride_array[j];
      }
    }
  }
}
template <typename InT, typename OutT, int VecSize, int IsBoundary>
__device__ void GetBlockCountImpl(const InT *in, OutT *out, int num,
                                  int repeat) {
  InT in_data[VecSize];
  OutT temp[VecSize];
  OutT result = static_cast<OutT>(0.0f);
  using Add = kps::AddFunctor<OutT>;
  using Cast = NonZeroFunctor<InT>;
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
  kps::WriteData<OutT, 1, 1, 1, true>(out + store_fix + 1, &result, 1);
}

/*
* get block_num_fix from 0 to total_true_num
*/
template <typename InT, typename OutT, int VecSize>
__global__ void GetBlockCountKernel(const InT *in, OutT *out, int numel,
                                    int main_offset) {
  int data_offset = BLOCK_ID_X * BLOCK_NUM_X * VecSize;
  int stride = BLOCK_NUM_X * GRID_NUM_X * VecSize;
  int repeat = 0;
  for (; data_offset < main_offset; data_offset += stride) {
    GetBlockCountImpl<InT, OutT, VecSize, false>(in + data_offset, out,
                                                 BLOCK_NUM_X * VecSize, repeat);
    repeat++;
  }

  int num = numel - main_offset;
  if (num > 0) {
    GetBlockCountImpl<InT, OutT, VecSize, true>(in + data_offset, out, num,
                                                repeat);
  }
}

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
  //kps::ReadData<InT, 1, 1, 1, IsBoundary>(&arg[0], in, num);
  kps::ReadData<InT, InT, VecSize, 1, 1, IsBoundary>(&arg[0], in, num, 1, BLOCK_NUM_X, 1);

  if (threadIdx.x <= 2) 
  printf("ffffff cumsums_0 %ld, cumsum_1 %d %ld arg_0 %ld arg_1 %ld\n", result[0], result[1], threadIdx.x, arg[0], arg[1]);
  // block cumsum
  kps::Cumsum<InT, OutT, 1, Functor>(&result[0], &arg[0], func);
  // result = cumsum_result + pre_cumsum
  kps::ElementwiseBinary<OutT, OutT, VecSize, 1, 1, Functor>(
      &result[0], &result[0], &temp[0], func);
  if ((THREAD_ID_X == BLOCK_NUM_X - 1) && !IsBoundary) {
    max_thread_data = result[VecSize - 1];
  }
  __syncthreads();
  // update pre_cumsum
  *pre_cumsum = max_thread_data;
  if (threadIdx.x <= 2) 
  printf("ffffff cumsums_0 %ld, cumsum_1 %d %ld arg_0 %ld arg_1 %ld\n", result[0], result[1], threadIdx.x, arg[0], arg[1]);
  kps::WriteData<OutT, OutT, VecSize, 1, 1, IsBoundary>(out, &result[0], num, 1, BLOCK_NUM_X, 1);
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

template <typename InT, typename MT, typename OutT, typename Functor,
          int VecSize, int MaskData,
          int IsBoundary>  // SelectType = 1 Mask_select else where_index
__device__ void
SelectKernelImpl(OutT *out, const MT *mask, InT *in, Functor &func, int num,
                 int data_offset, int store_rank) {
  const int kCVecSize = 2;  // each thread cumsum 2 data
  using IdT = int64_t;  // Set index data type
  using Add = kps::AddFunctor<IdT>;
  using Cast = NonZeroFunctor<InT>;

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
  kps::Reduce<IdT, VecSize, 1, 1, Add, Mode::kLocalMode>(&num_thread[0], &mask_idt[0], Add(), true);
  // Get cumsum_thread cumsum from 0 to num_thread cumsum_thread[0] is the thread_fix
  kps::Cumsum<IdT, IdT, 1, Add>(&cumsum_thread[0], &num_thread[0], Add());
  // if (MaskData) {
  //  // Load mask
  //  kps::ReadData<InT, VecSize, 1, 1, IsBoundary>(&in_data[0], in, num);
  //  // Get store data according to mask_idt
  //  kps::OperatorBinary<InT, OutT, Functor>(
  //      &store_data[0], &in_data[0], func, static_cast<int>(num_thread[0]));
  //} else {  // get the index when mask[i] = true
  // Set data index of global
  kps::InitWithDataIndex<IdT, VecSize, 1, 1>(&index_reg[0], data_offset);
  // Get store data(index) according to mask_idt
  kps::OperatorTernary<MT, IdT, OutT, Functor>(&store_data[0], &mask_data[0],
                                               &index_reg[0], func, VecSize);
  int thread_fix = (static_cast<int>(cumsum_thread[0] - num_thread[0]) * store_rank);
  int store_num = static_cast<int>(num_thread[0]) * store_rank;
  if (num_thread[1]) {
	printf("asdf*** thread_id %d, cumsum_0 %d, cumsum_1 %d, thread_fix %d data_offset %d\n", threadIdx.x, (int)(cumsum_thread[0]),(int)(cumsum_thread[1]), thread_fix, data_offset);
    printf("asdf***i %d index_reg_0 %d, index_reg_1 %d, num_thread_0 %d, num_thread_1 %d, cumsum_0 %d cumsum_1 %d\n",data_offset, (int)(index_reg[0]), (int)(index_reg[1]), (int)(num_thread[0]),(int)(num_thread[1]), (int)(cumsum_thread[0]), (int)(cumsum_thread[1]));

    printf("asdfl idx %d  cumsum_0 %d, cumsusm_1 %d store_num %d\n", threadIdx.x, (int)(cumsum_thread[0]), (int)(cumsum_thread[1]), store_num);
    printf("asdfl num_0 %d cumsum_0 %d thread_fix %d num_1 %d cumsum_1%d\n", 
           (int)num_thread[0],(int)(cumsum_thread[0]), thread_fix,  (int)(num_thread[1]), (int)(cumsum_thread[1]));
  }
  // }
  // thread store num data, each thread may has different num
  kps::details::WriteData<OutT>(out + thread_fix, &store_data[0], store_num);
}

template <typename MT, typename InT, typename OutT, typename Functor,
          int VecSize,
          int MaskData>  // SelectType = 1 Mask_select else where_index
__global__ void
SelectKernel(int64_t *out, const MT *mask, InT *in, InT *cumsum, Functor func,
             const int64_t numel, int main_offset, int store_rank) {
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
    if (threadIdx.x == 1) printf("LLLLLLLLL idx %d offset %d\n", idx_cumsum, block_store_offset * store_rank);
    SelectKernelImpl<InT, MT, OutT, Functor, VecSize, MaskData, true>(
        out + block_store_offset * store_rank, mask + data_offset,
        in + data_offset, func, num, data_offset, store_rank);
  }
}

template <typename T>
class CUDAWhereIndexKernelP : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *condition = context.Input<framework::Tensor>("Condition");
    auto *out = context.Output<framework::Tensor>("Out");
    auto &dev_ctx = context.template device_context<CUDADeviceContext>();

    const T *cond_data = condition->data<T>();
    const int64_t numel = condition->numel();
    auto dims = condition->dims();
    const int rank = dims.size();
    int vec_size = 8;
    auto config =
        phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, numel, vec_size);
    int max_grid = 1024;
    int grid = config.block_per_grid.x;
    grid = std::min(grid, max_grid);
    int block = config.thread_per_block.x;

    int main_offset = (numel / (vec_size * block)) * vec_size * block;
    auto block_count = memory::Alloc(dev_ctx, (grid) * sizeof(int));
    // launch count kernel

    // luanch Cumsum Kernel
    //

    auto d_array_mem = memory::Alloc(dev_ctx, (numel + rank) * sizeof(int64_t));
    auto h_array_mem =
        memory::Alloc(platform::CPUPlace(), (rank + 1) * sizeof(int64_t));

    // "stride_array" is an array and len(stride_array)==rank,
    // each element is the stride of each dimension -- the length from i to i+1.
    int64_t *h_stride_array = reinterpret_cast<int64_t *>(h_array_mem->ptr());
    int64_t *d_stride_array = reinterpret_cast<int64_t *>(d_array_mem->ptr());

    // "true_num_array" is an array and len(stride_array)==numel,
    // at the beginning,
    // "true_num_array" will set 1 if condition[i] == true else 0,
    // then it will be calculated by cub::InclusiveSum,
    // so that we can get the true number before i as the out index
    int64_t *d_true_num_array = d_stride_array + rank;

    // the total_true_num is the total number of condition[i] == true
    int64_t *h_total_true_num = h_stride_array + rank;

    // alloce cub memory
    size_t cub_size = 0;
    cub::DeviceScan::InclusiveSum(nullptr, cub_size, d_true_num_array,
                                  d_true_num_array, numel, dev_ctx.stream());
    auto cub_mem = memory::Alloc(dev_ctx, cub_size * sizeof(int64_t));
    void *cub_data = cub_mem->ptr();

    // set d_true_num_array[i]=1 if cond_data[i]==true else 0
    const int threads = std::min(numel, static_cast<int64_t>(128));
    const int64_t need_grids = (numel + threads - 1) / threads;
    const int grids = std::min(need_grids, static_cast<int64_t>(256));
    GetTrueNum<T><<<grids, threads, 0, dev_ctx.stream()>>>(cond_data, numel,
                                                           d_true_num_array);

    // calculate the inclusive prefix sum of "true_num_array"
    // to get the index of "out" tensor,
    // and the total number of cond_data[i]==true.
    // Example:
    // condition: F T T F F F T T
    // before:    0 1 1 0 0 0 1 1
    // after:     0 1 2 2 2 2 3 4
    // out:       1 2 6 7
    cub::DeviceScan::InclusiveSum(cub_data, cub_size, d_true_num_array,
                                  d_true_num_array, numel, dev_ctx.stream());

    // calculate each dimension's stride
    h_stride_array[rank - 1] = 1;
    for (int i = rank - 2; i >= 0; i--) {
      h_stride_array[i] = h_stride_array[i + 1] * dims[i + 1];
    }
    memory::Copy(dev_ctx.GetPlace(), d_stride_array, platform::CPUPlace(),
                 h_stride_array, rank * sizeof(int64_t), dev_ctx.stream());

    // get total ture number and set output size
    // the last element of cub::InclusiveSum is the total number
    memory::Copy(platform::CPUPlace(), h_total_true_num, dev_ctx.GetPlace(),
                 d_true_num_array + numel - 1, sizeof(int64_t),
                 dev_ctx.stream());
    dev_ctx.Wait();

    int64_t true_num = *h_total_true_num;
    out->Resize(phi::make_ddim({static_cast<int64_t>(true_num), rank}));
    auto out_data = out->mutable_data<int64_t>(context.GetPlace());

    if (true_num == 0) {
      return;
    }

    // using true_num_array and stride_array to calculate the output index
    SetTrueIndex<T><<<grids, threads, 0, dev_ctx.stream()>>>(
        out_data, cond_data, numel, d_stride_array, rank, d_true_num_array);
  }
};
template <typename T1, typename T2, typename OutT>
struct IndexFunctor {
  T2 stride[paddle::framework::DDim::kMaxRank];
  int dims;
  HOSTDEVICE IndexFunctor(framework::DDim &in_dims) {
    dims = in_dims.size();
    std::vector<T2> strides_in_tmp;
    strides_in_tmp.resize(dims, 1);
    for (T2 i = 1; i < dims; i++) {
      strides_in_tmp[i] = strides_in_tmp[i - 1] * in_dims[dims - i];
    }
    for (int i = 0; i < dims; i++) {
      printf("pppppppp   %d stride %d idx\n", strides_in_tmp[i], i);
    }
    memcpy(stride, strides_in_tmp.data(), dims * sizeof(T2));
  }

  HOSTDEVICE inline void operator()(OutT *out, const T1 *mask, const T2 *index,
                                    const int num) {
    int store_fix = 0;
    for (int idx = 0; idx < num; idx++) {
      if (mask[idx]) {  // if mask == true then store index
        T2 data_index = index[idx];
        for (int rank_id = dims - 1; rank_id >= 0; --rank_id) {
          out[store_fix] = static_cast<OutT>(data_index / stride[rank_id]);
          printf(
              "store_fix %d, data_index %d, out[store_fix] %d, rank_i %d, "
              "stride[rank_id] %d %d \n",
              store_fix, (int)data_index, (int)out[store_fix], rank_id,
              (int)stride[rank_id], threadIdx.x);
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
    std::cout << "asdf 1.2 need_grids " << need_grids << " grid " << grid
              << " block " << block << "  num " << numel << " main_offset "
              << main_offset << std::endl;
    using int64_t = int64_t;
    int size_count_block = need_grids + 1;
    auto count_mem =
        memory::Alloc(dev_ctx, size_count_block * sizeof(int64_t));
    int64_t *count_data = (int64_t *)(count_mem->ptr());
    std::cout << "asdf 1.2 count_mem after " << std::endl;
    // 1.3launch CountKernl
    GetBlockCountKernel<T, int64_t, VecSize><<<grid, block, 0, stream>>>(
        cond_data, count_data, numel, main_offset);
    std::cout << "asdf 1.3 GetBlockCountKernel after " << std::endl;
    memory::Copy(platform::CPUPlace(), h_stride_array, dev_ctx.GetPlace(),
                 count_data, sizeof(int64_t), dev_ctx.stream());
    std::cout << "asdf 1.4 GetBlockCountKernel after count_data "
              << h_stride_array[0] << std::endl;
    // 2.1 alloc cumsum data for CoutBlock prefix
    auto cumsum_mem =
        memory::Alloc(dev_ctx, size_count_block * sizeof(int64_t));
    int64_t *cumsum_data = (int64_t *)(cumsum_mem->ptr());
    std::cout << "asdf 2.1 cumsum after " << std::endl;
    // 2.2 get prefix of count_data for real out_index
    int block_c = 256;
    int main_offset_c = size_count_block/ (2 * block_c) * (2 * block_c);
    std::cout << "asdf 2.2 main_offset_c " << main_offset_c << " " << std::endl;
    CumsumOneBlock<int64_t, int64_t, kps::AddFunctor<int64_t>,
                   2><<<1, block_c, 0, stream>>>(
        count_data, cumsum_data, size_count_block, main_offset_c,
        kps::AddFunctor<int64_t>());
    std::cout << "asdf 2.2 CumsumOneBlock after " << std::endl;
    // 3.1 set temp ptr for in;
    // 3.1 alloc for out
    // 3.1.1 get true_num for gpu place the last cumsum is the true_num
    memory::Copy(platform::CPUPlace(), h_stride_array + 1, dev_ctx.GetPlace(),
                 cumsum_data + need_grids, sizeof(int64_t),
                 dev_ctx.stream());
    std::cout << "asdf 3.1.1 Copy get total_true_num " << h_stride_array[1]
              << std::endl;
    // 3.1.2 allock for out with total_true_num
    out->Resize(
        phi::make_ddim({static_cast<int64_t>(h_stride_array[1]), rank}));
    auto out_data = out->mutable_data<int64_t>(context.GetPlace());
    std::cout << "asdf 3.1.2 out resize " << h_stride_array[1] * rank
              << " rank " << rank << std::endl;
    // 3.2 get true data's index according to cond_data and cumsum_data
    int64_t *tmp_in = nullptr;
    using Functor = IndexFunctor<T, int64_t, int64_t>;
    Functor index_functor = Functor(dims);
    SelectKernel<T, int64_t, int64_t, Functor, VecSize,
                 0><<<grid, block, 0, stream>>>(out_data, cond_data, tmp_in,
                                                cumsum_data, index_functor,
                                                numel, main_offset, rank);
    std::cout << "asdf 3.2 SelectKernel after " << std::endl;
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
