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
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/where_index_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

using CUDADeviceContext = paddle::platform::CUDADeviceContext;

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
    memory::Copy(BOOST_GET_CONST(platform::CUDAPlace, dev_ctx.GetPlace()),
                 d_stride_array, platform::CPUPlace(), h_stride_array,
                 rank * sizeof(int64_t), dev_ctx.stream());

    // get total ture number and set output size
    // the last element of cub::InclusiveSum is the total number
    memory::Copy(platform::CPUPlace(), h_total_true_num,
                 BOOST_GET_CONST(platform::CUDAPlace, dev_ctx.GetPlace()),
                 d_true_num_array + numel - 1, sizeof(int64_t),
                 dev_ctx.stream());
    dev_ctx.Wait();

    int64_t true_num = *h_total_true_num;
    out->Resize(framework::make_ddim({static_cast<int64_t>(true_num), rank}));
    auto out_data = out->mutable_data<int64_t>(context.GetPlace());

    if (true_num == 0) {
      return;
    }

    // using true_num_array and stride_array to calculate the output index
    SetTrueIndex<T><<<grids, threads, 0, dev_ctx.stream()>>>(
        out_data, cond_data, numel, d_stride_array, rank, d_true_num_array);
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
