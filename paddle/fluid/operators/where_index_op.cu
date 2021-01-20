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
template <typename T>
__global__ void KeSetTrueIndex(int64_t *out_ptr, const T *cond_data,
                               const int64_t numel, const int64_t *ptr_stride,
                               const int64_t rank,
                               const int64_t *true_num_array) {
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int64_t idx = tid; idx < numel; idx += gridDim.x * blockDim.x) {
    if (CheckTrue<T>()(cond_data[idx])) {
      int64_t index = idx;
      int64_t true_index = true_num_array[idx] - 1;
      for (int j = 0; j < rank; j++) {
        int64_t out_index = index / ptr_stride[j];
        out_ptr[true_index * rank + j] = out_index;
        index -= out_index * ptr_stride[j];
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
    int64_t numel = condition->numel();
    auto dims = condition->dims();
    int rank = dims.size();

    int64_t threads =
        std::min(static_cast<int64_t>(dev_ctx.GetMaxThreadsPerBlock()), numel);
    int64_t need_grids = (numel + threads - 1) / threads;
    int64_t grids = std::min(
        static_cast<int64_t>(dev_ctx.GetCUDAMaxGridDimSize().x), need_grids);

    int64_t tmp_mem_size = rank + numel;
    auto d_tmp_mem = memory::Alloc(dev_ctx, tmp_mem_size * sizeof(int64_t));
    auto h_tmp_mem =
        memory::Alloc(platform::CPUPlace(), rank * sizeof(int64_t));

    int64_t *ptr_stride = reinterpret_cast<int64_t *>(d_tmp_mem->ptr());
    int64_t *ptr_true_num_array = ptr_stride + rank;

    int64_t *h_stride = reinterpret_cast<int64_t *>(h_tmp_mem->ptr());

    out->Resize(framework::make_ddim({static_cast<int64_t>(numel), rank}));
    auto out_ptr = out->mutable_data<int64_t>(context.GetPlace());

    h_stride[rank - 1] = 1;
    for (int i = rank - 2; i >= 0; i--) {
      h_stride[i] = h_stride[i + 1] * dims[i + 1];
    }
    memory::Copy(BOOST_GET_CONST(platform::CUDAPlace, dev_ctx.GetPlace()),
                 ptr_stride, platform::CPUPlace(), h_stride,
                 rank * sizeof(int64_t), dev_ctx.stream());

    KeGetTrueNum<T><<<grids, threads, 0, dev_ctx.stream()>>>(
        cond_data, numel, ptr_true_num_array);
    // TODO(jiangcheng): write prefix-sum kernel instead of
    // thrust::inclusive_scan
    thrust::inclusive_scan(thrust::device, ptr_true_num_array,
                           ptr_true_num_array + numel, ptr_true_num_array);
    KeSetTrueIndex<T><<<grids, threads, 0, dev_ctx.stream()>>>(
        out_ptr, cond_data, numel, ptr_stride, rank, ptr_true_num_array);

    int64_t true_num = 0;
    memory::Copy(platform::CPUPlace(), &true_num,
                 BOOST_GET_CONST(platform::CUDAPlace, dev_ctx.GetPlace()),
                 ptr_true_num_array + numel - 1, sizeof(int64_t),
                 dev_ctx.stream());
    dev_ctx.Wait();

    out->Resize(framework::make_ddim({static_cast<int64_t>(true_num), rank}));
    if (true_num == 0) {
      return;
    }
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
