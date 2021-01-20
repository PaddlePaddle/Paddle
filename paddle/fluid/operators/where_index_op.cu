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

#include <algorithm>
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/malloc.h"
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
template <typename T, int BLOCKDIM>
__global__ void KeGetTrueIndex(int64_t *out_ptr, const T *cond_data,
                               const int64_t numel, const int64_t *ptr_stride,
                               const int64_t rank, int64_t *true_num) {
  const int tid = threadIdx.x;
  __shared__ int64_t s_num[BLOCKDIM + 1];
  s_num[tid + 1] = 0;
  const int64_t cond_block = (numel + BLOCKDIM - 1) / BLOCKDIM;
  const int64_t t_beg = tid * cond_block;
  const int64_t t_over = min(numel, t_beg + cond_block);

  if (tid < BLOCKDIM && t_beg < numel) {
    // first: each thread counting true condition number
    int64_t t_num = 0;
    for (int64_t i = t_beg; i < t_over; i++) {
      if (CheckTrue<T>()(cond_data[i])) t_num++;
    }
    s_num[tid + 1] = t_num;
    __syncthreads();

    // second: thread 0 counting all true condition number
    if (tid == 0) {
      s_num[0] = 0;
      for (int i = 1; i <= BLOCKDIM; i++) {
        s_num[i] += s_num[i - 1];
      }
      *true_num = s_num[BLOCKDIM];
    }
    __syncthreads();

    // third: each thread set true index
    int64_t idx = s_num[tid];
    for (int64_t i = t_beg; i < t_over; i++) {
      if (CheckTrue<T>()(cond_data[i])) {
        int64_t index = i;
        for (int j = 0; j < rank; j++) {
          int64_t stride_val = ptr_stride[j];
          int64_t out_val = index / stride_val;

          out_ptr[idx * rank + j] = out_val;
          index -= out_val * stride_val;
        }
        idx++;
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

    int64_t tmp_mem_size = rank + 1;
    auto tmp_mem = memory::Alloc(dev_ctx, tmp_mem_size * sizeof(int64_t));
    auto h_tmp_mem =
        memory::Alloc(platform::CPUPlace(), rank * sizeof(int64_t));

    int64_t *ptr_stride = reinterpret_cast<int64_t *>(tmp_mem->ptr());
    int64_t *true_num_d = ptr_stride + rank;
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

// TODO(jiangcheng): A terrible CUDA kernel, need optimize
#define SetKenelParam(num)                                 \
  KeGetTrueIndex<T, num><<<1, num, 0, dev_ctx.stream()>>>( \
      out_ptr, cond_data, numel, ptr_stride, rank, true_num_d);

    if (numel > 1024) {
      SetKenelParam(1024)
    } else if (numel > 512) {
      SetKenelParam(512)
    } else if (numel > 256) {
      SetKenelParam(256)
    } else if (numel > 128) {
      SetKenelParam(128)
    } else if (numel > 64) {
      SetKenelParam(64)
    } else if (numel > 32) {
      SetKenelParam(32)
    } else {
      SetKenelParam(1)
    }

    int64_t true_num = 0;
    memory::Copy(platform::CPUPlace(), &true_num,
                 BOOST_GET_CONST(platform::CUDAPlace, dev_ctx.GetPlace()),
                 true_num_d, sizeof(int64_t), dev_ctx.stream());
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
