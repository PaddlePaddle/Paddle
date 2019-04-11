/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/where_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

__global__ void WhereKernel(thrust::device_ptr<int> true_index, size_t true_num,
                            const int* stride, int rank, int64_t* out) {
  CUDA_1D_KERNEL_LOOP(i, true_num) {
    int64_t index = true_index[i];
    for (int j = 0; j < rank; j++) {
      out[i * rank + j] = index / stride[j];
      index -= out[i * rank + j] * stride[j];
    }
  }
}

template <typename T>
class CUDAWhereKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* condition = context.Input<framework::Tensor>("Condition");
    auto* out = context.Output<framework::Tensor>("Out");

    framework::Tensor n;
    framework::TensorCopy(*condition, platform::CPUPlace(), &n);

    const bool* cond_data = n.data<bool>();
    int64_t numel = n.numel();
    auto dims = n.dims();
    int rank = dims.size();

    thrust::host_vector<int> h_true_index;
    for (int64_t i = 0; i < numel; i++) {
      if (cond_data[i]) {
        h_true_index.push_back(i);
      }
    }

    size_t true_num = h_true_index.size();

    thrust::device_vector<int> d_true_index = h_true_index;

    if (true_num == 0) {
      out->Resize(framework::make_ddim({0L, rank}));
      return;
    }

    out->Resize(framework::make_ddim({static_cast<int64_t>(true_num), rank}));
    auto* out_data = out->mutable_data<int64_t>(context.GetPlace());

    int* stride;
    cudaMallocManaged(&stride, rank * sizeof(int));
    stride[rank - 1] = 1;
    for (int i = rank - 2; i >= 0; i--) {
      stride[i] = stride[i + 1] * dims[i + 1];
    }

    auto stream = context.cuda_device_context().stream();
    int block = 512;
    int grid = (true_num + block - 1) / block;
    WhereKernel<<<grid, block, 0, stream>>>(d_true_index.data(), true_num,
                                            stride, rank, out_data);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(where, ops::CUDAWhereKernel<int64_t>);
