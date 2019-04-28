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

#include <thrust/device_vector.h>
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/where_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

using CUDADeviceContext = paddle::platform::CUDADeviceContext;

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
    thrust::device_vector<int> d_true_index = h_true_index;
    int* ptr_true_index = thrust::raw_pointer_cast(d_true_index.data());

    size_t true_num = h_true_index.size();

    out->Resize(framework::make_ddim({static_cast<int64_t>(true_num), rank}));
    auto out_ptr = out->mutable_data<T>(context.GetPlace());

    if (true_num == 0) {
      return;
    }

    thrust::host_vector<int> h_stride(rank, 0);
    h_stride[rank - 1] = 1;
    for (int i = rank - 2; i >= 0; i--) {
      h_stride[i] = h_stride[i + 1] * dims[i + 1];
    }
    thrust::device_vector<int> d_stride = h_stride;
    int* ptr_stride = thrust::raw_pointer_cast(d_stride.data());

    auto& dev_ctx = context.template device_context<CUDADeviceContext>();
    WhereFunctor<int*> functor(ptr_true_index, true_num, ptr_stride, rank,
                               out_ptr);
    platform::ForRange<CUDADeviceContext> for_range(dev_ctx, true_num);
    for_range(functor);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(where, ops::CUDAWhereKernel<int64_t>);
