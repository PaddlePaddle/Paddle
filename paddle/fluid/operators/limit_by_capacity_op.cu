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
//
// The file has been adapted from the two files:
//     https://github.com/laekov/fastmoe/blob/master/cuda/balancing.cu
//     https://github.com/laekov/fastmoe/blob/master/cuda/balancing.cuh
//     Git commit hash: 295a615aacce7e54a37e7935274ba15e901c78e4
// We retain the following license from the original files:
//      Copyright 2021, Jiaao He. All rights reserved.
//  Licensed under the Apache License, Version 2.0 (the "License").

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/limit_by_capacity_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;
using Tensor = framework::Tensor;

template <typename T>
__global__ void limit_by_capacity_impl(const T* expc, T* cap, T* out,
                                       const int n_expert, const int n_worker) {
  int eid, wid;
  CUDA_KERNEL_LOOP(i, (n_expert * n_worker)) {
    wid = i / n_expert;
    eid = i % n_expert;
    auto proposal = expc[wid * n_expert + eid];
    auto cap_left = paddle::platform::CudaAtomicAdd(cap + eid, proposal * (-1));
    if (cap_left >= proposal) {
      out[wid * n_expert + eid] = proposal;
    } else if (cap_left >= 0) {
      out[wid * n_expert + eid] = cap_left;
    } else {
      out[wid * n_expert + eid] = 0;
    }
  }
}

template <typename T>
class LimitByCapacityOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto expert_count = context.Input<Tensor>("expert_count");
    auto capacity = context.Input<Tensor>("capacity");
    auto n_worker = context.Attr<int>("n_worker");
    auto out = context.Output<Tensor>("Out");

    auto n_expert = expert_count->numel() / n_worker;
    const auto place = context.GetPlace();
    const auto& dev_ctx =
        context.template device_context<platform::CUDADeviceContext>();

    dim3 grid_dim(256);
    dim3 block_dim(1024);
    auto out_data = out->mutable_data<T>(place);
    const T* ec_data = expert_count->data<T>();

    framework::Tensor capacity_copy;
    framework::TensorCopy(*capacity, place, dev_ctx, &capacity_copy);
    T* cap_data = capacity_copy.mutable_data<T>(place);

    limit_by_capacity_impl<T><<<grid_dim, block_dim, 0, dev_ctx.stream()>>>(
        ec_data, cap_data, out_data, n_expert, n_worker);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(limit_by_capacity,
                        ops::LimitByCapacityOpCUDAKernel<int64_t>);
