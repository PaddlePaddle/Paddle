// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/expert_count_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

#define CEIL(_x_, _y_) (((_x_)-1) / (_y_) + 1)
#define PERTHREAD_EXPERTS 256
#define WARP_SIZE 32

using LoDTensor = framework::LoDTensor;
using Tensor = framework::Tensor;

template <typename T>
__global__ void expert_count_impl(const T* gate_idx, int* expert_count,
                                  int64_t batch_size, int n_expert) {
  int res_tmp[PERTHREAD_EXPERTS] = {0};
  int64_t expert_min = blockIdx.x * PERTHREAD_EXPERTS;
  int64_t expert_max = expert_min + PERTHREAD_EXPERTS;
  if (expert_max > n_expert) {
    expert_max = n_expert;
  }
  for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
    int64_t idx = gate_idx[i];
    if (idx == -1) {
      continue;
    }
    if (idx < expert_min || idx >= expert_max) {
      continue;
    }
    res_tmp[idx - expert_min] += 1;
  }
  for (int i = expert_min; i < expert_max; ++i) {
    int x = res_tmp[i - expert_min];
#pragma unroll
    for (int j = 1; j < WARP_SIZE; j <<= 1) {
      x = x + __shfl_down_sync(-1u, x, j);
    }
    if (threadIdx.x % WARP_SIZE == 0) {
      platform::CudaAtomicAdd(expert_count + i, x);
    }
  }
}

template <typename T>
class ExpertCountOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto gate_idx = context.Input<Tensor>("gate_idx");
    auto n_expert = context.Attr<int>("n_expert");
    auto expert_count = context.Output<Tensor>("Out");

    auto batch_size = gate_idx->numel();
    auto place = context.GetPlace();
    const auto& dev_ctx =
        context.template device_context<platform::CUDADeviceContext>();

    framework::DDim out_dims = framework::make_ddim({n_expert});
    auto out_data = expert_count->mutable_data<int>(out_dims, place);
    const T* gate_data = gate_idx->data<T>();

    expert_count_impl<
        T><<<CEIL(n_expert, PERTHREAD_EXPERTS), 256, 0, dev_ctx.stream()>>>(
        gate_data, out_data, batch_size, n_expert);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(expert_count, ops::ExpertCountOpCUDAKernel<int64_t>);
