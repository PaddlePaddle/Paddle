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

#include "paddle/fluid/operators/prune_gate_by_capacity_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

namespace paddle {
namespace operators {
using LoDTensor = framework::LoDTensor;
using Tensor = framework::Tensor;

static constexpr int kNumCUDAThreads = 512;
static constexpr int kNumMaxinumNumBlocks = 4096;

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaxinumNumBlocks);
}

template <typename T>
__global__ void PruneGateByCapacity(const T* gate_idx_data,
                                    T* new_gate_idx_data,
                                    int* expert_count_data,
                                    const int64_t batch_size) {
  CUDA_KERNEL_LOOP(i, batch_size) {
    auto orig_cap =
        platform::CudaAtomicAdd(expert_count_data + gate_idx_data[i], -1);
    if (orig_cap <= 0) {
      new_gate_idx_data[i] = -1;
    } else {
      new_gate_idx_data[i] = gate_idx_data[i];
    }
  }
}

template <typename DeviceContext, typename T>
class PruneGateByCapacityCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* gate_idx = context.Input<Tensor>("GateIdx");
    auto* expert_count_out = context.Output<Tensor>("ExpertCountOut");
    auto* new_gate_idx = context.Output<Tensor>("NewGateIdx");

    auto* expert_count_out_data =
        expert_count_out->mutable_data<int>(context.GetPlace());

    auto batch_size = gate_idx->numel();
    auto* gate_idx_data = gate_idx->data<T>();

    for (size_t i = 0; i < batch_size; i++) {
      printf("%d", expert_count_out_data[i]);
    }

    auto* new_gate_idx_data = new_gate_idx->mutable_data<T>(context.GetPlace());
    auto& dev_ctx = context.template device_context<DeviceContext>();

    int blocks = NumBlocks(batch_size);
    int threads = kNumCUDAThreads;

    PruneGateByCapacity<T><<<blocks, threads, 0, dev_ctx.stream()>>>(
        gate_idx_data, new_gate_idx_data, expert_count_out_data, batch_size);
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_CUDA_SUCCESS(hipStreamSynchronize(dev_ctx.stream()));
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamSynchronize(dev_ctx.stream()));
#endif
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(
    prune_gate_by_capacity,
    ops::PruneGateByCapacityCUDAKernel<plat::CUDADeviceContext, int>,
    ops::PruneGateByCapacityCUDAKernel<plat::CUDADeviceContext, int64_t>);
