/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/assign_pos_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

static constexpr int kNumCUDAThreads = 512;
static constexpr int kNumMaxinumNumBlocks = 4096;

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaxinumNumBlocks);
}

template <typename T>
__global__ void AssignPos(T* cum_count, const int* gate, int64_t* out,
                          int64_t limit) {
  CUDA_KERNEL_LOOP(i, limit) {
    int gate_idx = gate[i];
    if (gate_idx > -1) {
      int p = platform::CudaAtomicAdd(cum_count + gate_idx, -1);
      out[p - 1] = i;
    }
  }
}

template <typename T>
class AssignPosCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    // assign pos decides which tokens should be fetched belong to specially
    // expert orderingly.
    auto cum_count = context.Input<LoDTensor>(
        "cum_count");  // (num_expert * world_size) int32 | int64
    auto gate =
        context.Input<LoDTensor>("X");  // (batch_size * seq_len, topk) int32
    auto eff_gates_len =
        context.Input<LoDTensor>("eff_gates_len");  // (sum(cum_count))
    auto out = context.Output<LoDTensor>("Out");    // (cum_count) value ranges
                                                    // from 0 to batch_size *
                                                    // seq_len * topk
    auto place = context.GetPlace();
    auto numel = gate->numel();
    T* cum_data = const_cast<T*>(cum_count->data<T>());
    auto cum_size = cum_count->numel();

    framework::Tensor cpu_eff_gates_len;
    int64_t cpu_eff_gates_len_data = 0;
    if (platform::is_cpu_place(eff_gates_len->place())) {
      cpu_eff_gates_len_data = eff_gates_len->data<int64_t>()[0];
    } else {
      framework::TensorCopySync(*eff_gates_len, platform::CPUPlace(),
                                &cpu_eff_gates_len);
      cpu_eff_gates_len_data = cpu_eff_gates_len.data<int64_t>()[0];
    }
    const auto& dev_ctx =
        context.template device_context<platform::CUDADeviceContext>();

    framework::DDim out_dims = framework::make_ddim({cpu_eff_gates_len_data});
    auto out_data = out->mutable_data<int64_t>(out_dims, place);

    const int* gate_data = gate->data<int>();

    int blocks = NumBlocks(numel);
    int threads = kNumCUDAThreads;
    AssignPos<T><<<blocks, threads, 0, dev_ctx.stream()>>>(cum_data, gate_data,
                                                           out_data, numel);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(assign_pos, ops::AssignPosCUDAKernel<int>,
                        ops::AssignPosCUDAKernel<int64_t>);
