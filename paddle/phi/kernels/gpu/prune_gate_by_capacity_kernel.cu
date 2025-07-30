// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/kernels/prune_gate_by_capacity_kernel.h"

namespace phi {

static constexpr int kNumCUDAThreads = 512;
static constexpr int kNumMaximumNumBlocks = 4096;

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaximumNumBlocks);
}

template <typename T1, typename T2>
__global__ void prune_gate_by_capacity_kernel(const T1* gate_idx_data,
                                              T1* new_gate_idx_data,
                                              T2* expert_count_data,
                                              const int64_t batch_size) {
  CUDA_KERNEL_LOOP(i, batch_size) {
    auto orig_cap =
        phi::CudaAtomicAdd(expert_count_data + gate_idx_data[i], -1);
    if (orig_cap <= 0) {
      new_gate_idx_data[i] = -1;
    } else {
      new_gate_idx_data[i] = gate_idx_data[i];
    }
  }
}

template <typename Context, typename T1>
class PruneGateByCapacityFunctor {
 public:
  PruneGateByCapacityFunctor(const Context& dev_ctx,
                             const phi::DenseTensor* gate_idx,
                             phi::DenseTensor* expert_count_out,
                             T1* new_gate_idx_data)
      : dev_ctx_(dev_ctx),
        gate_idx_(gate_idx),
        expert_count_out_(expert_count_out),
        new_gate_idx_data_(new_gate_idx_data) {}

  template <typename T2>
  void apply() {
    auto batch_size = gate_idx_->numel();
    auto* gate_idx_data = gate_idx_->data<T1>();

    auto* expert_count_out_data = expert_count_out_->data<T2>();

    int blocks = NumBlocks(batch_size);
    int threads = kNumCUDAThreads;

    prune_gate_by_capacity_kernel<T1, T2>
        <<<blocks, threads, 0, dev_ctx_.stream()>>>(gate_idx_data,
                                                    new_gate_idx_data_,
                                                    expert_count_out_data,
                                                    batch_size);
  }

 private:
  const Context& dev_ctx_;
  const phi::DenseTensor* gate_idx_;
  phi::DenseTensor* expert_count_out_;
  T1* new_gate_idx_data_;
};

template <typename Visitor>
static void VisitType(phi::DataType type, Visitor visitor) {
  if (type == phi::DataType::INT64) {
    visitor.template apply<int64_t>();
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The received values gate_id type %s can not meet input requirements. "
        "Because the given gate_id data type of operators must be "
        "int64. Please input appropriate gate_id again! ",
        "framework::DataTypeToString(type)"));
  }
}

template <typename T, typename Context>
void PruneGateByCapacityKernel(const Context& dev_ctx,
                               const DenseTensor& gate_idx,
                               const DenseTensor& expert_count,
                               int64_t n_expert,
                               int64_t n_worker,
                               DenseTensor* new_gate_idx) {
  auto* gate_idx_ptr = &gate_idx;
  // auto* expert_count_out =
  // context.Output<phi::DenseTensor>("ExpertCountOut");
  auto* new_gate_idx_data = dev_ctx.template Alloc<T>(new_gate_idx);

  phi::DenseTensor expert_count_out;
  phi::Copy(
      dev_ctx, expert_count, dev_ctx.GetPlace(), false, &expert_count_out);
  PruneGateByCapacityFunctor<Context, T> functor(
      dev_ctx, gate_idx_ptr, &expert_count_out, new_gate_idx_data);
  VisitType(expert_count.type(), functor);
}

}  // namespace phi

PD_REGISTER_KERNEL(prune_gate_by_capacity,
                   GPU,
                   ALL_LAYOUT,
                   phi::PruneGateByCapacityKernel,
                   int64_t) {}
