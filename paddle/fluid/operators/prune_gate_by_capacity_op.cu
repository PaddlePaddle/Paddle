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
//
// The file has been adapted from the two files:
//     https://github.com/laekov/fastmoe/blob/master/cuda/balancing.cu
//     https://github.com/laekov/fastmoe/blob/master/cuda/balancing.cuh
//     Git commit hash: 295a615aacce7e54a37e7935274ba15e901c78e4
// We retain the following license from the original files:
//     Copyright 2021, Jiaao He. All rights reserved.
//  Licensed under the Apache License, Version 2.0 (the "License").

#include "paddle/fluid/operators/prune_gate_by_capacity_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

DECLARE_bool(avoid_op_randomness);

namespace paddle {
namespace operators {
using LoDTensor = framework::LoDTensor;

static constexpr int kNumCUDAThreads = 512;
static constexpr int kNumMaxinumNumBlocks = 4096;

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaxinumNumBlocks);
}

template <typename T1, typename T2>
__global__ void prune_gate_by_capacity_kernel(const T1* gate_idx_data,
                                              T1* new_gate_idx_data,
                                              T2* expert_count_data,
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

template <typename DeviceContext, typename T1>
class PruneGateByCapacityFunctor {
 public:
  PruneGateByCapacityFunctor(const framework::ExecutionContext& context,
                             const framework::LoDTensor* gate_idx,
                             framework::LoDTensor* expert_count_out,
                             T1* new_gate_idx_data)
      : context_(context),
        gate_idx_(gate_idx),
        expert_count_out_(expert_count_out),
        new_gate_idx_data_(new_gate_idx_data) {}

  template <typename T2>
  void apply() {
    auto batch_size = gate_idx_->numel();
    auto* gate_idx_data = gate_idx_->data<T1>();

    auto& dev_ctx = context_.template device_context<DeviceContext>();
    auto* expert_count_out_data = expert_count_out_->data<T2>();

    int blocks = NumBlocks(batch_size);
    int threads = kNumCUDAThreads;

    prune_gate_by_capacity_kernel<T1,
                                  T2><<<blocks, threads, 0, dev_ctx.stream()>>>(
        gate_idx_data, new_gate_idx_data_, expert_count_out_data, batch_size);
  }

 private:
  const framework::ExecutionContext context_;
  const framework::LoDTensor* gate_idx_;
  framework::LoDTensor* expert_count_out_;
  T1* new_gate_idx_data_;
};

template <typename Visitor>
static void VisitDataType(paddle::experimental::DataType type,
                          Visitor visitor) {
  if (type == paddle::experimental::DataType::INT64) {
    visitor.template apply<int64_t>();
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "The recieved values gate_id type %s can not meet input requirements. "
        "Because the given gate_id data type of operators must be "
        "int64. Please input appropriate gate_id again! ",
        "framework::DataTypeToString(type)"));
  }
}

template <typename DeviceContext, typename T>
class PruneGateByCapacityCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* gate_idx = context.Input<LoDTensor>("GateIdx");
    auto* expert_count = context.Input<LoDTensor>("ExpertCount");
    // auto* expert_count_out = context.Output<LoDTensor>("ExpertCountOut");
    auto* new_gate_idx = context.Output<LoDTensor>("NewGateIdx");
    auto* new_gate_idx_data = new_gate_idx->mutable_data<T>(context.GetPlace());

    framework::LoDTensor expert_count_out;
    framework::TensorCopy(*expert_count, context.GetPlace(), &expert_count_out);
    PruneGateByCapacityFunctor<DeviceContext, T> functor(
        context, gate_idx, &expert_count_out, new_gate_idx_data);
    VisitDataType(expert_count->type(), functor);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(
    prune_gate_by_capacity,
    ops::PruneGateByCapacityCUDAKernel<plat::CUDADeviceContext, int64_t>);
