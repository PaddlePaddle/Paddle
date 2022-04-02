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

#include "paddle/fluid/distributed/collective/ProcessGroupHeter.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/common/place.h"

constexpr int64_t kWaitBlockTImeout = 10;

namespace paddle {
namespace distributed {

using Place = paddle::platform::Place;

std::shared_ptr<ProcessGroupHeter::HeterTask> ProcessGroupHeter::CreateTask(
    int rank, CommType comm_type, const std::vector<Tensor>& inputs) {
  return std::make_shared<ProcessGroupHeter::HeterTask>(rank, comm_type,
                                                        inputs);
}

ProcessGroupHeter::HeterTask::HeterTask(int rank, CommType CommType,
                                        const std::vector<Tensor>& inputs)
    : Task(rank, inputs, CommType) {}

ProcessGroupHeter::HeterTask::~HeterTask() {}

bool ProcessGroupHeter::HeterTask::IsCompleted() { return true; }

// TODO(sheniang03): Add timeout for wait, now timeout unused
bool ProcessGroupHeter::HeterTask::Wait(std::chrono::milliseconds timeout) {
  return true;
}

ProcessGroupHeter::ProcessGroupHeter(const std::shared_ptr<Store>& store,
                                     int rank, int size, int gid,
                                     int local_rank, int local_size,
                                     int gloo_rank, int gloo_size,
                                     bool with_switch,
                                     std::string switch_endpoint)
    : ProcessGroup(rank, size, gid),
      store_(store),
      local_rank_(local_rank),
      local_size_(local_size),
      gloo_rank_(gloo_rank),
      gloo_size_(gloo_size),
      with_switch_(with_switch) {
#if defined(PADDLE_WITH_NCCL)
  inner_pg_ = std::make_shared<ProcessGroupNCCL>(store, local_rank, local_size,
                                                 IGNORE_ID);
#elif defined(PADDLE_WITH_ASCEND_CL)
  inner_pg_ = std::make_shared<ProcessGroupHCCL>(store, local_rank, local_size,
                                                 IGNORE_ID);
#else
  PADDLE_THROW(platform::errors::InvalidArgument(
      "ProcessGroupHeter only supports NCCL and HCCL now.");
#endif
  if (with_switch_) {
    // TODO(sandyhouse) starts a client to connect the cloud switch module
    // std::shared_ptr<HeterClient> client_ =
    // HeterClient::GetInstance({switch_endpoint}, {}, 0);
  } else if (local_rank_ == 0) {
    auto opts = ProcessGroupGloo::GlooOptions::create();
    opts->device = ProcessGroupGloo::createDefaultDevice();
    inter_pg_ = std::make_shared<ProcessGroupGloo>(store, gloo_rank_,
                                                   gloo_size_, IGNORE_ID, opts);
  }
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupHeter::AllReduce(
    std::vector<Tensor>& tensors, const AllreduceOptions& opts) {
#if defined(PADDLE_WITH_NCCL)
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCudaPlace(tensors), true,
      platform::errors::InvalidArgument("All inputs should be in CudaPlace."));
#endif

  // Step1: do allreduce in inner cluster
  auto task = inner_pg_->AllReduce(tensors, opts);
  task->Wait();

  // Step2: copy tensors to CPU
  if (local_rank_ == 0) {
    std::vector<Tensor> cpu_tensors(tensors.size());
    for (size_t i = 0; i < tensors.size(); i++) {
      auto dense_gpu_tensor =
          std::dynamic_pointer_cast<phi::DenseTensor>(tensors[i].impl());
      auto dense_cpu_tensor =
          std::dynamic_pointer_cast<phi::DenseTensor>(cpu_tensors[i].impl());
      dense_cpu_tensor->Resize(tensors[i].dims());
      framework::TensorCopySync(*dense_gpu_tensor, platform::CPUPlace(),
                                dense_cpu_tensor.get());
    }
    // Step3: do inter cluster allreduce
    if (with_switch_) {
      // TODO(sandyhouse) send to and recv from switch, and do add
    } else {
      auto gloo_task = inter_pg_->AllReduce(cpu_tensors, opts);
      gloo_task->Wait();
    }
    // Step4: copy cpu tensors to gpu
    // TODO(sandyhouse)
    // copy cpu tensors to gpu
    for (size_t i = 0; i < tensors.size(); i++) {
      auto dense_gpu_tensor =
          std::dynamic_pointer_cast<phi::DenseTensor>(tensors[i].impl());
      auto dense_cpu_tensor =
          std::dynamic_pointer_cast<phi::DenseTensor>(cpu_tensors[i].impl());
      // framework::TensorCopySync(*dense_cpu_tensor, tensors[i].place(),
      // dense_gpu_tensor.get());
      framework::TensorCopySync(*dense_cpu_tensor, dense_cpu_tensor->place(),
                                dense_gpu_tensor.get());
    }
  }

  // Step5: broadcast among inner cluster
  auto b_opts = BroadcastOptions();
  b_opts.source_root = 0;
  auto broadcast_task = inner_pg_->Broadcast(tensors, b_opts);
  broadcast_task->Wait();
  return CreateTask(rank_, CommType::ALLREDUCE, tensors);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupHeter::Broadcast(
    std::vector<Tensor>& tensors, const BroadcastOptions& opts) {
#if defined(PADDLE_WITH_NCCL)
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCudaPlace(tensors), true,
      platform::errors::InvalidArgument("All inputs should be in CudaPlace."));
#endif

  // Step1: do broadcast in inner cluster
  auto b_opts = BroadcastOptions();
  b_opts.source_root = 0;
  inner_pg_->Broadcast(tensors, b_opts);

  if (local_rank_ == 0) {
    std::vector<Tensor> cpu_tensors(tensors.size());
    for (size_t i = 0; i < tensors.size(); i++) {
      auto dense_gpu_tensor =
          std::dynamic_pointer_cast<phi::DenseTensor>(tensors[i].impl());
      auto dense_cpu_tensor =
          std::dynamic_pointer_cast<phi::DenseTensor>(cpu_tensors[i].impl());
      dense_cpu_tensor->Resize(tensors[i].dims());
      framework::TensorCopySync(*dense_gpu_tensor, platform::CPUPlace(),
                                dense_cpu_tensor.get());
    }
    if (with_switch_) {
      // TODO(sandyhouse) send to and recv
    } else {
      auto gloo_task = inter_pg_->Broadcast(cpu_tensors, opts);
      gloo_task->Wait();
    }
    for (size_t i = 0; i < tensors.size(); i++) {
      auto dense_gpu_tensor =
          std::dynamic_pointer_cast<phi::DenseTensor>(tensors[i].impl());
      auto dense_cpu_tensor =
          std::dynamic_pointer_cast<phi::DenseTensor>(cpu_tensors[i].impl());
      // framework::TensorCopySync(*dense_cpu_tensor, tensors[i].place(),
      // dense_gpu_tensor.get());
      framework::TensorCopySync(*dense_cpu_tensor, dense_cpu_tensor->place(),
                                dense_gpu_tensor.get());
    }
  }
  auto broadcast_task = inner_pg_->Broadcast(tensors, b_opts);
  broadcast_task->Wait();
  return CreateTask(rank_, CommType::BROADCAST, tensors);
}

void ProcessGroupHeter::Broadcast(const phi::DenseTensor* in,
                                  phi::DenseTensor* out) {
  // Step1: do broadcast in inner cluster
  inner_pg_->Broadcast(in, out);

  if (local_rank_ == 0) {
    Tensor cpu_tensor;
    auto dense_cpu_tensor =
        std::dynamic_pointer_cast<phi::DenseTensor>(cpu_tensor.impl());
    dense_cpu_tensor->Resize(in->dims());
    framework::TensorCopySync(*in, platform::CPUPlace(),
                              dense_cpu_tensor.get());
    if (with_switch_) {
      // TODO(sandyhouse) send to and recv
    } else {
      std::vector<Tensor> cpu_tensors = {cpu_tensor};
      // auto gloo_task = inter_pg_->Broadcast(cpu_tensors);
      // gloo_task->Wait();
      inter_pg_->Broadcast(cpu_tensors);
    }
    framework::TensorCopySync(*dense_cpu_tensor, dense_cpu_tensor->place(),
                              out);
  }
  inner_pg_->Broadcast(out, out);
}

}  //  namespace distributed
}  //  namespace paddle
