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
      with_switch_(with_switch),
      switch_endpoint_(switch_endpoint) {
#if defined(PADDLE_WITH_NCCL)
  inner_pg_ = std::make_shared<ProcessGroupNCCL>(store, local_rank, local_size,
                                                 IGNORE_ID);
#elif defined(PADDLE_WITH_ASCEND_CL)
  inner_pg_ = std::make_shared<ProcessGroupHCCL>(store, local_rank, local_size,
                                                 IGNORE_ID);
#else
  PADDLE_THROW(platform::errors::Fatal(
      "ProcessGroupHeter only supports NCCL and HCCL now.");
#endif
  if (local_rank_ == 0 && !with_switch_) {
    auto opts = ProcessGroupGloo::GlooOptions::create();
    opts->device = ProcessGroupGloo::createDefaultDevice();
    inter_pg_ = std::make_shared<ProcessGroupGloo>(store, gloo_rank_,
                                                   gloo_size_, IGNORE_ID, opts);
  }
}

template <typename T>
static void _do_add(T* dst, T* src, size_t size) {
  for (size_t i = 0; i < size; i++) {
    *dst += *src;
    dst++;
    src++;
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
    std::vector<Tensor> cpu_tensors;
    cpu_tensors.reserve(tensors.size());
    for (size_t i = 0; i < tensors.size(); i++) {
      auto dense_gpu_tensor =
          std::dynamic_pointer_cast<phi::DenseTensor>(tensors[i].impl());
      phi::DenseTensorMeta meta = phi::DenseTensorMeta(
          dense_gpu_tensor->dtype(), dense_gpu_tensor->dims());
      std::shared_ptr<phi::DenseTensor> dense_cpu_tensor =
          std::make_shared<phi::DenseTensor>(
              std::make_unique<paddle::experimental::DefaultAllocator>(
                  paddle::platform::CPUPlace())
                  .get(),
              meta);
      dense_cpu_tensor->ResizeAndAllocate(dense_gpu_tensor->dims());
      cpu_tensors[i] = paddle::experimental::Tensor(dense_cpu_tensor);
      framework::TensorCopySync(*dense_gpu_tensor, platform::CPUPlace(),
                                dense_cpu_tensor.get());
    }
    // Step3: do inter cluster allreduce
    if (with_switch_) {
      if (local_rank_ == 0) {
        HeterClient* client_ =
            HeterClient::GetInstance({switch_endpoint_}, {}, 0).get();
        auto dense_cpu_tensor =
            std::dynamic_pointer_cast<phi::DenseTensor>(cpu_tensors[0].impl());
        std::vector<int> send_size;
        send_size.push_back(dense_cpu_tensor->numel());
        int ret = client_->Send(
            gid_, {dense_cpu_tensor->name()}, send_size,
            dense_cpu_tensor->data(),
            dense_cpu_tensor->numel() *
                framework::DataTypeSize(dense_cpu_tensor->dtype()));
        PADDLE_ENFORCE_EQ(ret, 0, platform::errors::PreconditionNotMet(
                                      "Send to the switch module error."));
        phi::DenseTensorMeta meta = phi::DenseTensorMeta(
            dense_cpu_tensor->dtype(), dense_cpu_tensor->dims());
        std::shared_ptr<phi::DenseTensor> dense_cpu_tensor2 =
            std::make_shared<phi::DenseTensor>(
                std::make_unique<paddle::experimental::DefaultAllocator>(
                    paddle::platform::CPUPlace())
                    .get(),
                meta);
        dense_cpu_tensor2->ResizeAndAllocate(dense_cpu_tensor->dims());
        Tensor cpu_tensor_temp =
            paddle::experimental::Tensor(dense_cpu_tensor2);
        ret = client_->Recv(
            gid_, {dense_cpu_tensor->name()}, dense_cpu_tensor2->data(),
            dense_cpu_tensor2->numel() *
                framework::DataTypeSize(dense_cpu_tensor2->dtype()));
        PADDLE_ENFORCE_EQ(ret, 0, platform::errors::PreconditionNotMet(
                                      "Recv from the switch module error."));

        switch (dense_cpu_tensor->dtype()) {
          case DataType::FLOAT32:
            _do_add<float>(reinterpret_cast<float*>(dense_cpu_tensor->data()),
                           reinterpret_cast<float*>(dense_cpu_tensor2->data()),
                           dense_cpu_tensor->numel());
            break;
          case DataType::FLOAT64:
            _do_add<double>(
                reinterpret_cast<double*>(dense_cpu_tensor->data()),
                reinterpret_cast<double*>(dense_cpu_tensor2->data()),
                dense_cpu_tensor->numel());
            break;
          case DataType::INT32:
            _do_add<int>(reinterpret_cast<int*>(dense_cpu_tensor->data()),
                         reinterpret_cast<int*>(dense_cpu_tensor2->data()),
                         dense_cpu_tensor->numel());
            break;
          default:
            PADDLE_THROW(platform::errors::PreconditionNotMet(
                "Unsupported data type (%s) to do add.",
                framework::DataType2String(dense_cpu_tensor->dtype())));
        }
      }
    } else {
      auto gloo_task = inter_pg_->AllReduce(cpu_tensors, opts);
      gloo_task->Wait();
    }
    // Step4: copy cpu tensors to gpu
    // copy cpu tensors to gpu
    for (size_t i = 0; i < tensors.size(); i++) {
      auto dense_gpu_tensor =
          std::dynamic_pointer_cast<phi::DenseTensor>(tensors[i].impl());
      auto dense_cpu_tensor =
          std::dynamic_pointer_cast<phi::DenseTensor>(cpu_tensors[i].impl());
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
    std::vector<Tensor> cpu_tensors;
    cpu_tensors.reserve(tensors.size());
    for (size_t i = 0; i < tensors.size(); i++) {
      auto dense_gpu_tensor =
          std::dynamic_pointer_cast<phi::DenseTensor>(tensors[i].impl());
      phi::DenseTensorMeta meta = phi::DenseTensorMeta(
          dense_gpu_tensor->dtype(), dense_gpu_tensor->dims());
      std::shared_ptr<phi::DenseTensor> dense_cpu_tensor =
          std::make_shared<phi::DenseTensor>(
              std::make_unique<paddle::experimental::DefaultAllocator>(
                  paddle::platform::CPUPlace())
                  .get(),
              meta);
      dense_cpu_tensor->ResizeAndAllocate(dense_gpu_tensor->dims());
      cpu_tensors[i] = paddle::experimental::Tensor(dense_cpu_tensor);
      framework::TensorCopySync(*dense_gpu_tensor, platform::CPUPlace(),
                                dense_cpu_tensor.get());
    }
    if (with_switch_) {
      if (local_rank_ == 0) {
        HeterClient* client_ =
            HeterClient::GetInstance({switch_endpoint_}, {}, 0).get();
        auto dense_cpu_tensor =
            std::dynamic_pointer_cast<phi::DenseTensor>(cpu_tensors[0].impl());
        if (gloo_rank_ == 0) {
          std::vector<int> send_size;
          send_size.push_back(dense_cpu_tensor->numel());
          int ret = client_->Send(
              gid_, {dense_cpu_tensor->name()}, send_size,
              dense_cpu_tensor->data(),
              dense_cpu_tensor->numel() *
                  framework::DataTypeSize(dense_cpu_tensor->dtype()));
          PADDLE_ENFORCE_EQ(ret, 0, platform::errors::PreconditionNotMet(
                                        "Send to the switch module error."));
        } else {
          int ret = client_->Recv(
              gid_, {dense_cpu_tensor->name()}, dense_cpu_tensor->data(),
              dense_cpu_tensor->numel() *
                  framework::DataTypeSize(dense_cpu_tensor->dtype()));
          PADDLE_ENFORCE_EQ(ret, 0,
                            platform::errors::PreconditionNotMet(
                                "Receive from the switch module error."));
          ret = client_->Recv(
              gid_, {dense_cpu_tensor->name()}, dense_cpu_tensor->data(),
              dense_cpu_tensor->numel() *
                  framework::DataTypeSize(dense_cpu_tensor->dtype()));
          PADDLE_ENFORCE_EQ(ret, 0,
                            platform::errors::PreconditionNotMet(
                                "Receive from the switch module error."));
        }
      }
    } else {
      auto gloo_task = inter_pg_->Broadcast(cpu_tensors, opts);
      gloo_task->Wait();
    }
    for (size_t i = 0; i < tensors.size(); i++) {
      auto dense_gpu_tensor =
          std::dynamic_pointer_cast<phi::DenseTensor>(tensors[i].impl());
      auto dense_cpu_tensor =
          std::dynamic_pointer_cast<phi::DenseTensor>(cpu_tensors[i].impl());
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
    phi::DenseTensorMeta meta = phi::DenseTensorMeta(in->dtype(), in->dims());
    std::shared_ptr<phi::DenseTensor> dense_cpu_tensor =
        std::make_shared<phi::DenseTensor>(
            std::make_unique<paddle::experimental::DefaultAllocator>(
                paddle::platform::CPUPlace())
                .get(),
            meta);
    dense_cpu_tensor->ResizeAndAllocate(in->dims());
    Tensor cpu_tensor = paddle::experimental::Tensor(dense_cpu_tensor);
    framework::TensorCopySync(*in, platform::CPUPlace(),
                              dense_cpu_tensor.get());
    if (with_switch_) {
      if (local_rank_ == 0) {
        HeterClient* client_ =
            HeterClient::GetInstance({switch_endpoint_}, {}, 0).get();
        if (gloo_rank_ == 0) {
          std::vector<int> send_size;
          send_size.push_back(in->numel());
          int ret = client_->Send(
              gid_, {in->name()}, send_size, dense_cpu_tensor->data(),
              in->numel() * framework::DataTypeSize(in->dtype()));
          PADDLE_ENFORCE_EQ(ret, 0, platform::errors::PreconditionNotMet(
                                        "Send to the switch module error."));
        } else {
          int ret =
              client_->Recv(gid_, {in->name()}, dense_cpu_tensor->data(),
                            in->numel() * framework::DataTypeSize(in->dtype()));
          PADDLE_ENFORCE_EQ(ret, 0,
                            platform::errors::PreconditionNotMet(
                                "Receive from the switch module error."));
        }
      }
    } else {
      std::vector<Tensor> cpu_tensors = {cpu_tensor};
      auto gloo_task = inter_pg_->Broadcast(cpu_tensors);
      gloo_task->Wait();
    }
    framework::TensorCopySync(*dense_cpu_tensor, out->place(), out);
  }
  inner_pg_->Broadcast(out, out);
}

}  //  namespace distributed
}  //  namespace paddle
