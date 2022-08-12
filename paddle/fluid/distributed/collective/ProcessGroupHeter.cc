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

#include <chrono>

#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/common/place.h"

constexpr int64_t kWaitBlockTImeout = 10;

namespace paddle {
namespace distributed {

using Place = paddle::platform::Place;
int ProcessGroupHeter::send_count = 0;
int ProcessGroupHeter::recv_count = 0;

std::shared_ptr<ProcessGroupHeter::HeterTask> ProcessGroupHeter::CreateTask(
    int rank, CommType comm_type, const std::vector<phi::DenseTensor>& inputs) {
  return std::make_shared<ProcessGroupHeter::HeterTask>(
      rank, comm_type, inputs);
}

ProcessGroupHeter::HeterTask::HeterTask(
    int rank, CommType CommType, const std::vector<phi::DenseTensor>& inputs)
    : Task(rank, inputs, CommType) {}

ProcessGroupHeter::HeterTask::~HeterTask() {}

bool ProcessGroupHeter::HeterTask::IsCompleted() { return true; }

// TODO(sheniang03): Add timeout for wait, now timeout unused
bool ProcessGroupHeter::HeterTask::Wait(std::chrono::milliseconds timeout) {
  return true;
}

ProcessGroupHeter::ProcessGroupHeter(const std::shared_ptr<Store>& store,
                                     int rank,
                                     int size,
                                     const platform::Place& place,
                                     int gid,
                                     int local_rank,
                                     int local_size,
                                     int gloo_rank,
                                     int gloo_size,
                                     bool with_switch,
                                     std::string switch_endpoint,
                                     int src_rank,
                                     int dst_rank)
    : ProcessGroup(rank, size, place, gid),
      store_(store),
      local_rank_(local_rank),
      local_size_(local_size),
      gloo_rank_(gloo_rank),
      gloo_size_(gloo_size),
      with_switch_(with_switch),
      switch_endpoint_(switch_endpoint),
      src_rank_(src_rank),
      dst_rank_(dst_rank) {
  return;
#ifdef PADDLE_WITH_CUSTOM
  if (paddle::platform::is_custom_place(place_)) {
    inner_pg_ = std::make_shared<ProcessGroupCustom>(
        store, local_rank, local_size, place_, IGNORE_ID);
  } else {
#endif
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    inner_pg_ = std::make_shared<ProcessGroupNCCL>(
        store, local_rank, local_size, place_, IGNORE_ID);
#elif defined(PADDLE_WITH_ASCEND_CL)
  inner_pg_ = std::make_shared<ProcessGroupHCCL>(
      store, local_rank, local_size, place_, IGNORE_ID);
#else
  PADDLE_THROW(platform::errors::Unavailable(
      "ProcessGroupHeter only supports NCCL, RCCL and HCCL now."));
#endif
#ifdef PADDLE_WITH_CUSTOM
  }
#endif

  if (local_rank_ == 0 && !with_switch_) {
    auto opts = ProcessGroupGloo::GlooOptions::create();
    opts->device = ProcessGroupGloo::createDefaultDevice();
    inter_pg_ = std::make_shared<ProcessGroupGloo>(
        store, gloo_rank_, gloo_size_, place_, IGNORE_ID, opts);
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
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    const AllreduceOptions& opts) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCudaPlace(in_tensors),
      true,
      platform::errors::InvalidArgument("All inputs should be in CudaPlace."));
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCudaPlace(out_tensors),
      true,
      platform::errors::InvalidArgument("All outputs should be in CudaPlace."));
#endif

  // Step1: do allreduce in inner cluster
  auto task = inner_pg_->AllReduce(in_tensors, in_tensors, opts);
  task->Wait();

  // Step2: copy tensors to CPU
  if (local_rank_ == 0) {
    std::vector<phi::DenseTensor> cpu_tensors;
    cpu_tensors.reserve(in_tensors.size());
    phi::DenseTensor cpu_tensor;
    for (size_t i = 0; i < in_tensors.size(); i++) {
      auto gpu_tensor = in_tensors[i];
      cpu_tensor.Resize(gpu_tensor.dims());
      framework::TensorCopySync(gpu_tensor, platform::CPUPlace(), &cpu_tensor);
      cpu_tensors.push_back(cpu_tensor);
    }
    // Step3: do inter cluster allreduce
    if (with_switch_) {
      if (local_rank_ == 0) {
        HeterClient* client_ =
            HeterClient::GetInstance({switch_endpoint_}, {}, 0).get();
        auto dense_cpu_tensor = cpu_tensors[0];
        std::vector<int64_t> send_size;
        send_size.push_back(dense_cpu_tensor.numel());
        int ret = client_->Send(
            gid_,
            {dense_cpu_tensor.name()},
            send_size,
            dense_cpu_tensor.data(),
            dense_cpu_tensor.numel() *
                framework::DataTypeSize(dense_cpu_tensor.dtype()));
        PADDLE_ENFORCE_EQ(ret,
                          0,
                          platform::errors::PreconditionNotMet(
                              "Send to the switch module error."));
        phi::DenseTensor cpu_tensor2;
        cpu_tensor2.AllocateFrom(
            std::make_unique<paddle::experimental::DefaultAllocator>(
                paddle::platform::CPUPlace())
                .get(),
            dense_cpu_tensor.dtype(),
            dense_cpu_tensor.numel());
        ret = client_->Recv(
            gid_,
            {dense_cpu_tensor.name()},
            cpu_tensor2.data(),
            cpu_tensor2.numel() * framework::DataTypeSize(cpu_tensor2.dtype()));
        PADDLE_ENFORCE_EQ(ret,
                          0,
                          platform::errors::PreconditionNotMet(
                              "Recv from the switch module error."));

        switch (dense_cpu_tensor.dtype()) {
          case DataType::FLOAT32:
            _do_add<float>(reinterpret_cast<float*>(dense_cpu_tensor.data()),
                           reinterpret_cast<float*>(cpu_tensor2.data()),
                           dense_cpu_tensor.numel());
            break;
          case DataType::FLOAT64:
            _do_add<double>(reinterpret_cast<double*>(dense_cpu_tensor.data()),
                            reinterpret_cast<double*>(cpu_tensor2.data()),
                            dense_cpu_tensor.numel());
            break;
          case DataType::INT32:
            _do_add<int>(reinterpret_cast<int*>(dense_cpu_tensor.data()),
                         reinterpret_cast<int*>(cpu_tensor2.data()),
                         dense_cpu_tensor.numel());
            break;
          default:
            PADDLE_THROW(platform::errors::PreconditionNotMet(
                "Unsupported data type (%s) to do add.",
                framework::DataType2String(dense_cpu_tensor.dtype())));
        }
      }
    } else {
      auto gloo_task = inter_pg_->AllReduce(cpu_tensors, cpu_tensors, opts);
      gloo_task->Wait();
    }
    // Step4: copy cpu tensors to gpu
    // copy cpu tensors to gpu
    for (size_t i = 0; i < in_tensors.size(); i++) {
      auto gpu_tensor = out_tensors[i];
      auto cpu_tensor = cpu_tensors[i];
      framework::TensorCopySync(cpu_tensor, cpu_tensor.place(), &gpu_tensor);
    }
  }

  // Step5: broadcast among inner cluster
  auto b_opts = BroadcastOptions();
  b_opts.source_rank = 0;
  auto broadcast_task = inner_pg_->Broadcast(out_tensors, out_tensors, b_opts);
  broadcast_task->Wait();
  return CreateTask(rank_, CommType::ALLREDUCE, in_tensors);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupHeter::Broadcast(
    std::vector<phi::DenseTensor>& in_tensors,
    std::vector<phi::DenseTensor>& out_tensors,
    const BroadcastOptions& opts) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCudaPlace(in_tensors),
      true,
      platform::errors::InvalidArgument("All inputs should be in CudaPlace."));
  PADDLE_ENFORCE_EQ(
      CheckTensorsInCudaPlace(out_tensors),
      true,
      platform::errors::InvalidArgument("All outputs should be in CudaPlace."));
#endif

  // Step1: do broadcast in inner cluster
  auto b_opts = BroadcastOptions();
  b_opts.source_rank = 0;
  inner_pg_->Broadcast(in_tensors, out_tensors, b_opts);

  if (local_rank_ == 0) {
    std::vector<phi::DenseTensor> cpu_tensors;
    cpu_tensors.reserve(in_tensors.size());
    for (size_t i = 0; i < in_tensors.size(); i++) {
      auto gpu_tensor = in_tensors[i];
      phi::DenseTensor cpu_tensor;
      cpu_tensor.Resize(gpu_tensor.dims());
      framework::TensorCopySync(gpu_tensor, platform::CPUPlace(), &cpu_tensor);
      cpu_tensors.push_back(cpu_tensor);
    }
    if (with_switch_) {
      if (local_rank_ == 0) {
        HeterClient* client_ =
            HeterClient::GetInstance({switch_endpoint_}, {}, 0).get();
        auto dense_cpu_tensor = cpu_tensors[0];
        if (gloo_rank_ == 0) {
          std::vector<int64_t> send_size;
          send_size.push_back(dense_cpu_tensor.numel());
          int ret = client_->Send(
              gid_,
              {dense_cpu_tensor.name()},
              send_size,
              dense_cpu_tensor.data(),
              dense_cpu_tensor.numel() *
                  framework::DataTypeSize(dense_cpu_tensor.dtype()));
          PADDLE_ENFORCE_EQ(ret,
                            0,
                            platform::errors::PreconditionNotMet(
                                "Send to the switch module error."));
        } else {
          int ret = client_->Recv(
              gid_,
              {dense_cpu_tensor.name()},
              dense_cpu_tensor.data(),
              dense_cpu_tensor.numel() *
                  framework::DataTypeSize(dense_cpu_tensor.dtype()));
          PADDLE_ENFORCE_EQ(ret,
                            0,
                            platform::errors::PreconditionNotMet(
                                "Receive from the switch module error."));
        }
      }
    } else {
      auto gloo_task = inter_pg_->Broadcast(cpu_tensors, cpu_tensors, opts);
      gloo_task->Wait();
    }
    for (size_t i = 0; i < in_tensors.size(); i++) {
      auto gpu_tensor = out_tensors[i];
      auto cpu_tensor = cpu_tensors[i];
      framework::TensorCopySync(cpu_tensor, gpu_tensor.place(), &gpu_tensor);
    }
  }
  auto broadcast_task = inner_pg_->Broadcast(out_tensors, out_tensors, b_opts);
  broadcast_task->Wait();
  return CreateTask(rank_, CommType::BROADCAST, in_tensors);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupHeter::Send(
    std::vector<phi::DenseTensor>& in_tensors, int peer) {
  PADDLE_ENFORCE_EQ(
      in_tensors.size(),
      1,
      platform::errors::PreconditionNotMet(
          "For each send operation, there can only be one tensor to send."));
  // Copy Tensor to cpu
  auto start = std::chrono::high_resolution_clock::now();
  phi::DenseTensor cpu_tensor;
  auto& gpu_tensor = in_tensors[0];
  framework::TensorCopySync(gpu_tensor, platform::CPUPlace(), &cpu_tensor);
  PADDLE_ENFORCE_EQ(with_switch_,
                    true,
                    platform::errors::PreconditionNotMet(
                        "Gloo does not support the send operation."));
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  VLOG(2) << "Time to copy tensor of dims(" << cpu_tensor.dims()
          << ") from gpu to cpu for send " << std::setw(9)
          << " is: " << diff.count() << " s" << std::endl;

  // Send to switch
  HeterClient* client_ =
      HeterClient::GetInstance({switch_endpoint_}, {}, 0).get();
  int64_t tensor_size =
      cpu_tensor.numel() * framework::DataTypeSize(cpu_tensor.dtype());
  std::vector<int64_t> send_size;
  send_size.push_back(tensor_size);
  auto id = src_rank_ * 10000 + dst_rank_;
  std::string tensor_name = std::to_string(gid_) + "_id_" + std::to_string(id) +
                            std::string("_") + std::to_string(send_count++);
  VLOG(2) << "tensor_name:" << tensor_name;
  int ret = client_->Send(
      gid_, {tensor_name}, send_size, cpu_tensor.data(), tensor_size);
  PADDLE_ENFORCE_EQ(
      ret,
      0,
      platform::errors::PreconditionNotMet("Send to the switch module error."));
  return CreateTask(rank_, CommType::SEND, in_tensors);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupHeter::Recv(
    std::vector<phi::DenseTensor>& out_tensors, int peer) {
  PADDLE_ENFORCE_EQ(
      out_tensors.size(),
      1,
      platform::errors::PreconditionNotMet(
          "For each rece operation, there can only be one tensor to receive."));

  // Copy Tensor to cpu
  phi::DenseTensor cpu_tensor;
  auto& gpu_tensor = out_tensors[0];
  cpu_tensor.Resize(gpu_tensor.dims());
  cpu_tensor.set_layout(gpu_tensor.layout());
  cpu_tensor.mutable_data(platform::CPUPlace(), gpu_tensor.dtype());

  PADDLE_ENFORCE_EQ(with_switch_,
                    true,
                    platform::errors::PreconditionNotMet(
                        "Gloo does not support the send operation."));
  // recv from switch
  HeterClient* client_ =
      HeterClient::GetInstance({switch_endpoint_}, {}, 0).get();
  auto id = src_rank_ * 10000 + dst_rank_;
  std::string tensor_name = std::to_string(gid_) + "_id_" + std::to_string(id) +
                            std::string("_") + std::to_string(recv_count++);
  VLOG(2) << "tensor_name: " << tensor_name;
  auto start = std::chrono::high_resolution_clock::now();
  int ret = client_->Recv(
      gid_,
      {tensor_name},
      cpu_tensor.data(),
      cpu_tensor.numel() * framework::DataTypeSize(cpu_tensor.dtype()));
  PADDLE_ENFORCE_EQ(ret,
                    0,
                    platform::errors::PreconditionNotMet(
                        "receive to the switch module error."));
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  double goodput = cpu_tensor.numel() *
                   framework::DataTypeSize(cpu_tensor.dtype()) / diff.count();
  VLOG(2) << "Goodput: " << goodput << "B/s" << std::endl;
  start = std::chrono::high_resolution_clock::now();
  framework::TensorCopySync(cpu_tensor, gpu_tensor.place(), &gpu_tensor);
  end = std::chrono::high_resolution_clock::now();
  diff = end - start;
  VLOG(2) << "Time to copy tensor of dims(" << cpu_tensor.dims()
          << ") from cpu to gpu for recv " << std::setw(9)
          << " is: " << diff.count() << " s" << std::endl;
  return CreateTask(rank_, CommType::RECV, out_tensors);
}

}  // namespace distributed
}  // namespace paddle
