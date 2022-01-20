//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/imperative/hccl_context.h"

#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/variable.h"

#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/gen_comm_id_helper.h"
#include "paddle/fluid/platform/place.h"

#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/npu/hccl_helper.h"

namespace paddle {
namespace framework {
class Variable;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace imperative {

static void AllReduce(const framework::Tensor &src, framework::Tensor *dst,
                      const aclrtStream stream,
                      const platform::HCCLComm *comm) {
  const auto &place = src.place();
  PADDLE_ENFORCE_EQ(
      platform::is_npu_place(place), true,
      platform::errors::Unimplemented(
          "Imperative mode does not support multi-CPU training yet."));

  void *src_ptr = const_cast<void *>(src.data());
  dst->Resize(src.dims());
  void *dst_ptr = dst->mutable_data(src.place(), src.type());
  HcclDataType hccl_dtype = platform::ToHCCLDataType(src.type());

  PADDLE_ENFORCE_NPU_SUCCESS(platform::dynload::HcclAllReduce(
      src_ptr, dst_ptr, src.numel(), hccl_dtype, HCCL_REDUCE_SUM, comm->comm(),
      reinterpret_cast<void *>(stream)));
}

void HCCLParallelContext::BcastHCCLId(
    std::vector<HcclRootInfo> &hccl_ids,  // NOLINT
    int root, int server_fd) {
  if (strategy_.local_rank_ == root) {
    std::vector<std::string> other_trainers;
    for (auto &ep : strategy_.trainer_endpoints_) {
      if (ep != strategy_.current_endpoint_) {
        other_trainers.push_back(ep);
      }
    }
    platform::SendBroadCastCommID(other_trainers, &hccl_ids);
  } else {
    platform::RecvBroadCastCommID(server_fd, strategy_.current_endpoint_,
                                  &hccl_ids);
  }
}

void HCCLParallelContext::Init() {
  int server_fd = -1;

  std::vector<HcclRootInfo> hccl_ids;
  hccl_ids.resize(strategy_.nrings_);

  if (strategy_.local_rank_ == 0) {
    // generate the unique hcclid on the root worker
    for (size_t i = 0; i < hccl_ids.size(); ++i) {
      platform::dynload::HcclGetRootInfo(&hccl_ids[i]);
    }
  } else {
    server_fd = platform::SocketServer::GetInstance(strategy_.current_endpoint_)
                    .socket();
  }
  BcastHCCLId(hccl_ids, 0, server_fd);

  int npu_id = place_.device;
  for (int ring_id = 0; ring_id < strategy_.nrings_; ring_id++) {
    VLOG(0) << "init hccl context nranks: " << strategy_.nranks_
            << " local rank: " << strategy_.local_rank_ << " npu id: " << npu_id
            << " ring id: " << ring_id;
    // it will assign hccl_comm in NPUDeviceContext within ring_id
    platform::HCCLCommContext::Instance().CreateHCCLComm(
        &hccl_ids[ring_id], strategy_.nranks_, strategy_.local_rank_, npu_id,
        ring_id);

    compute_events_.emplace_back(
        platform::NpuEventResourcePool::Instance().New(place_.device));
    comm_events_.emplace_back(
        platform::NpuEventResourcePool::Instance().New(place_.device));
  }
}

void HCCLParallelContext::InitWithRingID(int ring_id) {
  int server_fd = -1;
  std::vector<HcclRootInfo> hccl_ids;
  hccl_ids.resize(1);

  if (strategy_.local_rank_ == 0) {
    // generate the unique hcclid on the root worker
    platform::dynload::HcclGetRootInfo(&hccl_ids[0]);
  } else {
    server_fd = platform::SocketServer::GetInstance(strategy_.current_endpoint_)
                    .socket();
  }
  BcastHCCLId(hccl_ids, 0, server_fd);

  int npu_id = place_.device;
  VLOG(0) << "init hccl context nranks: " << strategy_.nranks_
          << " local rank: " << strategy_.local_rank_ << " npu id: " << npu_id
          << " ring id: " << ring_id;
  // it will assign hccl_comm in NPUDeviceContext within ring_id
  platform::HCCLCommContext::Instance().CreateHCCLComm(
      &hccl_ids[0], strategy_.nranks_, strategy_.local_rank_, npu_id, ring_id);

  compute_events_.emplace_back(
      platform::NpuEventResourcePool::Instance().New(place_.device));
  comm_events_.emplace_back(
      platform::NpuEventResourcePool::Instance().New(place_.device));
}

void HCCLParallelContext::AllReduceByStream(const framework::Variable &src,
                                            framework::Variable *dst,
                                            int ring_id, bool use_calc_stream) {
  PADDLE_ENFORCE_EQ(
      platform::is_npu_place(place_), true,
      platform::errors::Unimplemented(
          "Dynamic graph mode does not support multi-CPU training yet."));
  auto *dev_ctx = static_cast<platform::NPUDeviceContext *>(
      platform::DeviceContextPool::Instance().Get(place_));
  platform::HCCLComm *comm =
      platform::HCCLCommContext::Instance().Get(ring_id, place_);
  aclrtStream stream = use_calc_stream ? dev_ctx->stream() : comm->stream();

  if (src.IsType<framework::LoDTensor>()) {
    if (!dst->IsType<framework::LoDTensor>()) {
      dst->Clear();
    }
    AllReduce(src.Get<framework::LoDTensor>(),
              dst->GetMutable<framework::LoDTensor>(), stream, comm);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "XPU unsupported variable type %s for imperative allreduce, only "
        "LoDTensor are supported.",
        platform::demangle(framework::ToTypeName(src.Type()))));
  }
}

void HCCLParallelContext::Broadcast(framework::Variable *src, int ring_id) {
  VLOG(3) << "/// DEBUG /// start inter broadcast with ring_id: " << ring_id;
  if (src->IsType<framework::LoDTensor>()) {
    framework::Tensor *src_tensor = src->GetMutable<framework::LoDTensor>();
    const auto &place = src_tensor->place();
    platform::HCCLComm *comm =
        platform::HCCLCommContext::Instance().Get(ring_id, place);
    aclrtStream stream = comm->stream();

    void *src_ptr =
        reinterpret_cast<void *>(const_cast<void *>(src_tensor->data()));
    auto hccl_dtype = platform::ToHCCLDataType(src_tensor->type());
    PADDLE_ENFORCE_NPU_SUCCESS(platform::dynload::HcclBroadcast(
        src_ptr, src_tensor->numel(), hccl_dtype, 0, comm->comm(),
        reinterpret_cast<void *>(stream)));
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Unsupported variable type %s for imperative allreduce, only "
        "LoDTensor is supported.",
        platform::demangle(framework::ToTypeName(src->Type()))));
  }
}

paddle::platform::DeviceContext *HCCLParallelContext::GetDeviceContext(
    int ring_id) {
  return static_cast<platform::DeviceContext *>(
      platform::HCCLCommContext::Instance()
          .Get(ring_id, place_)
          ->dev_context());
}

void HCCLParallelContext::WaitCompute(int ring_id) {
  PADDLE_ENFORCE_GE(ring_id, 0, platform::errors::OutOfRange(
                                    "ring id must >= 0, but got %d", ring_id));
  PADDLE_ENFORCE_LT(ring_id, compute_events_.size(),
                    platform::errors::OutOfRange(
                        "ring id must < compute events size,"
                        "but got ring id = %d, compute events size = %d",
                        ring_id, compute_events_.size()));

  auto compute_stream = static_cast<platform::NPUDeviceContext *>(
                            platform::DeviceContextPool::Instance().Get(place_))
                            ->stream();
  auto comm_stream =
      platform::HCCLCommContext::Instance().Get(ring_id, place_)->stream();
  auto event = compute_events_[ring_id].get();

  // compute_stream-->event-->comm_stream
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtRecordEvent(event, compute_stream));
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtStreamWaitEvent(comm_stream, event));
}

void HCCLParallelContext::WaitComm(int ring_id) {
  PADDLE_ENFORCE_GE(ring_id, 0, platform::errors::OutOfRange(
                                    "ring id must >= 0, but got %d", ring_id));
  PADDLE_ENFORCE_LT(ring_id, comm_events_.size(),
                    platform::errors::OutOfRange(
                        "ring id must < comm events size,"
                        "but got ring id = %d, comm events size = %d",
                        ring_id, comm_events_.size()));

  auto compute_stream = static_cast<platform::NPUDeviceContext *>(
                            platform::DeviceContextPool::Instance().Get(place_))
                            ->stream();
  auto comm_stream =
      platform::HCCLCommContext::Instance().Get(ring_id, place_)->stream();
  auto event = comm_events_[ring_id].get();

  // comm_stream-->event-->compute_stream
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtRecordEvent(event, comm_stream));
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtStreamWaitEvent(compute_stream, event));
}

void HCCLParallelContext::SynchronizeCompute() {
  auto *compute_dev_ctx = static_cast<platform::NPUDeviceContext *>(
      platform::DeviceContextPool::Instance().Get(place_));
  compute_dev_ctx->Wait();
}

}  //  namespace imperative
}  //  namespace paddle
