/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#if defined(PADDLE_WITH_CNCL)
#include "paddle/fluid/imperative/cncl_context.h"

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/variable.h"

#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/gen_comm_id_helper.h"
#include "paddle/fluid/platform/place.h"

#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/mlu/cncl_helper.h"
#include "paddle/fluid/platform/device/mlu/mlu_info.h"

namespace paddle {
namespace framework {
class Variable;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace imperative {

static void AllReduce(const framework::Tensor &src, framework::Tensor *dst,
                      const mluStream stream, const platform::CNCLComm *comm) {
  const auto &place = src.place();
  PADDLE_ENFORCE_EQ(
      platform::is_mlu_place(place), true,
      platform::errors::Unimplemented(
          "Imperative mode does not support multi-CPU training yet."));

  const void *src_ptr = src.data();
  dst->Resize(src.dims());
  auto *dst_ptr = dst->mutable_data(src.place(), src.dtype());
  auto cncl_dtype =
      platform::ToCNCLDataType(framework::TransToProtoVarType(src.dtype()));
  PADDLE_ENFORCE_MLU_SUCCESS(cnclAllReduce(src_ptr, dst_ptr, src.numel(),
                                           cncl_dtype, cnclSum, comm->comm(),
                                           stream));
}

void CNCLParallelContext::BcastCNCLId(
    std::vector<cnclCliqueId> &cncl_ids,  // NOLINT
    int root, int server_fd) {
  if (strategy_.local_rank_ == root) {
    std::vector<std::string> other_trainers;
    for (auto &ep : strategy_.trainer_endpoints_) {
      if (ep != strategy_.current_endpoint_) {
        other_trainers.push_back(ep);
      }
    }
    platform::SendBroadCastCommID(other_trainers, &cncl_ids);
  } else {
    platform::RecvBroadCastCommID(server_fd, strategy_.current_endpoint_,
                                  &cncl_ids);
  }
}

void CNCLParallelContext::Init() {
  int server_fd = -1;

  std::vector<cnclCliqueId> cncl_ids;
  cncl_ids.resize(strategy_.nrings_);

  if (strategy_.local_rank_ == 0) {
    // generate the unique cnclid on the root worker
    for (size_t i = 0; i < cncl_ids.size(); ++i) {
      PADDLE_ENFORCE_MLU_SUCCESS(cnclGetCliqueId(&cncl_ids[i]));
    }
  } else {
    server_fd = platform::SocketServer::GetInstance(strategy_.current_endpoint_)
                    .socket();
  }
  BcastCNCLId(cncl_ids, 0, server_fd);

  int mlu_id = place_.device;
  for (int ring_id = 0; ring_id < strategy_.nrings_; ++ring_id) {
    VLOG(0) << "init cncl context nranks: " << strategy_.nranks_
            << " local rank: " << strategy_.local_rank_ << " mlu id: " << mlu_id
            << " ring id: " << ring_id;
    // it will assign cncl_comm in MLUDeviceContext within ring_id
    platform::CNCLCommContext::Instance().CreateComm(
        &cncl_ids[ring_id], strategy_.nranks_, strategy_.local_rank_, mlu_id,
        ring_id);

    compute_events_.emplace_back(
        platform::MluEventResourcePool::Instance().New(place_.device));
    comm_events_.emplace_back(
        platform::MluEventResourcePool::Instance().New(place_.device));
  }
}

void CNCLParallelContext::InitWithRingID(int ring_id) {
  int server_fd = -1;
  std::vector<cnclCliqueId> cncl_ids;
  cncl_ids.resize(1);

  if (strategy_.local_rank_ == 0) {
    // generate the unique cnclid on the root worker
    PADDLE_ENFORCE_MLU_SUCCESS(cnclGetCliqueId(&cncl_ids[0]));
  } else {
    server_fd = platform::SocketServer::GetInstance(strategy_.current_endpoint_)
                    .socket();
  }
  BcastCNCLId(cncl_ids, 0, server_fd);

  int mlu_id = place_.device;
  VLOG(0) << "init cncl context nranks: " << strategy_.nranks_
          << " local rank: " << strategy_.local_rank_ << " mlu id: " << mlu_id
          << " ring id: " << ring_id;
  // it will assign cncl_comm in MLUDeviceContext within ring_id
  platform::CNCLCommContext::Instance().CreateComm(
      &cncl_ids[0], strategy_.nranks_, strategy_.local_rank_, mlu_id, ring_id);

  compute_events_.emplace_back(
      platform::MluEventResourcePool::Instance().New(place_.device));
  comm_events_.emplace_back(
      platform::MluEventResourcePool::Instance().New(place_.device));
}

void CNCLParallelContext::AllReduceByStream(const framework::Variable &src,
                                            framework::Variable *dst,
                                            int ring_id, bool use_calc_stream) {
  PADDLE_ENFORCE_EQ(
      platform::is_mlu_place(place_), true,
      platform::errors::Unimplemented(
          "Dynamic graph mode does not support multi-CPU training yet."));
  auto *dev_ctx = static_cast<platform::MLUDeviceContext *>(
      platform::DeviceContextPool::Instance().Get(place_));
  platform::CNCLComm *comm =
      platform::CNCLCommContext::Instance().Get(ring_id, place_);
  mluStream stream = (use_calc_stream ? dev_ctx->stream() : comm->stream());

  if (src.IsType<framework::LoDTensor>()) {
    if (!dst->IsType<framework::LoDTensor>()) {
      dst->Clear();
    }
    AllReduce(src.Get<framework::LoDTensor>(),
              dst->GetMutable<framework::LoDTensor>(), stream, comm);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Unsupported variable type %s for imperative allreduce, only "
        "LoDTensor is supported.",
        platform::demangle(framework::ToTypeName(src.Type()))));
  }
}

void CNCLParallelContext::Broadcast(framework::Variable *src, int ring_id) {
  VLOG(3) << "/// DEBUG /// start inter broadcast with ring_id: " << ring_id;
  framework::Tensor *src_tensor = src->GetMutable<framework::LoDTensor>();
  const auto &place = src_tensor->place();
  platform::CNCLComm *comm =
      platform::CNCLCommContext::Instance().Get(ring_id, place);
  mluStream stream = comm->stream();

  void *src_ptr = src_tensor->data();
  auto cncl_dtype = platform::ToCNCLDataType(
      framework::TransToProtoVarType(src_tensor->dtype()));
  PADDLE_ENFORCE_MLU_SUCCESS(cnclBcast(src_ptr, src_tensor->numel(), cncl_dtype,
                                       0, comm->comm(), stream));
}

paddle::platform::DeviceContext *CNCLParallelContext::GetDeviceContext(
    int ring_id) {
  return static_cast<platform::DeviceContext *>(
      platform::CNCLCommContext::Instance()
          .Get(ring_id, place_)
          ->dev_context());
}

void CNCLParallelContext::WaitCompute(int ring_id) {
  PADDLE_ENFORCE_GE(ring_id, 0, platform::errors::OutOfRange(
                                    "ring id must >= 0, but got %d", ring_id));
  PADDLE_ENFORCE_LT(ring_id, compute_events_.size(),
                    platform::errors::OutOfRange(
                        "ring id must < compute events size,"
                        "but got ring id = %d, compute events size = %d",
                        ring_id, compute_events_.size()));

  auto compute_stream = static_cast<platform::MLUDeviceContext *>(
                            platform::DeviceContextPool::Instance().Get(place_))
                            ->stream();
  auto comm_stream =
      platform::CNCLCommContext::Instance().Get(ring_id, place_)->stream();
  auto event = compute_events_[ring_id].get();

  // compute_stream-->event-->comm_stream
  PADDLE_ENFORCE_MLU_SUCCESS(cnrtPlaceNotifier(event, compute_stream));
  PADDLE_ENFORCE_MLU_SUCCESS(cnrtQueueWaitNotifier(event, comm_stream, 0));
}

void CNCLParallelContext::WaitComm(int ring_id) {
  PADDLE_ENFORCE_GE(ring_id, 0, platform::errors::OutOfRange(
                                    "ring id must >= 0, but got %d", ring_id));
  PADDLE_ENFORCE_LT(ring_id, comm_events_.size(),
                    platform::errors::OutOfRange(
                        "ring id must < comm events size,"
                        "but got ring id = %d, comm events size = %d",
                        ring_id, comm_events_.size()));

  auto compute_stream = static_cast<platform::MLUDeviceContext *>(
                            platform::DeviceContextPool::Instance().Get(place_))
                            ->stream();
  auto comm_stream =
      platform::CNCLCommContext::Instance().Get(ring_id, place_)->stream();
  auto event = comm_events_[ring_id].get();

  // comm_stream-->event-->compute_stream
  PADDLE_ENFORCE_MLU_SUCCESS(cnrtPlaceNotifier(event, comm_stream));
  PADDLE_ENFORCE_MLU_SUCCESS(cnrtQueueWaitNotifier(event, compute_stream, 0));
}

void CNCLParallelContext::SynchronizeCompute() {
  auto *compute_dev_ctx = static_cast<platform::MLUDeviceContext *>(
      platform::DeviceContextPool::Instance().Get(place_));
  compute_dev_ctx->Wait();
}

}  //  namespace imperative
}  //  namespace paddle

#endif
