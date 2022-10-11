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

#if defined(PADDLE_WITH_XPU_BKCL)

#include "paddle/fluid/imperative/bkcl_context.h"

#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/xpu/bkcl_helper.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/gen_comm_id_helper.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/string/split.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace imperative {

static void AllReduce(const phi::DenseTensor &src,
                      phi::DenseTensor *dst,
                      const XPUStream stream,
                      const platform::BKCLComm *comm) {
  const auto &place = src.place();
  PADDLE_ENFORCE_EQ(
      platform::is_xpu_place(place),
      true,
      platform::errors::Unimplemented(
          "Dynamic graph mode does not support multi-CPU training yet."));

  const void *src_ptr = src.data();
  dst->Resize(src.dims());
  auto *dst_ptr = dst->mutable_data(src.place(), src.dtype());
  auto bkcl_dtype =
      platform::ToBKCLDataType(framework::TransToProtoVarType(src.dtype()));

  PADDLE_ENFORCE_EQ(
      bkcl_all_reduce(comm->comm(),
                      src_ptr,
                      dst_ptr,
                      src.numel(),
                      bkcl_dtype,
                      BKCL_ADD,
                      stream),
      BKCL_SUCCESS,
      platform::errors::PreconditionNotMet("BKCL all reduce failed"));
}
/*
Baidu Kunlun Communication Library(BKCL) is designed for multi Baidu Kunlun
cards communication
as NVIDIA Collective Communications Library(NCCL) in multi Nvidia GPU cards.
Please refer to bkcl.h in xpu.tar.gz linked in cmake/external/xpu.cmake.
*/
void BKCLParallelContext::BcastBKCLId(
    std::vector<BKCLUniqueId> &bkcl_ids,  // NOLINT
    int root) {
  if (strategy_.local_rank_ == root) {
    std::vector<std::string> other_trainers;
    for (auto &ep : strategy_.trainer_endpoints_) {
      if (ep != strategy_.current_endpoint_) {
        other_trainers.push_back(ep);
      }
    }
    platform::SendBroadCastCommID(other_trainers, &bkcl_ids);
  } else {
    platform::RecvBroadCastCommID(strategy_.current_endpoint_, &bkcl_ids);
  }
}

void BKCLParallelContext::Init() {
  std::vector<BKCLUniqueId> bkcl_ids;
  bkcl_ids.resize(strategy_.nrings_);

  if (strategy_.local_rank_ == 0) {
    // generate the unique bkclid on the root worker
    for (size_t i = 0; i < bkcl_ids.size(); ++i) {
      auto ret = bkcl_get_unique_id(&bkcl_ids[i]);
      PADDLE_ENFORCE_EQ(BKCL_SUCCESS,
                        ret,
                        platform::errors::PreconditionNotMet(
                            "BKCL get unique id failed [%d]", ret));
    }
  }
  BcastBKCLId(bkcl_ids, 0);

  int xpu_id = place_.device;
  for (int ring_id = 0; ring_id < strategy_.nrings_; ring_id++) {
    VLOG(0) << "init BKCL context nranks: " << strategy_.nranks_
            << " local rank: " << strategy_.local_rank_ << " xpu id: " << xpu_id
            << " ring id: " << ring_id;
    // it will assign bkcl_comm in XPUDeviceContext within ring_id
    platform::BKCLCommContext::Instance().CreateComm(&bkcl_ids[ring_id],
                                                     strategy_.nranks_,
                                                     strategy_.local_rank_,
                                                     xpu_id,
                                                     ring_id);
    compute_events_.emplace_back(
        platform::XpuEventResourcePool::Instance().New(place_.device));
    comm_events_.emplace_back(
        platform::XpuEventResourcePool::Instance().New(place_.device));
  }
}

void BKCLParallelContext::InitWithRingID(int ring_id) {
  std::vector<BKCLUniqueId> bkcl_ids;
  bkcl_ids.resize(1);

  if (strategy_.local_rank_ == 0) {
    // generate the unique bkclid on the root worker
    auto ret = bkcl_get_unique_id(&bkcl_ids[0]);
    PADDLE_ENFORCE_EQ(BKCL_SUCCESS,
                      ret,
                      platform::errors::PreconditionNotMet(
                          "BKCL get unique id failed [%d]", ret));
  }
  BcastBKCLId(bkcl_ids, 0);

  int xpu_id = place_.device;
  VLOG(0) << "init BKCL context nranks: " << strategy_.nranks_
          << " local rank: " << strategy_.local_rank_ << " xpu id: " << xpu_id
          << " ring id: " << ring_id;
  // it will assign bkcl_comm in XPUDeviceContext within ring_id
  platform::BKCLCommContext::Instance().CreateComm(
      &bkcl_ids[0], strategy_.nranks_, strategy_.local_rank_, xpu_id, ring_id);

  compute_events_.emplace_back(
      platform::XpuEventResourcePool::Instance().New(place_.device));
  comm_events_.emplace_back(
      platform::XpuEventResourcePool::Instance().New(place_.device));
}

void BKCLParallelContext::AllReduceByStream(const framework::Variable &src,
                                            framework::Variable *dst,
                                            int ring_id,
                                            bool use_calc_stream) {
  PADDLE_ENFORCE_EQ(
      platform::is_xpu_place(place_),
      true,
      platform::errors::Unimplemented(
          "Dynamic graph mode does not support multi-CPU training yet."));
  auto place = place_;

  auto *dev_ctx = static_cast<platform::XPUDeviceContext *>(
      platform::DeviceContextPool::Instance().Get(place));
  platform::BKCLComm *comm =
      platform::BKCLCommContext::Instance().Get(ring_id, place);
  XPUStream stream =
      use_calc_stream ? dev_ctx->x_context()->xpu_stream : comm->stream();

  if (src.IsType<framework::LoDTensor>()) {
    if (!dst->IsType<framework::LoDTensor>()) {
      dst->Clear();
    }
    AllReduce(src.Get<framework::LoDTensor>(),
              dst->GetMutable<framework::LoDTensor>(),
              stream,
              comm);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "XPU unsupported variable type %s for imperative allreduce, only "
        "LoDTensor are supported.",
        platform::demangle(framework::ToTypeName(src.Type()))));
  }
}

void BKCLParallelContext::Broadcast(framework::Variable *src, int ring_id) {
  VLOG(3) << "/// DEBUG /// start inter broadcast with ring_id: " << ring_id;
  phi::DenseTensor *src_tensor = src->GetMutable<framework::LoDTensor>();
  const auto &place = src_tensor->place();
  platform::BKCLComm *comm =
      platform::BKCLCommContext::Instance().Get(ring_id, place);
  XPUStream stream = comm->stream();

  void *src_ptr = src_tensor->data();
  auto data_type = platform::ToBKCLDataType(
      framework::TransToProtoVarType(src_tensor->dtype()));

  PADDLE_ENFORCE_EQ(bkcl_broadcast(comm->comm(),
                                   src_ptr,
                                   src_ptr,
                                   src_tensor->numel(),
                                   data_type,
                                   0,
                                   stream),
                    BKCL_SUCCESS,
                    platform::errors::Unavailable("bkcl_broadcast failed"));
}

paddle::platform::DeviceContext *BKCLParallelContext::GetDeviceContext(
    int ring_id) {
  return static_cast<platform::DeviceContext *>(
      platform::BKCLCommContext::Instance()
          .Get(ring_id, place_)
          ->dev_context());
}

void BKCLParallelContext::WaitCompute(int ring_id) {
  PADDLE_ENFORCE_GE(ring_id,
                    0,
                    platform::errors::OutOfRange(
                        "Ring id expected >= 0, but got %d", ring_id));
  PADDLE_ENFORCE_LT(
      ring_id,
      strategy_.nrings_,
      platform::errors::OutOfRange("Ring id expected < nrings,"
                                   "but got ring id = %d, nrings = %d",
                                   ring_id,
                                   strategy_.nrings_));
  auto compute_stream = static_cast<platform::XPUDeviceContext *>(
                            platform::DeviceContextPool::Instance().Get(place_))
                            ->stream();
  auto comm_stream = platform::BKCLCommContext::Instance()
                         .Get(ring_id, place_)
                         ->dev_context()
                         ->stream();
  auto event = compute_events_[ring_id].get();

  // compute_stream-->event-->comm_stream
  PADDLE_ENFORCE_XPU_SUCCESS(xpu_event_record(event, compute_stream));
  PADDLE_ENFORCE_XPU_SUCCESS(xpu_stream_wait_event(comm_stream, event));
}

void BKCLParallelContext::WaitComm(int ring_id) {
  PADDLE_ENFORCE_GE(ring_id,
                    0,
                    platform::errors::OutOfRange(
                        "Ring id expected >= 0, but got %d", ring_id));
  PADDLE_ENFORCE_LT(
      ring_id,
      strategy_.nrings_,
      platform::errors::OutOfRange("Ring id expected < nrings,"
                                   "but got ring id = %d, nrings = %d",
                                   ring_id,
                                   strategy_.nrings_));
  auto comm_stream = platform::BKCLCommContext::Instance()
                         .Get(ring_id, place_)
                         ->dev_context()
                         ->stream();
  auto compute_stream = static_cast<platform::XPUDeviceContext *>(
                            platform::DeviceContextPool::Instance().Get(place_))
                            ->stream();
  auto event = compute_events_[ring_id].get();

  // comm_stream-->event-->compute_stream
  PADDLE_ENFORCE_XPU_SUCCESS(xpu_event_record(event, comm_stream));
  PADDLE_ENFORCE_XPU_SUCCESS(xpu_stream_wait_event(compute_stream, event));
}

void BKCLParallelContext::SynchronizeCompute() {
  auto compute_dev_ctx = static_cast<platform::XPUDeviceContext *>(
      platform::DeviceContextPool::Instance().Get(place_));
  compute_dev_ctx->Wait();
}

}  //  namespace imperative
}  //  namespace paddle
#endif
