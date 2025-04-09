//   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/imperative/xccl_context.h"

#if defined(PADDLE_WITH_CUSTOM_DEVICE)
#include "paddle/phi/core/platform/collective_helper.h"
#include "paddle/phi/core/platform/gen_comm_id_helper.h"
#endif

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/platform/device_context.h"

namespace paddle {
namespace framework {
class Variable;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace imperative {

static void XcclAllReduce(const phi::DenseTensor &src,
                          phi::DenseTensor *dst,
                          const phi::stream::Stream &stream,
                          const phi::ccl::CCLComm &comm) {
  const auto &place = src.place();
  PADDLE_ENFORCE_EQ(
      phi::is_custom_place(place),
      true,
      common::errors::Unimplemented(
          "Dynamic graph mode does not support multi-CPU training yet."));

  void *src_ptr = const_cast<void *>(src.data());
  dst->Resize(src.dims());
  auto *dst_ptr = phi::DeviceContextPool::Instance()
                      .Get(src.place())
                      ->Alloc(dst, src.dtype());

  phi::DeviceManager::CCLAllReduce(place.GetDeviceType(),
                                   src_ptr,
                                   dst_ptr,
                                   src.numel(),
                                   src.dtype(),
                                   phi::ccl::CCLReduceOp::SUM,
                                   comm,
                                   stream);
}

void XCCLParallelContext::BcastXCCLId(
    std::vector<phi::ccl::CCLRootId> &xccl_ids,  // NOLINT
    int root,
    int server_fd) {
  if (strategy_.local_rank_ == root) {
    std::vector<std::string> other_trainers;
    for (auto &ep : strategy_.trainer_endpoints_) {
      if (ep != strategy_.current_endpoint_) {
        other_trainers.push_back(ep);
      }
    }
    platform::SendBroadCastCommID(other_trainers, &xccl_ids);
  } else {
    platform::RecvBroadCastCommID(
        server_fd, strategy_.current_endpoint_, &xccl_ids);
  }
}

void XCCLParallelContext::Init() {
  int server_fd = -1;

  std::vector<phi::ccl::CCLRootId> xccl_ids;
  xccl_ids.resize(strategy_.nrings_);

  if (strategy_.local_rank_ == 0) {
    // generate the unique ncclid on the root worker
    for (size_t i = 0; i < xccl_ids.size(); ++i) {
      phi::DeviceManager::CCLGetUniqueId(place_.GetDeviceType(), &xccl_ids[i]);
    }
  } else {
    // FIXME(wangxi): gloo will use rank0 endpoint, so not create socket server
    // on rank0.
    server_fd = platform::SocketServer::GetInstance(strategy_.current_endpoint_)
                    .socket();
  }
  BcastXCCLId(xccl_ids, 0, server_fd);

  int dev_id = place_.device;  // NOLINT
  for (int ring_id = 0; ring_id < strategy_.nrings_; ring_id++) {
    VLOG(0) << "init nccl context nranks: " << strategy_.nranks_
            << " local rank: " << strategy_.local_rank_ << " dev id: " << dev_id
            << " ring id: " << ring_id;
    // it will assign nccl_comm in phi::CustomContext within ring_id
    platform::XCCLCommContext::Instance(place_.GetDeviceType())
        .CreateComm(&xccl_ids[ring_id],
                    strategy_.nranks_,
                    strategy_.local_rank_,
                    dev_id,
                    ring_id);
    auto compute_event = std::make_shared<phi::event::Event>();
    auto comm_event = std::make_shared<phi::event::Event>();
    compute_event->Init(place_);
    comm_event->Init(place_);
    compute_events_.emplace_back(compute_event);
    comm_events_.emplace_back(comm_event);
  }
}

void XCCLParallelContext::InitWithRingID(int ring_id) {
  int server_fd = -1;
  std::vector<phi::ccl::CCLRootId> xccl_ids;
  xccl_ids.resize(1);

  if (strategy_.local_rank_ == 0) {
    // generate the unique ncclid on the root worker
    phi::DeviceManager::CCLGetUniqueId(place_.GetDeviceType(), &xccl_ids[0]);
  } else {
    // FIXME(wangxi): gloo will use rank0 endpoint, so not create socket server
    // on rank0.
    server_fd = platform::SocketServer::GetInstance(strategy_.current_endpoint_)
                    .socket();
  }
  BcastXCCLId(xccl_ids, 0, server_fd);

  int dev_id = place_.device;  // NOLINT
  VLOG(0) << "init xccl context nranks: " << strategy_.nranks_
          << " local rank: " << strategy_.local_rank_ << " dev id: " << dev_id
          << " ring id: " << ring_id;
  // it will assign xccl_comm in phi::CustomContext within ring_id
  platform::XCCLCommContext::Instance(place_.GetDeviceType())
      .CreateComm(&xccl_ids[0],
                  strategy_.nranks_,
                  strategy_.local_rank_,
                  dev_id,
                  ring_id);

  auto compute_event = std::make_shared<phi::event::Event>();
  auto comm_event = std::make_shared<phi::event::Event>();
  compute_event->Init(place_);
  comm_event->Init(place_);
  compute_events_.emplace_back(compute_event);
  comm_events_.emplace_back(comm_event);
}

void XCCLParallelContext::AllReduceByStream(const framework::Variable &src,
                                            framework::Variable *dst,
                                            int ring_id,
                                            bool use_calc_stream) {
  PADDLE_ENFORCE_EQ(
      phi::is_custom_place(place_),
      true,
      common::errors::Unimplemented(
          "Dynamic graph mode does not support multi-CPU training yet."));
  auto place = place_;

  auto *dev_ctx = static_cast<phi::CustomContext *>(
      phi::DeviceContextPool::Instance().Get(place));
  platform::XCCLComm *comm =
      platform::XCCLCommContext::Instance(place.GetDeviceType())
          .Get(ring_id, place);
  auto stream = use_calc_stream ? dev_ctx->GetStream() : comm->stream();

  if (src.IsType<phi::DenseTensor>()) {
    if (!dst->IsType<phi::DenseTensor>()) {
      dst->Clear();
    }
    XcclAllReduce(src.Get<phi::DenseTensor>(),
                  dst->GetMutable<phi::DenseTensor>(),
                  *stream,
                  comm->comm());
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "custom device unsupported variable type %s for imperative allreduce, "
        "only "
        "DenseTensor are supported.",
        common::demangle(framework::ToTypeName(src.Type()))));
  }
}

void XCCLParallelContext::Broadcast(framework::Variable *src, int ring_id) {
  VLOG(3) << "/// DEBUG /// start inter broadcast with ring_id: " << ring_id;
  phi::DenseTensor *src_tensor = src->GetMutable<phi::DenseTensor>();
  const auto &place = src_tensor->place();
  platform::XCCLComm *comm =
      platform::XCCLCommContext::Instance(place_.GetDeviceType())
          .Get(ring_id, place);
  auto stream = comm->stream();

  void *src_ptr = src_tensor->data();

  phi::DeviceManager::CCLBroadcast(place_.GetDeviceType(),
                                   src_ptr,
                                   src_tensor->numel(),
                                   src_tensor->dtype(),
                                   0,
                                   comm->comm(),
                                   *stream);
}

phi::DeviceContext *XCCLParallelContext::GetDeviceContext(int ring_id) {
  return static_cast<phi::DeviceContext *>(
      platform::XCCLCommContext::Instance(place_.GetDeviceType())
          .Get(ring_id, place_)
          ->dev_context());
}

void XCCLParallelContext::WaitCompute(int ring_id) {
  PADDLE_ENFORCE_GE(
      ring_id,
      0,
      common::errors::OutOfRange("ring id must >= 0, but got %d", ring_id));
  PADDLE_ENFORCE_LT(ring_id,
                    compute_events_.size(),
                    common::errors::OutOfRange(
                        "ring id must < compute events size,"
                        "but got ring id = %d, compute events size = %d",
                        ring_id,
                        compute_events_.size()));

  auto compute_stream = static_cast<phi::CustomContext *>(
                            phi::DeviceContextPool::Instance().Get(place_))
                            ->GetStream();
  auto comm_stream = platform::XCCLCommContext::Instance(place_.GetDeviceType())
                         .Get(ring_id, place_)
                         ->stream();
  auto event = compute_events_[ring_id].get();

  // compute_stream-->event-->comm_stream
  event->Record(compute_stream.get());
  comm_stream->WaitEvent(event);
}

void XCCLParallelContext::WaitComm(int ring_id) {
  PADDLE_ENFORCE_GE(
      ring_id,
      0,
      common::errors::OutOfRange("ring id must >= 0, but got %d", ring_id));
  PADDLE_ENFORCE_LT(
      ring_id,
      comm_events_.size(),
      common::errors::OutOfRange("ring id must < comm events size,"
                                 "but got ring id = %d, comm events size = %d",
                                 ring_id,
                                 comm_events_.size()));

  auto compute_stream = static_cast<phi::CustomContext *>(
                            phi::DeviceContextPool::Instance().Get(place_))
                            ->GetStream();
  auto comm_stream = platform::XCCLCommContext::Instance(place_.GetDeviceType())
                         .Get(ring_id, place_)
                         ->stream();
  auto event = comm_events_[ring_id].get();

  // comm_stream-->event-->compute_stream
  event->Record(comm_stream.get());
  compute_stream->WaitEvent(event);
}

void XCCLParallelContext::SynchronizeCompute() {
  auto *compute_dev_ctx = static_cast<phi::CustomContext *>(
      phi::DeviceContextPool::Instance().Get(place_));
  compute_dev_ctx->Wait();
}

}  //  namespace imperative
}  //  namespace paddle
