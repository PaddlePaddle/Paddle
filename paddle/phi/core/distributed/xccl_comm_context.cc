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

#include "paddle/phi/core/distributed/xccl_comm_context.h"

#include <list>

#include "glog/logging.h"

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/distributed/check/static_check.h"
#include "paddle/phi/core/distributed/utils.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/utils/data_type.h"

namespace phi {
namespace distributed {

std::list<XCCLCommContext*> g_xccl_comm_contexts;
std::mutex g_xccl_comm_contexts_mutex;

void XCCLCommContext::ReleaseAll() {
  std::unique_lock lock(g_xccl_comm_contexts_mutex);
  for (auto xccl_comm_ctx : g_xccl_comm_contexts) {
    phi::DeviceManager::CCLDestroyComm(xccl_comm_ctx->GetDeviceType(),
                                       xccl_comm_ctx->GetXcclComm());
    xccl_comm_ctx->xccl_comm_ = nullptr;
  }
  g_xccl_comm_contexts.clear();
}

XCCLCommContext::~XCCLCommContext() {
  std::unique_lock lock(g_xccl_comm_contexts_mutex);
  if (phi::DeviceManager::HasDeviceType(this->GetDeviceType()) &&
      xccl_comm_ != nullptr) {
    phi::DeviceManager::CCLDestroyComm(this->GetDeviceType(), xccl_comm_);
    xccl_comm_ = nullptr;
  }
  g_xccl_comm_contexts.remove(this);
}

XCCLCommContext::XCCLCommContext(const phi::Place& place,
                                 int rank,
                                 int size,
                                 const ccl::CCLRootId& xccl_id)
    : CommContext(rank, size) {
  place_ = place;
  phi::DeviceManager::CCLCommInitRank(place_.GetDeviceType(),
                                      size_,
                                      const_cast<ccl::CCLRootId*>(&xccl_id),
                                      rank,
                                      &xccl_comm_);
  stream_ = std::make_shared<phi::stream::Stream>();
  stream_->Init(place_);
  std::unique_lock lock(g_xccl_comm_contexts_mutex);
  g_xccl_comm_contexts.push_back(this);
}

void XCCLCommContext::Broadcast(phi::DenseTensor* out_tensor,
                                const phi::DenseTensor& in_tensor,
                                int root,
                                const phi::stream::Stream& stream) const {
  CommStaticCheck::SameShape(*out_tensor,
                             in_tensor,
                             /*dst_rank*/ rank_,
                             /*cur_rank*/ rank_,
                             size_,
                             phi::AllocationType::CUSTOM);
  if (rank_ == root) {
    phi::DeviceManager::CCLBroadcast(place_.GetDeviceType(),
                                     const_cast<void*>(in_tensor.data()),
                                     in_tensor.numel(),
                                     in_tensor.dtype(),
                                     root,
                                     xccl_comm_,
                                     stream);
  } else {
    phi::DeviceManager::CCLBroadcast(place_.GetDeviceType(),
                                     out_tensor->data(),
                                     out_tensor->numel(),
                                     in_tensor.dtype(),
                                     root,
                                     xccl_comm_,
                                     stream);
  }
}

void XCCLCommContext::AllGather(phi::DenseTensor* out_tensor,
                                const phi::DenseTensor& in_tensor,
                                const phi::stream::Stream& stream) const {
  phi::distributed::CommStaticCheck::GatherLikeShape(
      *out_tensor,
      in_tensor,
      /*dst_rank*/ rank_,
      /*cur_rank*/ rank_,
      size_,
      phi::AllocationType::CUSTOM);
  phi::DeviceManager::CCLAllGather(place_.GetDeviceType(),
                                   const_cast<void*>(in_tensor.data()),
                                   out_tensor->data(),
                                   in_tensor.numel(),
                                   in_tensor.dtype(),
                                   xccl_comm_,
                                   stream);
}
void XCCLCommContext::ReduceScatter(phi::DenseTensor* out_tensor,
                                    const phi::DenseTensor& in_tensor,
                                    phi::ccl::CCLReduceOp reduce_type,
                                    const phi::stream::Stream& stream) const {
  phi::distributed::CommStaticCheck::ScatterLikeShape(
      *out_tensor,
      in_tensor,
      /*dst_rank*/ rank_,
      /*cur_rank*/ rank_,
      size_,
      phi::AllocationType::CUSTOM);
  phi::DeviceManager::CCLReduceScatter(place_.GetDeviceType(),
                                       const_cast<void*>(in_tensor.data()),
                                       out_tensor->data(),
                                       out_tensor->numel(),
                                       in_tensor.dtype(),
                                       reduce_type,
                                       xccl_comm_,
                                       stream);
}

void XCCLCommContext::Send(const phi::DenseTensor& in_tensor,
                           const int64_t& count,
                           const int& peer,
                           const phi::stream::Stream& stream) const {
  phi::distributed::CommStaticCheck::CheckShape(
      in_tensor, rank_, size_, phi::AllocationType::CUSTOM);
  phi::DeviceManager::CCLSend(place_.GetDeviceType(),
                              const_cast<void*>(in_tensor.data()),
                              count,
                              in_tensor.dtype(),
                              peer,
                              xccl_comm_,
                              stream);
  VLOG(3) << "rank " << GetRank() << " send "
          << common::product(in_tensor.dims()) << " to " << peer;
}

void XCCLCommContext::Recv(phi::DenseTensor* out_tensor,
                           const int64_t& count,
                           const int& peer,
                           const phi::stream::Stream& stream) const {
  phi::distributed::CommStaticCheck::CheckShape(
      *out_tensor, rank_, size_, phi::AllocationType::CUSTOM);
  phi::DeviceManager::CCLRecv(place_.GetDeviceType(),
                              out_tensor->data(),
                              count,
                              out_tensor->dtype(),
                              peer,
                              xccl_comm_,
                              stream);
  VLOG(3) << "rank " << GetRank() << " recv "
          << common::product(out_tensor->dims()) << " from " << peer;
}

void XCCLCommContext::AllReduce(phi::DenseTensor* out_tensor,
                                const phi::DenseTensor& in_tensor,
                                phi::ccl::CCLReduceOp reduce_type,
                                const phi::stream::Stream& stream) const {
  phi::distributed::CommStaticCheck::SameShape(*out_tensor,
                                               in_tensor,
                                               /*dst_rank*/ rank_,
                                               /*cur_rank*/ rank_,
                                               size_,
                                               phi::AllocationType::CUSTOM);
  phi::DeviceManager::CCLAllReduce(place_.GetDeviceType(),
                                   const_cast<void*>(in_tensor.data()),
                                   out_tensor->data(),
                                   in_tensor.numel(),
                                   in_tensor.dtype(),
                                   reduce_type,
                                   xccl_comm_,
                                   stream);
}

void XCCLCommContext::Reduce(phi::DenseTensor* out_tensor,
                             const phi::DenseTensor& in_tensor,
                             phi::ccl::CCLReduceOp reduce_type,
                             int root,
                             const phi::stream::Stream& stream) const {
  phi::distributed::CommStaticCheck::SameShape(*out_tensor,
                                               in_tensor,
                                               /*dst_rank*/ root,
                                               /*cur_rank*/ rank_,
                                               size_,
                                               phi::AllocationType::CUSTOM);
  phi::DeviceManager::CCLReduce(place_.GetDeviceType(),
                                const_cast<void*>(in_tensor.data()),
                                out_tensor->data(),
                                in_tensor.numel(),
                                in_tensor.dtype(),
                                reduce_type,
                                root,
                                xccl_comm_,
                                stream);
}

void XCCLCommContext::GroupStart() const {
  phi::DeviceManager::CCLGroupStart(place_.GetDeviceType());
}
void XCCLCommContext::GroupEnd() const {
  phi::DeviceManager::CCLGroupEnd(place_.GetDeviceType());
}

}  // namespace distributed
}  // namespace phi
