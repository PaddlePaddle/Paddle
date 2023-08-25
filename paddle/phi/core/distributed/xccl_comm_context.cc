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

#include "glog/logging.h"

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/distributed/check/static_check.h"
#include "paddle/phi/core/distributed/utils.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/utils/data_type.h"

namespace phi {
namespace distributed {

XCCLCommContext::XCCLCommContext(const std::string& device_type,
                                 int rank,
                                 int size,
                                 const ccl::CCLRootId& xccl_id)
    : CommContext(rank, size) {
  device_type_ = device_type;
  phi::DeviceManager::CCLCommInitRank(device_type,
                                      size_,
                                      const_cast<ccl::CCLRootId*>(&xccl_id),
                                      rank,
                                      &xccl_comm_);
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
    phi::DeviceManager::CCLBroadcast(device_type_,
                                     const_cast<void*>(in_tensor.data()),
                                     in_tensor.numel(),
                                     phi::ccl::ToCCLDataType(in_tensor.dtype()),
                                     root,
                                     xccl_comm_,
                                     stream);
  } else {
    phi::DeviceManager::CCLBroadcast(device_type_,
                                     out_tensor->data(),
                                     out_tensor->numel(),
                                     phi::ccl::ToCCLDataType(in_tensor.dtype()),
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
  phi::DeviceManager::CCLAllGather(device_type_,
                                   const_cast<void*>(in_tensor.data()),
                                   out_tensor->data(),
                                   in_tensor.numel(),
                                   phi::ccl::ToCCLDataType(in_tensor.dtype()),
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
  phi::DeviceManager::CCLReduceScatter(
      device_type_,
      const_cast<void*>(in_tensor.data()),
      out_tensor->data(),
      out_tensor->numel(),
      phi::ccl::ToCCLDataType(in_tensor.type()),
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
  phi::DeviceManager::CCLSend(device_type_,
                              const_cast<void*>(in_tensor.data()),
                              count,
                              phi::ccl::ToCCLDataType(in_tensor.type()),
                              peer,
                              xccl_comm_,
                              stream);
  VLOG(3) << "rank " << GetRank() << " send " << phi::product(in_tensor.dims())
          << " to " << peer;
}

void XCCLCommContext::Recv(phi::DenseTensor* out_tensor,
                           const int64_t& count,
                           const int& peer,
                           const phi::stream::Stream& stream) const {
  phi::distributed::CommStaticCheck::CheckShape(
      *out_tensor, rank_, size_, phi::AllocationType::CUSTOM);
  phi::DeviceManager::CCLRecv(device_type_,
                              out_tensor->data(),
                              count,
                              phi::ccl::ToCCLDataType(out_tensor->type()),
                              peer,
                              xccl_comm_,
                              stream);
  VLOG(3) << "rank " << GetRank() << " recv "
          << phi::product(out_tensor->dims()) << " from " << peer;
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
  phi::DeviceManager::CCLAllReduce(device_type_,
                                   const_cast<void*>(in_tensor.data()),
                                   out_tensor->data(),
                                   in_tensor.numel(),
                                   phi::ccl::ToCCLDataType(in_tensor.type()),
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
  phi::DeviceManager::CCLReduce(device_type_,
                                const_cast<void*>(in_tensor.data()),
                                out_tensor->data(),
                                in_tensor.numel(),
                                phi::ccl::ToCCLDataType(in_tensor.type()),
                                reduce_type,
                                root,
                                xccl_comm_,
                                stream);
}

void XCCLCommContext::GroupStart() const {
  phi::DeviceManager::CCLGroupStart(device_type_);
}
void XCCLCommContext::GroupEnd() const {
  phi::DeviceManager::CCLGroupEnd(device_type_);
}

}  // namespace distributed
}  // namespace phi
