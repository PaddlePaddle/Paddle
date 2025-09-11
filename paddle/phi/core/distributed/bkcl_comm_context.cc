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

#include "paddle/phi/core/distributed/bkcl_comm_context.h"

#include "glog/logging.h"

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/distributed/check/static_check.h"
#include "paddle/phi/core/distributed/utils.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/utils/data_type.h"

namespace phi {
namespace distributed {

BKCLCommContext::BKCLCommContext(int rank, int size, BKCLUniqueId bkcl_id)
    : CommContext(rank, size) {
  PADDLE_ENFORCE_BKCL_SUCCESS(
      bkcl_init_rank(&bkcl_comm_, rank_, size_, &bkcl_id));
}

#if defined(PADDLE_WITH_FLAGCX)
BKCLCommContext::BKCLCommContext(int rank,
                                 int size,
                                 flagcxHandlerGroup_t flagcx_handler)
    : CommContext(rank, size), flagcx_handler_(flagcx_handler) {
  phi::dynload::flagcxCommInitRank(
      &flagcx_handler_->comm, size_, flagcx_handler_->uniqueId, rank_);
}
#endif

BKCLContext_t BKCLCommContext::GetBKCLComm() { return bkcl_comm_; }

XPUStream BKCLCommContext::GetStream() { return dev_ctx_->stream(); }

phi::XPUContext* BKCLCommContext::GetDevContext() { return dev_ctx_.get(); }

void BKCLCommContext::SetDevContext(
    std::unique_ptr<phi::XPUContext>&& dev_ctx) {
  dev_ctx_ = std::move(dev_ctx);
}

XPUEvent BKCLCommContext::GetComputeEvent() { return compute_event_.get(); }

void BKCLCommContext::SetComputeEvent(
    std::shared_ptr<std::remove_pointer<XPUEvent>::type>&& compute_event) {
  compute_event_ = std::move(compute_event);
}

XPUEvent BKCLCommContext::GetCommEvent() { return comm_event_.get(); }

void BKCLCommContext::SetCommEvent(
    std::shared_ptr<std::remove_pointer<XPUEvent>::type>&& comm_event) {
  comm_event_ = std::move(comm_event);
}

void BKCLCommContext::Broadcast(phi::DenseTensor* out_tensor,
                                const phi::DenseTensor& in_tensor,
                                int root,
                                XPUStream stream) {
  CommStaticCheck::SameShape(*out_tensor,
                             in_tensor,
                             /*dst_rank*/ rank_,
                             /*cur_rank*/ rank_,
                             size_,
                             phi::AllocationType::XPU);
#if defined(PADDLE_WITH_FLAGCX)
  FLAGCX_CHECK(
      phi::dynload::flagcxBroadcast(in_tensor.data(),
                                    out_tensor->data(),
                                    in_tensor.numel(),
                                    ToFlagcxDataType(in_tensor.type()),
                                    root,
                                    flagcx_handler_->comm,
                                    reinterpret_cast<flagcxStream_t>(&stream)));
#else
  PADDLE_ENFORCE_BKCL_SUCCESS(bkcl_broadcast(bkcl_comm_,
                                             in_tensor.data(),
                                             out_tensor->data(),
                                             in_tensor.numel(),
                                             ToBKCLDataType(in_tensor.type()),
                                             root,
                                             stream));
#endif
}

void BKCLCommContext::AllGather(phi::DenseTensor* out_tensor,
                                const phi::DenseTensor& in_tensor,
                                XPUStream stream) {
  phi::distributed::CommStaticCheck::GatherLikeShape(*out_tensor,
                                                     in_tensor,
                                                     /*dst_rank*/ rank_,
                                                     /*cur_rank*/ rank_,
                                                     size_,
                                                     phi::AllocationType::XPU);
#if defined(PADDLE_WITH_FLAGCX)
  FLAGCX_CHECK(
      phi::dynload::flagcxAllGather(in_tensor.data(),
                                    out_tensor->data(),
                                    in_tensor.numel(),
                                    ToFlagcxDataType(in_tensor.type()),
                                    flagcx_handler_->comm,
                                    reinterpret_cast<flagcxStream_t>(&stream)));
#else
  PADDLE_ENFORCE_BKCL_SUCCESS(bkcl_all_gather(bkcl_comm_,
                                              in_tensor.data(),
                                              in_tensor.numel(),
                                              out_tensor->data(),
                                              ToBKCLDataType(in_tensor.type()),
                                              stream));
#endif
}

void BKCLCommContext::ReduceScatter(phi::DenseTensor* out_tensor,
                                    const phi::DenseTensor& in_tensor,
                                    BKCLOp reduce_type,
                                    XPUStream stream) {
  phi::distributed::CommStaticCheck::ScatterLikeShape(*out_tensor,
                                                      in_tensor,
                                                      /*dst_rank*/ rank_,
                                                      /*cur_rank*/ rank_,
                                                      size_,
                                                      phi::AllocationType::XPU);
#if defined(PADDLE_WITH_FLAGCX)
  FLAGCX_CHECK(phi::dynload::flagcxReduceScatter(
      in_tensor.data(),
      out_tensor->data(),
      out_tensor->numel(),
      ToFlagcxDataType(in_tensor.type()),
      BkclToFlagcxRedType(reduce_type),
      flagcx_handler_->comm,
      reinterpret_cast<flagcxStream_t>(&stream)));
#else
  PADDLE_ENFORCE_BKCL_SUCCESS(
      bkcl_reduce_scatter(bkcl_comm_,
                          in_tensor.data(),
                          out_tensor->data(),
                          out_tensor->numel(),
                          ToBKCLDataType(in_tensor.type()),
                          reduce_type,
                          stream));
#endif
}

#if defined(PADDLE_WITH_FLAGCX)
void BKCLCommContext::Scatter(phi::DenseTensor* out_tensor,
                              const phi::DenseTensor& in_tensor,
                              int root,
                              XPUStream stream) {
  phi::distributed::CommStaticCheck::ScatterLikeShape(*out_tensor,
                                                      in_tensor,
                                                      /*dst_rank*/ rank_,
                                                      /*cur_rank*/ rank_,
                                                      size_,
                                                      phi::AllocationType::XPU);

  FLAGCX_CHECK(
      phi::dynload::flagcxScatter(in_tensor.data(),
                                  out_tensor->data(),
                                  out_tensor->numel(),
                                  ToFlagcxDataType(in_tensor.type()),
                                  root,
                                  flagcx_handler_->comm,
                                  reinterpret_cast<flagcxStream_t>(&stream)));
}
#endif

void BKCLCommContext::Send(const phi::DenseTensor& in_tensor,
                           const int64_t& count,
                           const int& peer,
                           XPUStream stream) {
  phi::distributed::CommStaticCheck::CheckShape(
      in_tensor, rank_, size_, phi::AllocationType::XPU);

#if defined(PADDLE_WITH_FLAGCX)
  FLAGCX_CHECK(
      phi::dynload::flagcxSend(in_tensor.data(),
                               count,
                               ToFlagcxDataType(in_tensor.dtype()),
                               peer,
                               flagcx_handler_->comm,
                               reinterpret_cast<flagcxStream_t>(&stream)));
#else

  PADDLE_ENFORCE_BKCL_SUCCESS(bkcl_send(bkcl_comm_,
                                        in_tensor.data(),
                                        count,
                                        peer,
                                        ToBKCLDataType(in_tensor.dtype()),
                                        stream));
#endif
  VLOG(3) << "rank " << GetRank() << " send " << phi::product(in_tensor.dims())
          << " to " << peer;
}

void BKCLCommContext::Recv(phi::DenseTensor* out_tensor,
                           const int64_t& count,
                           const int& peer,
                           XPUStream stream) {
  phi::distributed::CommStaticCheck::CheckShape(
      *out_tensor, rank_, size_, phi::AllocationType::XPU);
#if defined(PADDLE_WITH_FLAGCX)
  FLAGCX_CHECK(
      phi::dynload::flagcxRecv(out_tensor->data(),
                               count,
                               ToFlagcxDataType(out_tensor->dtype()),
                               peer,
                               flagcx_handler_->comm,
                               reinterpret_cast<flagcxStream_t>(&stream)));
#else

  PADDLE_ENFORCE_BKCL_SUCCESS(bkcl_recv(bkcl_comm_,
                                        out_tensor->data(),
                                        count,
                                        peer,
                                        ToBKCLDataType(out_tensor->dtype()),
                                        stream));
#endif
  VLOG(3) << "rank " << GetRank() << " recv "
          << common::product(out_tensor->dims()) << " from " << peer;
}

void BKCLCommContext::AllReduce(phi::DenseTensor* out_tensor,
                                const phi::DenseTensor& in_tensor,
                                BKCLOp reduce_type,
                                XPUStream stream) {
  phi::distributed::CommStaticCheck::SameShape(*out_tensor,
                                               in_tensor,
                                               /*dst_rank*/ rank_,
                                               /*cur_rank*/ rank_,
                                               size_,
                                               phi::AllocationType::XPU);

#if defined(PADDLE_WITH_FLAGCX)
  FLAGCX_CHECK(
      phi::dynload::flagcxAllReduce(in_tensor.data(),
                                    out_tensor->data(),
                                    in_tensor.numel(),
                                    ToFlagcxDataType(in_tensor.type()),
                                    BkclToFlagcxRedType(reduce_type),
                                    flagcx_handler_->comm,
                                    reinterpret_cast<flagcxStream_t>(&stream)));
#else
  PADDLE_ENFORCE_BKCL_SUCCESS(bkcl_all_reduce(bkcl_comm_,
                                              in_tensor.data(),
                                              out_tensor->data(),
                                              in_tensor.numel(),
                                              ToBKCLDataType(in_tensor.type()),
                                              reduce_type,
                                              stream));
#endif
}

void BKCLCommContext::AllToAll(phi::DenseTensor* out_tensor,
                               const phi::DenseTensor& in_tensor,
                               XPUStream stream) {
  phi::distributed::CommStaticCheck::SameShape(*out_tensor,
                                               in_tensor,
                                               /*dst_rank*/ rank_,
                                               /*cur_rank*/ rank_,
                                               size_,
                                               phi::AllocationType::XPU);

#if defined(PADDLE_WITH_FLAGCX)
  FLAGCX_CHECK(
      phi::dynload::flagcxAlltoAll(in_tensor.data(),
                                   out_tensor->data(),
                                   in_tensor.numel() / size_,
                                   ToFlagcxDataType(in_tensor.type()),
                                   flagcx_handler_->comm,
                                   reinterpret_cast<flagcxStream_t>(&stream)));
#else
  PADDLE_ENFORCE_BKCL_SUCCESS(bkcl_all_to_all(bkcl_comm_,
                                              in_tensor.data(),
                                              in_tensor.numel() / size_,
                                              out_tensor->data(),
                                              ToBKCLDataType(in_tensor.type()),
                                              stream));
#endif
}

void BKCLCommContext::AllToAllUnequalSplit(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const phi::DenseTensor& out_size_tensor,
    const phi::DenseTensor& out_offset_tensor,
    const phi::DenseTensor& in_size_tensor,
    const phi::DenseTensor& in_offset_tensor,
    XPUStream stream) {
  auto in_size_ptr = reinterpret_cast<const size_t*>(in_size_tensor.data());
  auto in_offset_ptr = reinterpret_cast<const size_t*>(in_offset_tensor.data());
  auto out_size_ptr = reinterpret_cast<const size_t*>(out_size_tensor.data());
  auto out_offset_ptr =
      reinterpret_cast<const size_t*>(out_offset_tensor.data());

#if defined(PADDLE_WITH_FLAGCX)
  FLAGCX_CHECK(
      phi::dynload::flagcxAlltoAllv(in_tensor.data(),
                                    const_cast<size_t*>(in_size_ptr),
                                    const_cast<size_t*>(in_offset_ptr),
                                    out_tensor->data(),
                                    const_cast<size_t*>(out_size_ptr),
                                    const_cast<size_t*>(out_offset_ptr),
                                    ToFlagcxDataType(in_tensor.type()),
                                    flagcx_handler_->comm,
                                    reinterpret_cast<flagcxStream_t>(&stream)));
#else

  PADDLE_ENFORCE_BKCL_SUCCESS(
      bkcl_all_to_all_v(bkcl_comm_,
                        in_tensor.data(),
                        in_size_ptr,
                        in_offset_ptr,
                        ToBKCLDataType(in_tensor.type()),
                        out_tensor->data(),
                        out_size_ptr,
                        out_offset_ptr,
                        ToBKCLDataType(out_tensor->type()),
                        stream));
#endif
}

void BKCLCommContext::Reduce(phi::DenseTensor* out_tensor,
                             const phi::DenseTensor& in_tensor,
                             BKCLOp reduce_type,
                             int root,
                             XPUStream stream) {
  phi::distributed::CommStaticCheck::SameShape(*out_tensor,
                                               in_tensor,
                                               /*dst_rank*/ root,
                                               /*cur_rank*/ rank_,
                                               size_,
                                               phi::AllocationType::XPU);

#if defined(PADDLE_WITH_FLAGCX)
  FLAGCX_CHECK(
      phi::dynload::flagcxReduce(in_tensor.data(),
                                 out_tensor->data(),
                                 in_tensor.numel(),
                                 ToFlagcxDataType(in_tensor.type()),
                                 BkclToFlagcxRedType(reduce_type),
                                 root,
                                 flagcx_handler_->comm,
                                 reinterpret_cast<flagcxStream_t>(&stream)));
#else
  PADDLE_ENFORCE_BKCL_SUCCESS(bkcl_reduce(bkcl_comm_,
                                          in_tensor.data(),
                                          out_tensor->data(),
                                          in_tensor.numel(),
                                          ToBKCLDataType(in_tensor.type()),
                                          reduce_type,
                                          root,
                                          stream));
#endif
}

void BKCLCommContext::GroupStart() {
#if defined(PADDLE_WITH_FLAGCX)
  FLAGCX_CHECK(phi::dynload::flagcxGroupStart(flagcx_handler_->comm));
#else
  PADDLE_ENFORCE_BKCL_SUCCESS(bkcl_group_start());
#endif
}
void BKCLCommContext::GroupEnd() {
#if defined(PADDLE_WITH_FLAGCX)
  FLAGCX_CHECK(phi::dynload::flagcxGroupEnd(flagcx_handler_->comm));
#else
  PADDLE_ENFORCE_BKCL_SUCCESS(bkcl_group_end());
#endif
}

#if defined(PADDLE_WITH_FLAGCX)
flagcxRedOp_t BKCLCommContext::BkclToFlagcxRedType(BKCLOp redOp) {
  switch (redOp) {
    case BKCL_MIN:
      return flagcxMin;
    case BKCL_MAX:
      return flagcxMax;
    case BKCL_ADD:
      return flagcxSum;
  }
}
#endif
}  // namespace distributed
}  // namespace phi
