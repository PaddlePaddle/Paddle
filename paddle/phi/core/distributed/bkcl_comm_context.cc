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
  PADDLE_ENFORCE_XPU_SUCCESS(
      bkcl_init_rank(&bkcl_comm_, rank_, size_, &bkcl_id));
}

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
  PADDLE_ENFORCE_XPU_SUCCESS(bkcl_broadcast(bkcl_comm_,
                                            in_tensor.data(),
                                            out_tensor->data(),
                                            in_tensor.numel(),
                                            ToBKCLDataType(in_tensor.type()),
                                            root,
                                            stream));
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
  PADDLE_ENFORCE_XPU_SUCCESS(bkcl_all_gather(bkcl_comm_,
                                             in_tensor.data(),
                                             in_tensor.numel(),
                                             out_tensor->data(),
                                             ToBKCLDataType(in_tensor.type()),
                                             stream));
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
  PADDLE_ENFORCE_XPU_SUCCESS(
      bkcl_reduce_scatter(bkcl_comm_,
                          in_tensor.data(),
                          out_tensor->data(),
                          out_tensor->numel(),
                          ToBKCLDataType(in_tensor.type()),
                          reduce_type,
                          stream));
}

void BKCLCommContext::Send(const phi::DenseTensor& in_tensor,
                           const int64_t& count,
                           const int& peer,
                           XPUStream stream) {
  phi::distributed::CommStaticCheck::CheckShape(
      in_tensor, rank_, size_, phi::AllocationType::XPU);

  PADDLE_ENFORCE_XPU_SUCCESS(bkcl_send(bkcl_comm_,
                                       in_tensor.data(),
                                       count,
                                       peer,
                                       ToBKCLDataType(in_tensor.dtype()),
                                       stream));
  VLOG(3) << "rank " << GetRank() << " send " << phi::product(in_tensor.dims())
          << " to " << peer;
}

void BKCLCommContext::Recv(phi::DenseTensor* out_tensor,
                           const int64_t& count,
                           const int& peer,
                           XPUStream stream) {
  phi::distributed::CommStaticCheck::CheckShape(
      *out_tensor, rank_, size_, phi::AllocationType::XPU);

  PADDLE_ENFORCE_XPU_SUCCESS(bkcl_recv(bkcl_comm_,
                                       out_tensor->data(),
                                       count,
                                       peer,
                                       ToBKCLDataType(out_tensor->dtype()),
                                       stream));
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
  PADDLE_ENFORCE_XPU_SUCCESS(bkcl_all_reduce(bkcl_comm_,
                                             in_tensor.data(),
                                             out_tensor->data(),
                                             in_tensor.numel(),
                                             ToBKCLDataType(in_tensor.type()),
                                             reduce_type,
                                             stream));
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
  PADDLE_ENFORCE_XPU_SUCCESS(bkcl_all_to_all(bkcl_comm_,
                                             in_tensor.data(),
                                             in_tensor.numel() / size_,
                                             out_tensor->data(),
                                             ToBKCLDataType(in_tensor.type()),
                                             stream));
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
  PADDLE_ENFORCE_XPU_SUCCESS(bkcl_reduce(bkcl_comm_,
                                         in_tensor.data(),
                                         out_tensor->data(),
                                         in_tensor.numel(),
                                         ToBKCLDataType(in_tensor.type()),
                                         reduce_type,
                                         root,
                                         stream));
}

void BKCLCommContext::GroupStart() {
  PADDLE_ENFORCE_XPU_SUCCESS(bkcl_group_start());
}
void BKCLCommContext::GroupEnd() {
  PADDLE_ENFORCE_XPU_SUCCESS(bkcl_group_end());
}
}  // namespace distributed
}  // namespace phi
