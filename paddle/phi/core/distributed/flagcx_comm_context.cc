// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/distributed/flagcx_comm_context.h"

#include "glog/logging.h"

#include "paddle/phi/core/dense_tensor.h"
// #include "paddle/phi/core/distributed/check/nccl_dynamic_check.h"
#include "paddle/phi/core/distributed/check/static_check.h"
#include "paddle/phi/core/distributed/flagcx_tools.h"
#include "paddle/phi/core/distributed/utils.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/utils/data_type.h"

#ifdef PADDLE_WITH_FLAGCX
#include <flagcx.h>
#endif

namespace phi::distributed {

// set this flag to `true` and recompile to enable dynamic checks
// constexpr bool FLAGS_enable_flagcx_dynamic_check = false;

FlagcxCommContext::FlagcxCommContext(int rank,
                                     int size,
                                     flagcxHandlerGroup_t flagcx_handler)
    : CommContext(rank, size),
      flagcx_version_(0),
      flagcx_handler_(flagcx_handler) {
  phi::dynload::flagcxCommInitRank(
      &flagcx_handler_->comm, size_, flagcx_handler_->uniqueId, rank_),
      phi::dynload::flagcxGetVersion(&flagcx_version_);
}

int FlagcxCommContext::GetFlagcxVersion() { return flagcx_version_; }

flagcxComm_t FlagcxCommContext::GetFlagcxComm() {
  return flagcx_handler_->comm;
}

void FlagcxCommContext::Broadcast(phi::DenseTensor* out_tensor,
                                  const phi::DenseTensor& in_tensor,
                                  int root,
                                  flagcxStream_t stream) {
  CommStaticCheck::SameShape(*out_tensor,
                             in_tensor,
                             /*dst_rank*/ rank_,
                             /*cur_rank*/ rank_,
                             size_);
  FLAGCX_CHECK(phi::dynload::flagcxBroadcast(in_tensor.data(),
                                             out_tensor->data(),
                                             in_tensor.numel(),
                                             ToFlagcxDataType(in_tensor.type()),
                                             root,
                                             flagcx_handler_->comm,
                                             stream));
}

void FlagcxCommContext::AllGather(phi::DenseTensor* out_tensor,
                                  const phi::DenseTensor& in_tensor,
                                  flagcxStream_t stream) {
  phi::distributed::CommStaticCheck::GatherLikeShape(*out_tensor,
                                                     in_tensor,
                                                     /*dst_rank*/ rank_,
                                                     /*cur_rank*/ rank_,
                                                     size_);
  FLAGCX_CHECK(phi::dynload::flagcxAllGather(in_tensor.data(),
                                             out_tensor->data(),
                                             in_tensor.numel(),
                                             ToFlagcxDataType(in_tensor.type()),
                                             flagcx_handler_->comm,
                                             stream));
}
void FlagcxCommContext::ReduceScatter(phi::DenseTensor* out_tensor,
                                      const phi::DenseTensor& in_tensor,
                                      flagcxRedOp_t reduce_type,
                                      flagcxStream_t stream) {
  phi::distributed::CommStaticCheck::ScatterLikeShape(*out_tensor,
                                                      in_tensor,
                                                      /*dst_rank*/ rank_,
                                                      /*cur_rank*/ rank_,
                                                      size_);
  FLAGCX_CHECK(
      phi::dynload::flagcxReduceScatter(in_tensor.data(),
                                        out_tensor->data(),
                                        out_tensor->numel(),
                                        ToFlagcxDataType(in_tensor.type()),
                                        reduce_type,
                                        flagcx_handler_->comm,
                                        stream));
}

void FlagcxCommContext::Send(const phi::DenseTensor& in_tensor,
                             const int64_t& count,
                             const int& peer,
                             flagcxStream_t stream) {
  phi::distributed::CommStaticCheck::CheckShape(in_tensor, rank_, size_);

  FLAGCX_CHECK(phi::dynload::flagcxSend(in_tensor.data(),
                                        count,
                                        ToFlagcxDataType(in_tensor.dtype()),
                                        peer,
                                        flagcx_handler_->comm,
                                        stream));
  VLOG(3) << "rank " << GetRank() << " send "
          << common::product(in_tensor.dims()) << " to " << peer;
}

void FlagcxCommContext::Recv(phi::DenseTensor* out_tensor,
                             const int64_t& count,
                             const int& peer,
                             flagcxStream_t stream) {
  phi::distributed::CommStaticCheck::CheckShape(*out_tensor, rank_, size_);

  FLAGCX_CHECK(phi::dynload::flagcxRecv(out_tensor->data(),
                                        count,
                                        ToFlagcxDataType(out_tensor->dtype()),
                                        peer,
                                        flagcx_handler_->comm,
                                        stream));
  VLOG(3) << "rank " << GetRank() << " recv "
          << common::product(out_tensor->dims()) << " from " << peer;
}

void FlagcxCommContext::AllReduce(phi::DenseTensor* out_tensor,
                                  const phi::DenseTensor& in_tensor,
                                  flagcxRedOp_t reduce_type,
                                  flagcxStream_t stream) {
  phi::distributed::CommStaticCheck::SameShape(*out_tensor,
                                               in_tensor,
                                               /*dst_rank*/ rank_,
                                               /*cur_rank*/ rank_,
                                               size_);
  FLAGCX_CHECK(phi::dynload::flagcxAllReduce(in_tensor.data(),
                                             out_tensor->data(),
                                             in_tensor.numel(),
                                             ToFlagcxDataType(in_tensor.type()),
                                             reduce_type,
                                             flagcx_handler_->comm,
                                             stream));
}

void FlagcxCommContext::Reduce(phi::DenseTensor* out_tensor,
                               const phi::DenseTensor& in_tensor,
                               flagcxRedOp_t reduce_type,
                               int root,
                               flagcxStream_t stream) {
  phi::distributed::CommStaticCheck::SameShape(*out_tensor,
                                               in_tensor,
                                               /*dst_rank*/ root,
                                               /*cur_rank*/ rank_,
                                               size_);
  FLAGCX_CHECK(phi::dynload::flagcxReduce(in_tensor.data(),
                                          out_tensor->data(),
                                          in_tensor.numel(),
                                          ToFlagcxDataType(in_tensor.type()),
                                          reduce_type,
                                          root,
                                          flagcx_handler_->comm,
                                          stream));
}

void FlagcxCommContext::GroupStart() {
  FLAGCX_CHECK(phi::dynload::flagcxGroupStart(flagcx_handler_->comm));
}
void FlagcxCommContext::GroupEnd() {
  FLAGCX_CHECK(phi::dynload::flagcxGroupEnd(flagcx_handler_->comm));
}

}  // namespace phi::distributed
