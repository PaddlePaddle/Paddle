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

#include "paddle/phi/core/distributed/nccl_comm_context.h"

#include "glog/logging.h"

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/distributed/check/nccl_dynamic_check.h"
#include "paddle/phi/core/distributed/check/static_check.h"
#include "paddle/phi/core/distributed/utils.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/utils/data_type.h"

namespace phi {
namespace distributed {

// set this flag to `true` and recompile to enable dynamic checks
constexpr bool FLAGS_enable_nccl_dynamic_check = false;

NCCLCommContext::NCCLCommContext(int rank, int size, ncclUniqueId nccl_id)
    : CommContext(rank, size) {
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::ncclCommInitRank(&nccl_comm_, size_, nccl_id, rank_));
}

ncclComm_t NCCLCommContext::GetNcclComm() { return nccl_comm_; }

gpuStream_t NCCLCommContext::GetStream() { return dev_ctx_->stream(); }

phi::GPUContext* NCCLCommContext::GetDevContext() { return dev_ctx_.get(); }

void NCCLCommContext::SetDevContext(
    std::unique_ptr<phi::GPUContext>&& dev_ctx) {
  dev_ctx_ = std::move(dev_ctx);
}

gpuEvent_t NCCLCommContext::GetComputeEvent() { return compute_event_.get(); }

void NCCLCommContext::SetComputeEvent(
    std::shared_ptr<std::remove_pointer<phi::gpuEvent_t>::type>&&
        compute_event) {
  compute_event_ = std::move(compute_event);
}

gpuEvent_t NCCLCommContext::GetCommEvent() { return comm_event_.get(); }

void NCCLCommContext::SetCommEvent(
    std::shared_ptr<std::remove_pointer<phi::gpuEvent_t>::type>&& comm_event) {
  comm_event_ = std::move(comm_event);
}

void NCCLCommContext::Broadcast(phi::DenseTensor* out_tensor,
                                const phi::DenseTensor& in_tensor,
                                int root,
                                gpuStream_t stream) {
  CommStaticCheck::SameShape(*out_tensor,
                             in_tensor,
                             /*dst_rank*/ rank_,
                             /*cur_rank*/ rank_,
                             size_);
  if (FLAGS_enable_nccl_dynamic_check) {
    NCCLDynamicCheck::CheckShape(*out_tensor, root, rank_, nccl_comm_);
  }
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::ncclBroadcast(in_tensor.data(),
                                  out_tensor->data(),
                                  in_tensor.numel(),
                                  ToNCCLDataType(in_tensor.type()),
                                  root,
                                  nccl_comm_,
                                  stream));
}

void NCCLCommContext::AllGather(phi::DenseTensor* out_tensor,
                                const phi::DenseTensor& in_tensor,
                                gpuStream_t stream) {
  phi::distributed::CommStaticCheck::GatherLikeShape(*out_tensor,
                                                     in_tensor,
                                                     /*dst_rank*/ rank_,
                                                     /*cur_rank*/ rank_,
                                                     size_);
  if (FLAGS_enable_nccl_dynamic_check) {
    phi::distributed::NCCLDynamicCheck::CheckShape(*out_tensor,
                                                   /*root_rank*/ 0,
                                                   rank_,
                                                   nccl_comm_);
  }
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::ncclAllGather(in_tensor.data(),
                                  out_tensor->data(),
                                  in_tensor.numel(),
                                  ToNCCLDataType(in_tensor.type()),
                                  nccl_comm_,
                                  stream));
}
void NCCLCommContext::ReduceScatter(phi::DenseTensor* out_tensor,
                                    const phi::DenseTensor& in_tensor,
                                    ncclRedOp_t reduce_type,
                                    gpuStream_t stream) {
  phi::distributed::CommStaticCheck::ScatterLikeShape(*out_tensor,
                                                      in_tensor,
                                                      /*dst_rank*/ rank_,
                                                      /*cur_rank*/ rank_,
                                                      size_);
  if (FLAGS_enable_nccl_dynamic_check) {
    phi::distributed::NCCLDynamicCheck::CheckShape(*out_tensor,
                                                   /*root_rank*/ 0,
                                                   rank_,
                                                   nccl_comm_);
  }
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::ncclReduceScatter(in_tensor.data(),
                                      out_tensor->data(),
                                      out_tensor->numel(),
                                      ToNCCLDataType(in_tensor.type()),
                                      reduce_type,
                                      nccl_comm_,
                                      stream));
}

void NCCLCommContext::Send(const phi::DenseTensor& in_tensor,
                           const int64_t& count,
                           const int& peer,
                           gpuStream_t stream) {
  phi::distributed::CommStaticCheck::CheckShape(in_tensor, rank_, size_);

  if (FLAGS_enable_nccl_dynamic_check) {
    NCCLDynamicCheck::CheckShape(in_tensor, rank_, rank_, nccl_comm_);
  }

  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::ncclSend(in_tensor.data(),
                             count,
                             ToNCCLDataType(in_tensor.dtype()),
                             peer,
                             nccl_comm_,
                             stream));
  VLOG(3) << "rank " << GetRank() << " send " << phi::product(in_tensor.dims())
          << " to " << peer;
}

void NCCLCommContext::Recv(phi::DenseTensor* out_tensor,
                           const int64_t& count,
                           const int& peer,
                           gpuStream_t stream) {
  phi::distributed::CommStaticCheck::CheckShape(*out_tensor, rank_, size_);
  if (FLAGS_enable_nccl_dynamic_check) {
    NCCLDynamicCheck::CheckShape(*out_tensor, peer, rank_, nccl_comm_);
  }

  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::ncclRecv(out_tensor->data(),
                             count,
                             ToNCCLDataType(out_tensor->dtype()),
                             peer,
                             nccl_comm_,
                             stream));
  VLOG(3) << "rank " << GetRank() << " recv "
          << phi::product(out_tensor->dims()) << " from " << peer;
}

void NCCLCommContext::AllReduce(phi::DenseTensor* out_tensor,
                                const phi::DenseTensor& in_tensor,
                                ncclRedOp_t reduce_type,
                                gpuStream_t stream) {
  phi::distributed::CommStaticCheck::SameShape(*out_tensor,
                                               in_tensor,
                                               /*dst_rank*/ rank_,
                                               /*cur_rank*/ rank_,
                                               size_);
  if (FLAGS_enable_nccl_dynamic_check) {
    phi::distributed::NCCLDynamicCheck::CheckShape(*out_tensor,
                                                   /*root_rank*/ 0,
                                                   rank_,
                                                   nccl_comm_);
  }
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::ncclAllReduce(in_tensor.data(),
                                  out_tensor->data(),
                                  in_tensor.numel(),
                                  ToNCCLDataType(in_tensor.type()),
                                  reduce_type,
                                  nccl_comm_,
                                  stream));
}

void NCCLCommContext::Reduce(phi::DenseTensor* out_tensor,
                             const phi::DenseTensor& in_tensor,
                             ncclRedOp_t reduce_type,
                             int root,
                             gpuStream_t stream) {
  phi::distributed::CommStaticCheck::SameShape(*out_tensor,
                                               in_tensor,
                                               /*dst_rank*/ root,
                                               /*cur_rank*/ rank_,
                                               size_);
  if (FLAGS_enable_nccl_dynamic_check) {
    phi::distributed::NCCLDynamicCheck::CheckShape(*out_tensor,
                                                   /*root_rank*/ root,
                                                   rank_,
                                                   nccl_comm_);
  }
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::ncclReduce(in_tensor.data(),
                               out_tensor->data(),
                               in_tensor.numel(),
                               ToNCCLDataType(in_tensor.type()),
                               reduce_type,
                               root,
                               nccl_comm_,
                               stream));
}

void NCCLCommContext::GroupStart() {
  NCCL_CHECK(phi::dynload::ncclGroupStart());
}
void NCCLCommContext::GroupEnd() { NCCL_CHECK(phi::dynload::ncclGroupEnd()); }

}  // namespace distributed
}  // namespace phi
