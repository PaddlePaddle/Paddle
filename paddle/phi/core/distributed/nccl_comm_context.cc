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
#include "paddle/phi/core/distributed/nccl_tools.h"
#include "paddle/phi/core/distributed/utils.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/utils/data_type.h"

namespace phi::distributed {

// set this flag to `true` and recompile to enable dynamic checks
constexpr bool FLAGS_enable_nccl_dynamic_check = false;

NCCLCommContext::NCCLCommContext(int rank,
                                 int size,
                                 ncclUniqueId nccl_id,
                                 int nccl_comm_init_option)
    : CommContext(rank, size), nccl_version_(0), nccl_comm_(nullptr) {
  if (nccl_comm_init_option > 0 && phi::dynload::ncclCommInitRank2.IsValid()) {
    LOG(WARNING) << "Creating modified qp with ncclCommInitRank2.";
    NCCL_CHECK(phi::dynload::ncclCommInitRank2(
        &nccl_comm_, size_, nccl_id, rank_, nccl_comm_init_option));
  } else {
    if (nccl_comm_init_option > 0) {
      LOG(WARNING) << "ncclCommInitRank2 is not supported.";
    }
    NCCL_CHECK(
        phi::dynload::ncclCommInitRank(&nccl_comm_, size_, nccl_id, rank_));
  }
  NCCL_CHECK(phi::dynload::ncclGetVersion(&nccl_version_));
}

int NCCLCommContext::GetNcclVersion() { return nccl_version_; }

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
  NCCL_CHECK(phi::dynload::ncclBroadcast(in_tensor.data(),
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
  NCCL_CHECK(phi::dynload::ncclAllGather(in_tensor.data(),
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
  NCCL_CHECK(phi::dynload::ncclReduceScatter(in_tensor.data(),
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

  NCCL_CHECK(phi::dynload::ncclSend(in_tensor.data(),
                                    count,
                                    ToNCCLDataType(in_tensor.dtype()),
                                    peer,
                                    nccl_comm_,
                                    stream));
  VLOG(3) << "rank " << GetRank() << " send "
          << common::product(in_tensor.dims()) << " to " << peer;
}

void NCCLCommContext::Recv(phi::DenseTensor* out_tensor,
                           const int64_t& count,
                           const int& peer,
                           gpuStream_t stream) {
  phi::distributed::CommStaticCheck::CheckShape(*out_tensor, rank_, size_);
  if (FLAGS_enable_nccl_dynamic_check) {
    NCCLDynamicCheck::CheckShape(*out_tensor, peer, rank_, nccl_comm_);
  }

  NCCL_CHECK(phi::dynload::ncclRecv(out_tensor->data(),
                                    count,
                                    ToNCCLDataType(out_tensor->dtype()),
                                    peer,
                                    nccl_comm_,
                                    stream));
  VLOG(3) << "rank " << GetRank() << " recv "
          << common::product(out_tensor->dims()) << " from " << peer;
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
  NCCL_CHECK(phi::dynload::ncclAllReduce(in_tensor.data(),
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
  NCCL_CHECK(phi::dynload::ncclReduce(in_tensor.data(),
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

#if NCCL_VERSION_CODE >= 21100
void NCCLCommContext::RedOpCreatePreMulSum(ncclRedOp_t* op,
                                           void* scalar,
                                           ncclDataType_t dtype,
                                           ncclScalarResidence_t residence) {
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclRedOpCreatePreMulSum(
      op, scalar, dtype, residence, nccl_comm_));
}

void NCCLCommContext::RedOpDestroy(ncclRedOp_t op) {
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclRedOpDestroy(op, nccl_comm_));
}
#endif

}  // namespace phi::distributed
