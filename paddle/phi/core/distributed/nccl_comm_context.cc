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
                                    gpuStream_t stream) {
  int64_t out_size = in_tensor.numel() / GetSize();
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::ncclReduceScatter(in_tensor.data(),
                                      out_tensor->data(),
                                      out_size,
                                      ToNCCLDataType(in_tensor.type()),
                                      ncclSum,
                                      nccl_comm_,
                                      stream));
}

void NCCLCommContext::Send(const phi::DenseTensor& in_tensor,
                           const int& peer,
                           gpuStream_t stream) {
  if (FLAGS_enable_nccl_dynamic_check) {
    NCCLDynamicCheck::CheckShape(in_tensor, rank_, rank_, nccl_comm_);
  }

  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::ncclSend(in_tensor.data(),
                             in_tensor.numel(),
                             ToNCCLDataType(in_tensor.type()),
                             peer,
                             nccl_comm_,
                             stream));
  VLOG(3) << "rank " << GetRank() << " send " << phi::product(in_tensor.dims())
          << " to " << peer;
}

void NCCLCommContext::Recv(phi::DenseTensor* out_tensor,
                           const int& peer,
                           gpuStream_t stream) {
  if (FLAGS_enable_nccl_dynamic_check) {
    NCCLDynamicCheck::CheckShape(*out_tensor, rank_, rank_, nccl_comm_);
  }

  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::ncclRecv(out_tensor->data(),
                             out_tensor->numel(),
                             ToNCCLDataType(out_tensor->type()),
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

}  // namespace distributed
}  // namespace phi
