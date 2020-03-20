// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/fleet/nccl_wrapper.h"
#include <utility>
#include "paddle/fluid/framework/data_feed.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace framework {

std::shared_ptr<NCCLWrapper> NCCLWrapper::s_instance_ = NULL;
bool NCCLWrapper::is_initialized_ = false;

void NCCLWrapper::InitNCCL() {
#if defined(PADDLE_WITH_NCCL)
  PADDLE_ENFORCE(platform::dynload::ncclCommInitRank(
      &(nccl_info_.comm_), nccl_info_.global_ranks_, nccl_info_.nccl_id_,
      nccl_info_.my_global_rank_));
#endif
  return;
}

void NCCLWrapper::SetNCCLId(const NCCLInfo& nccl_info) {
#if defined(PADDLE_WITH_NCCL)
  nccl_info_.nccl_id_ = nccl_info.nccl_id_;
#endif
  return;
}

NCCLInfo NCCLWrapper::GetNCCLId() {
#if defined(PADDLE_WITH_NCCL)
  PADDLE_ENFORCE(platform::dynload::ncclGetUniqueId(&(nccl_info_.nccl_id_)));
#endif
  return nccl_info_;
}

void NCCLWrapper::SetRankInfo(const int local_rank, const int global_rank,
                              const int ranks) {
#if defined(PADDLE_WITH_NCCL)
  nccl_info_.local_rank_ = local_rank;
  nccl_info_.my_global_rank_ = global_rank;
  nccl_info_.global_ranks_ = ranks;
  PADDLE_ENFORCE(cudaSetDevice(local_rank));
  PADDLE_ENFORCE(cudaStreamCreate(&(nccl_info_.stream_)));
#endif
  return;
}

void NCCLWrapper::SyncVar(const int root_rank, const Scope& scope,
                          const std::vector<std::string>& var_names) {
#if defined(PADDLE_WITH_NCCL)
  for (auto& name : var_names) {
    auto var = scope.FindVar(name);
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    int32_t total_size = tensor->numel();
    PADDLE_ENFORCE(platform::dynload::ncclBcast(
        reinterpret_cast<void*>(tensor->data<float>()), total_size, ncclFloat,
        root_rank, nccl_info_.comm_, nccl_info_.stream_));
    cudaStreamSynchronize(nccl_info_.stream_);
  }
#endif
  return;
}

}  // end namespace framework
}  // end namespace paddle
