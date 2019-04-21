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

#include "paddle/fluid/framework/fleet/collective_wrapper.h"
#include <utility>
#include "paddle/fluid/framework/data_feed.h"
#include "paddle/fluid/framework/scope.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/nccl_helper.h"
#endif

namespace paddle {
namespace framework {

#ifdef PADDLE_WITH_CUDA

#define CUDACHECK(cmd)                                              \
  do {                                                              \
    cudaError_t e = cmd;                                            \
    if (e != cudaSuccess) {                                         \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                                \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

#endif

std::shared_ptr<NCCLWrapper> NCCLWrapper::s_instance_ = NULL;
bool NCCLWrapper::is_initialized_ = false;
NCCLInfo NCCLWrapper::nccl_info_;

void NCCLWrapper::InitNCCL() {
#ifdef PADDLE_WITH_CUDA
  VLOG(3) << "Going to init nccl";
  VLOG(3) << "local rank " << nccl_info_.local_rank_;
  VLOG(3) << "my global rank " << nccl_info_.my_global_rank_;
  VLOG(3) << "total ranks " << nccl_info_.global_ranks_;

  PADDLE_ENFORCE(cudaSetDevice(nccl_info_.local_rank_));
  PADDLE_ENFORCE(cudaStreamCreate(&(nccl_info_.stream_)));
  VLOG(3) << platform::dynload::ncclGetErrorString(
      platform::dynload::ncclCommInitRank(
          &(nccl_info_.comm_), nccl_info_.global_ranks_, nccl_info_.nccl_id_,
          nccl_info_.my_global_rank_));
  VLOG(3) << "init nccl done. local rank is " << nccl_info_.local_rank_;
#endif
  return;
}

void NCCLWrapper::SetNCCLId(const NCCLInfo& nccl_info) {
#ifdef PADDLE_WITH_CUDA
  nccl_info_.nccl_id_ = nccl_info.nccl_id_;
#endif
  return;
}

NCCLInfo NCCLWrapper::GetNCCLId() {
#ifdef PADDLE_WITH_CUDA
  PADDLE_ENFORCE(platform::dynload::ncclGetUniqueId(&(nccl_info_.nccl_id_)));
#endif
  return nccl_info_;
}

void NCCLWrapper::SetRankInfo(const int local_rank, const int global_rank,
                              const int ranks) {
#ifdef PADDLE_WITH_CUDA
  nccl_info_.local_rank_ = local_rank;
  nccl_info_.my_global_rank_ = global_rank;
  nccl_info_.global_ranks_ = ranks;
  VLOG(3) << "set rank info:"
          << " local rank " << local_rank << " global_rank " << global_rank
          << " total_ranks " << ranks;
#endif
  return;
}

void NCCLWrapper::SyncVar(const int root_rank, const Scope& scope,
                          const std::vector<std::string>& var_names) {
#ifdef PADDLE_WITH_CUDA
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

void NCCLWrapper::AllReduce(const Scope& scope, const std::string& in_var_name,
                            const std::string& out_var_name,
                            const platform::Place& place) {
#ifdef PADDLE_WITH_CUDA
  auto* in = scope.FindVar(in_var_name);
  auto in_tensor = in->GetMutable<framework::LoDTensor>();
  int dtype = -1;
  dtype = platform::ToNCCLDataType(in_tensor->type());
  int64_t numel = in_tensor->numel();
  auto* sendbuff = in_tensor->mutable_data<float>(place);
  auto* recvbuff = sendbuff;
  VLOG(3) << platform::dynload::ncclGetErrorString(
      platform::dynload::ncclAllReduce(
          (const void*)sendbuff, reinterpret_cast<void*>(recvbuff), numel,
          static_cast<ncclDataType_t>(dtype), ncclSum, nccl_info_.comm_,
          nccl_info_.stream_));
  CUDACHECK(cudaStreamSynchronize(nccl_info_.stream_));
/*
auto* in = scope.FindVar(in_var_name);
auto* out = scope.FindVar(out_var_name);
auto in_tensor = in->Get<framework::LoDTensor>();
int dtype = -1;
dtype = platform::ToNCCLDataType(in_tensor.type());
int64_t numel = in_tensor.numel();
auto* sendbuff = in_tensor.data<void>();
auto* out_tensor = out->GetMutable<framework::LoDTensor>();
out_tensor->Resize(in_tensor.dims());
platform::CUDAPlace gpu_place;
auto* recvbuff = out_tensor->mutable_data<float>(gpu_place);
platform::dynload::ncclAllReduce(
    sendbuff, recvbuff, numel, static_cast<ncclDataType_t>(dtype), ncclSum,
    nccl_info_.comm_, nccl_info_.stream_);
CUDACHECK(cudaStreamSynchronize(nccl_info_.stream_));
*/
#endif
  return;
}

}  // end namespace framework
}  // end namespace paddle
