//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/imperative/nccl_context.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/imperative/all_reduce.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/gen_comm_id_helper.h"
#endif

#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace framework {
class Variable;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace imperative {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)

void NCCLParallelContext::BcastNCCLId(
    std::vector<ncclUniqueId> &nccl_ids,  // NOLINT
    int root) {
  if (strategy_.local_rank_ == root) {
    std::vector<std::string> other_trainers;
    for (auto &ep : strategy_.trainer_endpoints_) {
      if (ep != strategy_.current_endpoint_) {
        other_trainers.push_back(ep);
      }
    }
    platform::SendBroadCastCommID(other_trainers, &nccl_ids);
  } else {
    platform::RecvBroadCastCommID(strategy_.current_endpoint_, &nccl_ids);
  }
}

void NCCLParallelContext::Init() {
  std::vector<ncclUniqueId> nccl_ids;
  nccl_ids.resize(strategy_.nrings_);

  if (strategy_.local_rank_ == 0) {
    // generate the unique ncclid on the root worker
    for (size_t i = 0; i < nccl_ids.size(); ++i) {
      platform::dynload::ncclGetUniqueId(&nccl_ids[i]);
    }
  }
  BcastNCCLId(nccl_ids, 0);

  int gpu_id = BOOST_GET_CONST(platform::CUDAPlace, place_).device;
  for (int ring_id = 0; ring_id < strategy_.nrings_; ring_id++) {
    VLOG(0) << "init nccl context nranks: " << strategy_.nranks_
            << " local rank: " << strategy_.local_rank_ << " gpu id: " << gpu_id
            << " ring id: " << ring_id;
    // it will assign nccl_comm in CUDADeviceContext within ring_id
    platform::NCCLCommContext::Instance().CreateNCCLComm(
        &nccl_ids[ring_id], strategy_.nranks_, strategy_.local_rank_, gpu_id,
        ring_id);

    compute_events_.emplace_back(
        platform::CudaEventResourcePool::Instance().New(
            BOOST_GET_CONST(platform::CUDAPlace, place_).device));
    comm_events_.emplace_back(platform::CudaEventResourcePool::Instance().New(
        BOOST_GET_CONST(platform::CUDAPlace, place_).device));
  }
}

void NCCLParallelContext::AllReduceByStream(const framework::Variable &src,
                                            framework::Variable *dst,
                                            int ring_id, bool use_calc_stream) {
  PADDLE_ENFORCE_EQ(
      platform::is_gpu_place(place_), true,
      platform::errors::Unimplemented(
          "Dynamic graph mode does not support multi-CPU training yet."));
  AllReduce(src, dst, strategy_, ring_id, use_calc_stream);
}

paddle::platform::DeviceContext *NCCLParallelContext::GetDeviceContext(
    int ring_id) {
  return static_cast<platform::DeviceContext *>(
      platform::NCCLCommContext::Instance()
          .Get(ring_id, place_)
          ->dev_context());
}

void NCCLParallelContext::WaitCompute(int ring_id) {
  PADDLE_ENFORCE_GE(ring_id, 0, platform::errors::OutOfRange(
                                    "ring id must >= 0, but got %d", ring_id));
  PADDLE_ENFORCE_LT(ring_id, compute_events_.size(),
                    platform::errors::OutOfRange(
                        "ring id must < compute events size,"
                        "but got ring id = %d, compute events size = %d",
                        ring_id, compute_events_.size()));

  auto compute_stream = static_cast<platform::CUDADeviceContext *>(
                            platform::DeviceContextPool::Instance().Get(place_))
                            ->stream();
  auto comm_stream =
      platform::NCCLCommContext::Instance().Get(ring_id, place_)->stream();
  auto event = compute_events_[ring_id].get();

// compute_stream-->event-->comm_stream
#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE_CUDA_SUCCESS(hipEventRecord(event, compute_stream));
  PADDLE_ENFORCE_CUDA_SUCCESS(hipStreamWaitEvent(comm_stream, event, 0));
#else
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaEventRecord(event, compute_stream));
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamWaitEvent(comm_stream, event, 0));
#endif
}

void NCCLParallelContext::WaitComm(int ring_id) {
  PADDLE_ENFORCE_GE(ring_id, 0, platform::errors::OutOfRange(
                                    "ring id must >= 0, but got %d", ring_id));
  PADDLE_ENFORCE_LT(ring_id, comm_events_.size(),
                    platform::errors::OutOfRange(
                        "ring id must < comm events size,"
                        "but got ring id = %d, comm events size = %d",
                        ring_id, comm_events_.size()));

  auto compute_stream = static_cast<platform::CUDADeviceContext *>(
                            platform::DeviceContextPool::Instance().Get(place_))
                            ->stream();
  auto comm_stream =
      platform::NCCLCommContext::Instance().Get(ring_id, place_)->stream();
  auto event = comm_events_[ring_id].get();

// comm_stream-->event-->compute_stream
#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE_CUDA_SUCCESS(hipEventRecord(event, comm_stream));
  PADDLE_ENFORCE_CUDA_SUCCESS(hipStreamWaitEvent(compute_stream, event, 0));
#else
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaEventRecord(event, comm_stream));
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamWaitEvent(compute_stream, event, 0));
#endif
}

#endif

}  //  namespace imperative
}  //  namespace paddle
