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

#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
#include "paddle/fluid/platform/collective_helper.h"

#include <memory>
#include <utility>

#include "paddle/fluid/platform/dynload/nccl.h"

namespace paddle {
namespace platform {

NCCLContext* NCCLCommContext::CreateNCCLContext(ncclUniqueId* nccl_id,
                                                int nranks, int rank,
                                                int dev_id, int ring_id) {
  PADDLE_ENFORCE_NOT_NULL(nccl_id);
  PADDLE_ENFORCE_GT(nranks, 1);
  PADDLE_ENFORCE_GE(rank, 0);
  PADDLE_ENFORCE_LT(rank, nranks);
  PADDLE_ENFORCE_GE(dev_id, 0);

  ncclComm_t comm = nullptr;
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaSetDevice(dev_id));
  PADDLE_ENFORCE_CUDA_SUCCESS(
      platform::dynload::ncclCommInitRank(&comm, nranks, *nccl_id, rank));

  comm_map_mutex_.lock();
  if (comm_map_.count(ring_id) == 0) {
    comm_map_.emplace(ring_id, std::map<int, std::unique_ptr<NCCLContext>>());
    comm_map_[ring_id].emplace(
        dev_id, std::unique_ptr<NCCLContext>(new NCCLContext(dev_id, comm)));
  }
  comm_map_mutex_.unlock();

  VLOG(1) << "nccl communicator of rank " << rank << " in ring " << ring_id
          << " has been created";

  std::call_once(once_flag_, []() {
    std::atexit([]() { NCCLCommContext::Instance().ReleaseNCCLComms(); });
  });

  return comm_map_[ring_id][dev_id].get();
}

void NCCLCommContext::CreateAllNCCLContexts(const std::vector<int>& dev_ids,
                                            int ring_id) {
  PADDLE_ENFORCE_GT(dev_ids.size(), 0);

  const int kDevices = dev_ids.size();
  ncclComm_t comms[kDevices];
  PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclCommInitAll(
      comms, dev_ids.size(), dev_ids.data()));

  PADDLE_ENFORCE_EQ(comm_map_.count(ring_id), 0);
  comm_map_.emplace(ring_id, std::map<int, std::unique_ptr<NCCLContext>>());

  auto& dev2comm = comm_map_[ring_id];
  for (size_t i = 0; i < dev_ids.size(); ++i) {
    dev2comm.emplace(dev_ids[i], std::unique_ptr<NCCLContext>(
                                     new NCCLContext(dev_ids[i], comms[i])));
  }

  std::call_once(once_flag_, []() {
    std::atexit([]() { NCCLCommContext::Instance().ReleaseNCCLComms(); });
  });
}

void NCCLCommContext::ReleaseNCCLComms() {
  // CUDADeviceContext maintain the lifetime of nccl_comm_t, so we should not
  // destroy nccl_comm_t explicitly. Please refer to
  // platform::CUDADeviceContext::~CUDADeviceContext()
  for (auto& p : comm_map_) {
    for (auto& q : p.second) {
      q.second.reset();
    }
  }
}

}  // namespace platform
}  // namespace paddle

#endif
