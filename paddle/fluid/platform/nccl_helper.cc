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
#include "paddle/fluid/platform/nccl_helper.h"

#include <memory>
#include <utility>

#include "paddle/fluid/platform/dynload/nccl.h"

namespace paddle {
namespace platform {

NCCLContextMap::NCCLContextMap(const std::vector<platform::Place>& places,
                               ncclUniqueId* nccl_id, size_t num_trainers,
                               size_t trainer_id) {
  PADDLE_ENFORCE_EQ(places.empty(), false);
  order_.reserve(places.size());
  std::set<int> dev_set;
  for (auto& p : places) {
    int dev_id = boost::get<CUDAPlace>(p).device;
    order_.emplace_back(dev_id);
    dev_set.insert(dev_id);
  }
  PADDLE_ENFORCE_EQ(
      order_.size(), dev_set.size(),
      "NCCL Context Map does not support contain two or more same device");

  const int kOrders = order_.size();
  ncclComm_t comms[kOrders];
  if (num_trainers == 1 && nccl_id == nullptr) {
    std::lock_guard<std::mutex> guard(NCCLGroupGuard::NCCLMutex());
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclCommInitAll(
        comms, static_cast<int>(order_.size()), order_.data()));
  } else {
    PADDLE_ENFORCE_NOT_NULL(nccl_id);
    {
      int nranks = num_trainers * order_.size();
      NCCLGroupGuard gurad;
      for (size_t i = 0; i < order_.size(); ++i) {
        int gpu_id = order_[i];
        int rank;
        if (order_.size() > 1) {
          rank = trainer_id * order_.size() + i;
        } else {
          rank = trainer_id;
        }
        VLOG(1) << "init nccl rank:" << rank << ", nranks:" << nranks
                << ", gpu_id:" << gpu_id;
        PADDLE_ENFORCE_CUDA_SUCCESS(cudaSetDevice(gpu_id));
        PADDLE_ENFORCE_CUDA_SUCCESS(
            dynload::ncclCommInitRank(comms + i, nranks, *nccl_id, rank));
      }
    }
  }
  int i = 0;
  for (auto& dev_id : order_) {
    auto ptr = new NCCLContext(dev_id, comms[i++]);
    contexts_.emplace(dev_id, std::unique_ptr<NCCLContext>(ptr));
  }
}

void NCCLCommunicator::InitNCCLContexts(const std::vector<Place>& places,
                                        ncclUniqueId* nccl_id, int ntrainers,
                                        int trainer_id, int ring_id) {
  PADDLE_ENFORCE_GE(ntrainers, 1);
  PADDLE_ENFORCE_GE(trainer_id, 0);
  PADDLE_ENFORCE_LT(trainer_id, ntrainers);

  // TODO(liuyi05): support update NCCLContextMap, e.g. multithreads
  PADDLE_ENFORCE_EQ(ring2map_.count(ring_id), 0,
                    "Cannot call this function twice with the same ring_id by "
                    "now, we will fix it soon");

  auto ptr = new NCCLContextMap(places, nccl_id, ntrainers, trainer_id);
  ring2map_.emplace(ring_id, std::unique_ptr<NCCLContextMap>(ptr));

  VLOG(1) << "nccl communicator of ring_id" << ring_id << " has been created";

  std::call_once(once_flag_, []() {
    std::atexit([]() { NCCLCommunicator::Instance().ReleaseNCCLResource(); });
  });
}

void NCCLCommunicator::ReleaseNCCLResource() {
  // CUDADeviceContext maintain the lifetime of nccl_comm_t, so we should not
  // destroy nccl_comm_t explicitly. Please refer to
  // platform::CUDADeviceContext::~CUDADeviceContext()
  for (auto& p : ring2map_) {
    p.second.reset();
  }
}

}  // namespace platform
}  // namespace paddle

#endif
