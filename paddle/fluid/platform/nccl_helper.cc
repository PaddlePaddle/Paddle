//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/platform/nccl_helper.h"

#define NCCL_ID_VARNAME "NCCLID"

namespace paddle {
namespace platform {

NCCLContextMap *NCCLContextMap::ctx_map_ptr_ = nullptr;

NCCLContextMap::NCCLContextMap(const std::vector<platform::Place> &places,
                               ncclUniqueId *nccl_id, size_t num_trainers,
                               size_t trainer_id) {
  PADDLE_ENFORCE(!places.empty());
  order_.reserve(places.size());
  for (auto &p : places) {
    int dev_id = boost::get<CUDAPlace>(p).device;
    order_.emplace_back(dev_id);
    VLOG(10) << "init contexts_ dev_id: " << dev_id;
    contexts_.emplace(dev_id, NCCLContext(dev_id));
  }
  PADDLE_ENFORCE_EQ(
      order_.size(), contexts_.size(),
      "NCCL Context Map does not support contain two or more same device");

  if (places.size() <= 1) {
    return;
  }
  std::unique_ptr<ncclComm_t[]> comms(new ncclComm_t[order_.size()]);
  if (nccl_id == nullptr) {
    nccl_id = new ncclUniqueId();
    PADDLE_ENFORCE(platform::dynload::ncclGetUniqueId(nccl_id));
  }
  int nranks = num_trainers * order_.size();
  {
    NCCLGroupGuard gurad;
    for (auto &gpu_id : order_) {
      int rank = trainer_id * order_.size() + gpu_id;
      VLOG(30) << "init nccl rank: " << rank << " nranks: " << nranks;
      PADDLE_ENFORCE(cudaSetDevice(gpu_id));
      PADDLE_ENFORCE(platform::dynload::ncclCommInitRank(
          comms.get() + gpu_id, nranks, *nccl_id, rank));
    }
  }

  int i = 0;
  nranks_ = nranks;
  for (auto &dev_id : order_) {
    contexts_.at(dev_id).comm_ = comms[i++];
    contexts_.at(dev_id).rank_ = trainer_id * order_.size() + dev_id;
  }
}

}  // namespace platform
}  // namespace paddle
