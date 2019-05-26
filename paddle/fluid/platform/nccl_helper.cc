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

#ifndef _WIN32

#include "paddle/fluid/platform/nccl_helper.h"

namespace paddle {
namespace platform {

void NCCLCommGroup::InitRank(int dev_id, int rank_id) {
  std::lock_guard<std::mutex> lg(mutex_);

  PADDLE_ENFORCE(dev_id >= 0 && rank_id >= 0,
                 "invalid arguments: dev_id=%d, rank_id=%d",
                 dev_id, rank_id);
  PADDLE_ENFORCE(comm_map_.count(dev_id) == 0,
                 "device %d is used in the communication group %d",
                 dev_id, group_id_);

  ncclComm_t comm = nullptr;
  PADDLE_ENFORCE(cudaSetDevice(dev_id));
  PADDLE_ENFORCE(platform::dynload::ncclCommInitRank(
      &comm, nranks_, *nccl_id_, rank_id));

  // FIXME(liuyi05): destroy ncclComm_t
  comm_map_.emplace(dev_id, std::unique_ptr<NCCLComm>(
        new NCCLComm(rank_id, comm,
          NCCLCommContext::Instance().DevCtx(dev_id))));
}

}  // namespace platform
}  // namespace paddle

#endif
