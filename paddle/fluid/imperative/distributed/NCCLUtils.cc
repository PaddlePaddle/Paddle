//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/imperative/distributed/NCCLUtils.h"

namespace paddle {
namespace imperative {

ncclComm_t NCCLComm::getNcclComm() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (aborted_) {
    auto commFailureMsg =
        commFailureReason_ != ""
            ? std::string(" Original reason for failure was: ") +
                  commFailureReason_
            : "";
    // TORCH_CHECK(
    //     false,
    //     c10::str(
    //         "NCCL communicator was aborted on rank ",
    //         rank_,
    //         ". ",
    //         commFailureMsg));
    // PADDLE_THROW(platform::errors::InvalidArgument(
    //       "ProcessGroup %s does not support allreduce",  getBackendName()));
  }
  return ncclComm_;
}

}  // namespace imperative
}  // namespace paddle
