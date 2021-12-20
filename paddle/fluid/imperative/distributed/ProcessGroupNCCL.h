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

#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/imperative/distributed/ProcessGroup.h"
#include "paddle/fluid/platform/dynload/nccl.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/gen_comm_id_helper.h"

namespace paddle {
namespace framework {
class Variable;
class Tensor;
}  // namespace framework
}  // namespace paddle

// constexpr auto kNoTimeout = std::chrono::milliseconds(0);
// constexpr auto kProcessGroupDefaultTimeout =
//     std::chrono::milliseconds(30 * 60 * 1000);

constexpr const char* NCCL_BACKEND_NAME = "nccl";

namespace paddle {
namespace imperative {

class ProcessGroupNCCL : public ProcessGroup {
 public:
  class WorkNCCL : public ProcessGroup::Work,
                   public std::enable_shared_from_this<WorkNCCL> {
   public:
    WorkNCCL(int rank, OpType OpType);

    virtual ~WorkNCCL();
  };

  ProcessGroupNCCL(const ProcessGroupStrategy& strategy, int rank, int size);

  // const std::string getBackendName() const override {
  //     return std::string(NCCL_BACKEND_NAME);
  // }

 protected:
  ProcessGroupStrategy strategy_;

 private:
  void BcastNCCLId(std::vector<ncclUniqueId>& nccl_ids, int root,  // NOLINT
                   int server_fd);

  void Init();
};

}  //  namespace imperative
}  //  namespace paddle
