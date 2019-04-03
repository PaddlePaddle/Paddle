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
#pragma once

// network header files
#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <sys/socket.h>

#include <string>
#include <vector>

#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/device_context.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/dynload/nccl.h"
#endif
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/string/split.h"

namespace paddle {
namespace imperative {

struct ParallelStrategy {
  int nranks_{1};
  int local_rank_{0};
  std::vector<std::string> trainer_endpoints_{};
  std::string current_endpoint_{""};
};

class ParallelContext {
 public:
  explicit ParallelContext(const ParallelStrategy& strategy,
                           const platform::Place& place)
      : strategy_(strategy), place_(place) {}

  virtual ~ParallelContext() {}

  virtual void Init() = 0;

 protected:
  ParallelStrategy strategy_;
  platform::Place place_;
};

#ifdef PADDLE_WITH_CUDA
class NCCLParallelContext : ParallelContext {
 public:
  explicit NCCLParallelContext(const ParallelStrategy& strategy,
                               const platform::Place& place)
      : ParallelContext(strategy, place) {}

  ~NCCLParallelContext() {}

  void BcastNCCLId(ncclUniqueId* nccl_id, int root);

  void Init() override;

 protected:
  void RecvNCCLID(const std::string& endpoint, ncclUniqueId* nccl_id);

  void SendNCCLID(const std::string& endpoint, ncclUniqueId* nccl_id);
};
#endif

}  //  namespace imperative
}  //  namespace paddle
