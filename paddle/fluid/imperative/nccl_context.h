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

#include <string>
#include <vector>

#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/nccl_helper.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace imperative {

struct ParallelStrategy {
  int nranks_{1};
  int local_rank_{0};
  int device_id_{0};
  std::vector<std::string> trainer_endpoints_{};
  std::string current_endpoint_{""};
};

class NCCLParallelContext {
 public:
  NCCLParallelContext(const ParallelStrategy& strategy,
                      const platform::Place& place)
      : strategy_(strategy), place_(place) {}

  ~NCCLParallelContext() {}
  void Init();

 protected:
  void BcastNCCLID(ncclUniqueId* nccl_id, int root);

 private:
  ParallelStrategy strategy_;
  platform::Place place_;
};

void InitNCCLContext(const platform::CUDAPlace& place, int nranks,
                     int local_rank);

}  //  namespace imperative
}  //  namespace paddle
