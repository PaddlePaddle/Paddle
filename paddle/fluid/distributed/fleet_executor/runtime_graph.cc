// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/fleet_executor/runtime_graph.h"
#include "paddle/fluid/distributed/fleet_executor/task_node.h"

namespace paddle {
namespace distributed {

std::string RuntimeGraph::DebugString() const {
  std::ostringstream os;
  os << "\nRuntime Graph Debug: \n";
  for (const auto& pair : interceptor_id_to_node_) {
    os << pair.second->DebugString();
    os << "\n";
  }
  return os.str();
}

}  // namespace distributed
}  // namespace paddle
