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

#pragma once

#include <ThreadPool.h>  // ThreadPool in thrird party

#include <deque>
#include <functional>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/blocking_queue.h"
#include "paddle/fluid/framework/details/exception_holder.h"
#include "paddle/fluid/framework/details/execution_strategy.h"
#include "paddle/fluid/framework/details/fetch_op_handle.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/details/ssa_graph_executor.h"
#include "paddle/fluid/framework/ir/graph.h"

namespace paddle {
namespace framework {
class Scope;

namespace details {

struct OpDependentData {
  std::unordered_map<OpHandleBase *, size_t> pending_ops_;
  std::unordered_set<VarHandleBase *> pending_vars_;
  std::unordered_set<OpHandleBase *> ready_ops_;
  size_t num_ops_{0};
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
