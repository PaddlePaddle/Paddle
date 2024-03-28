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

#pragma once
#include <ThreadPool.h>

#include <condition_variable>  // NOLINT
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/blocking_queue.h"
#include "paddle/fluid/framework/details/exception_holder.h"
#include "paddle/fluid/framework/details/execution_strategy.h"
#include "paddle/fluid/framework/details/ssa_graph_executor.h"

#if defined(PADDLE_WITH_XPU)
namespace paddle {
namespace framework {
class Scope;
namespace details {

struct RunningItem {
  std::atomic<int> dep_num;
  OpHandleBase *op;
};

class OpHandleBase;
}  // namespace details
}  // namespace framework
}  // namespace paddle

#endif
