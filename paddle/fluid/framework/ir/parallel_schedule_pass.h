// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace framework {
namespace ir {

const char kStreamMap[] = "__stream_map__";
const char kEventDependMap[] = "__event_depend_map__";

// Only analyze the main block, so the key is string.
// Each stream will have an event.
using stream_map_t =
    std::unordered_map<std::string /*node repr*/, int /*stream id*/>;

// Record the input events each operator depends on, and record the events the
// outputs will update.
using event_depend_map_t =
    std::unordered_map<std::string, std::set<int> /*stream ids*/>;

class ParallelSchedulePass : public Pass {
 public:
 protected:
  std::unique_ptr<Graph> ApplyImpl(std::unique_ptr<Graph> graph) const override;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(parallel_schedule_pass,
              paddle::framework::ir::ParallelSchedulePass);
