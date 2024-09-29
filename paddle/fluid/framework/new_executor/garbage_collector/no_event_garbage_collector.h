// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <queue>

#include "paddle/fluid/framework/new_executor/garbage_collector/garbage_collector.h"
#include "paddle/fluid/framework/new_executor/workqueue/workqueue.h"

namespace paddle {
namespace framework {

class InterpreterCoreNoEventGarbageCollector
    : public InterpreterCoreGarbageCollector {
 public:
  InterpreterCoreNoEventGarbageCollector();
  ~InterpreterCoreNoEventGarbageCollector();
  void Add(Variable* var, const Instruction& instr) override;

  void Add(Variable* var, const InstructionBase* instr) override;

 private:
  void Add(Variable* var, const phi::DeviceContext* ctx);
  void Add(Garbage garbage, const phi::DeviceContext* ctx);
  std::unique_ptr<WorkQueue> queue_;
  std::unordered_set<const phi::DeviceContext*> ctxs_;
};

}  // namespace framework
}  // namespace paddle
