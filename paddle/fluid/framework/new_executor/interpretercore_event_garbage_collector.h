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

#include <queue>
#include "paddle/fluid/framework/new_executor/interpretercore_garbage_collector.h"
#include "paddle/fluid/framework/new_executor/workqueue/workqueue.h"

namespace paddle {
namespace framework {

class InterpreterCoreEventGarbageCollector
    : public InterpreterCoreGarbageCollector {
 public:
  InterpreterCoreEventGarbageCollector();
  ~InterpreterCoreEventGarbageCollector();

  virtual void Add(Variable* var, platform::DeviceEvent& event,
                   const platform::DeviceContext* ctx) override;

 private:
  void Add(Garbage garbage, platform::DeviceEvent& event,
           const platform::DeviceContext* ctx);
  void Free(GarbageQueue* garbages, platform::DeviceEvent& event,
            const platform::DeviceContext* ctx);
  void Free(Garbage& garbage, platform::DeviceEvent& event,
            const platform::DeviceContext* ctx);

  std::unique_ptr<WorkQueue> queue_;
  paddle::memory::SpinLock spinlock_;
};
}  // namespace framework
}  // namespace paddle
