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
#include "paddle/fluid/memory/allocation/spin_lock.h"
#include "paddle/fluid/platform/device_event.h"

namespace paddle {
namespace framework {

using Garbage = std::shared_ptr<memory::Allocation>;
using GarbageQueue = std::deque<Garbage>;

class InterpreterCoreGarbageCollector {
 public:
  InterpreterCoreGarbageCollector();
  virtual ~InterpreterCoreGarbageCollector(){};
  virtual void Add(Variable* var);
  virtual void Add(Variable* var, platform::DeviceEvent& event,
                   const platform::DeviceContext* ctx);
  DISABLE_COPY_AND_ASSIGN(InterpreterCoreGarbageCollector);

 protected:
  std::unique_ptr<GarbageQueue> garbages_;
  int64_t max_memory_size_;
  int64_t cur_memory_size_;
  memory::SpinLock spinlock_;
};

}  // namespace framework
}  // namespace paddle
