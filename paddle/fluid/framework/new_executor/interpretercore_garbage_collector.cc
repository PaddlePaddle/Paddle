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

#include "paddle/fluid/framework/new_executor/interpretercore_garbage_collector.h"
#include "paddle/fluid/framework/garbage_collector.h"

namespace paddle {
namespace framework {

InterpreterCoreGarbageCollector::InterpreterCoreGarbageCollector() {
  garbages_ = std::make_unique<GarbageQueue>();
  max_memory_size_ = static_cast<size_t>(GetEagerDeletionThreshold());
  cur_memory_size_ = 0;
}

void InterpreterCoreGarbageCollector::Add(Variable* var) {
  PADDLE_THROW(
      platform::errors::Unimplemented("Not allowed to call the member function "
                                      "of InterpreterCoreGarbageCollector"));
}

void InterpreterCoreGarbageCollector::Add(Variable* var,
                                          platform::DeviceEvent& event,
                                          const platform::DeviceContext* ctx) {
  PADDLE_THROW(
      platform::errors::Unimplemented("Not allowed to call the member function "
                                      "of InterpreterCoreGarbageCollector"));
}

}  // namespace framework
}  // namespace paddle