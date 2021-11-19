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
#include <memory>
#include <string>

#include "paddle/fluid/distributed/fleet_executor/fleet_executor_desc.pb.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace framework {
class ProgramDesc;
}

namespace distributed {
class RuntimeGraph;
class Carrier;
class MessageBus;

class FleetExecutor final {
 public:
  FleetExecutor() = delete;
  explicit FleetExecutor(const std::string& exe_desc_str);
  ~FleetExecutor();
  void Init(const paddle::framework::ProgramDesc& program_desc);
  void Run();
  void Release();

 private:
  DISABLE_COPY_AND_ASSIGN(FleetExecutor);
  FleetExecutorDesc exe_desc_;
  std::unique_ptr<RuntimeGraph> runtime_graph_;
  void InitMessageBus();
  void InitCarrier();
};

}  // namespace distributed
}  // namespace paddle
