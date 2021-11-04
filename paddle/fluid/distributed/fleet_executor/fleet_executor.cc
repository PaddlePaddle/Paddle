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

#include "paddle/fluid/distributed/fleet_executor/fleet_executor.h"
#include "paddle/fluid/distributed/fleet_executor/runtime_graph.h"
#include "paddle/fluid/framework/program_desc.h"

namespace paddle {
namespace distributed {

FleetExecutor::FleetExecutor(const std::string& exe_desc_str) {
  // Initialize Executor
}

FleetExecutor::~FleetExecutor() {
  // Destroy Executor
}

void FleetExecutor::Init(const paddle::framework::ProgramDesc& program_desc) {
  // Compile and Initialize
}

void FleetExecutor::Run() {
  // Run
}

void FleetExecutor::Release() {
  // Release
}

std::shared_ptr<Carrier> FleetExecutor::GetCarrier() {
  // get carrier
  return nullptr;
}

std::shared_ptr<MessageBus> FleetExecutor::GetMessageBus() {
  // get message bus
  return nullptr;
}

}  // namespace distributed
}  // namespace paddle
