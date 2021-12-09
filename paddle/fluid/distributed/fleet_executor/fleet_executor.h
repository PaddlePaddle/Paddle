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
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace framework {
class ProgramDesc;
class Scope;
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
  void Init(const framework::ProgramDesc& program_desc, framework::Scope* scope,
            const platform::Place& place);
  void Run();

 private:
  DISABLE_COPY_AND_ASSIGN(FleetExecutor);
  void InitMessageBus();
  void InitCarrier();
  void CopyParameters(int microbatch_id, const framework::ProgramDesc& program);
  FleetExecutorDesc exe_desc_;
  std::shared_ptr<RuntimeGraph> runtime_graph_;
  framework::Scope* root_scope_;
  framework::Scope* minibatch_scope_;
  platform::Place place_;
  std::vector<framework::Scope*> microbatch_scopes_;
};

}  // namespace distributed
}  // namespace paddle
