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
#include <vector>

#include "paddle/fluid/distributed/fleet_executor/carrier.h"
#include "paddle/fluid/distributed/fleet_executor/fleet_executor_desc.pb.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/macros.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace framework {
class ProgramDesc;
class Scope;
}  // namespace framework

namespace distributed {
class RuntimeGraph;
class MessageBus;
class TaskNode;

class FleetExecutor final {
 public:
  FleetExecutor() = delete;
  explicit FleetExecutor(const std::string& exe_desc_str);
  explicit FleetExecutor(const FleetExecutorDesc& exe_desc);
  ~FleetExecutor();
  void Init(
      int32_t num_of_carriers,
      const std::vector<framework::ProgramDesc*>& program_desc,
      framework::Scope* scope,
      const platform::Place& place,
      int64_t num_micro_batches,
      const std::vector<std::vector<TaskNode*>>& task_nodes,
      const std::vector<std::unordered_map<int64_t, int64_t>>& task_id_to_rank,
      const std::vector<std::string>& inference_root_scope_vars = {},
      const std::vector<framework::Scope*>& micro_scope_list = {},
      paddle::framework::ProgramDesc* source_program = nullptr);
  void Run();
  std::shared_ptr<RuntimeGraph> CreateRuntimeGraph(
      const framework::ProgramDesc& program_desc,
      const std::vector<TaskNode*>& task_nodes,
      const std::unordered_map<int64_t, int64_t>& task_id_to_rank,
      const std::vector<std::string>& inference_root_scope_vars);

 private:
  DISABLE_COPY_AND_ASSIGN(FleetExecutor);
  void InitMessageBus();
  void InitCarrier(Carrier* carrier,
                   framework::Scope* scope,
                   framework::Scope* minibatch_scope,
                   const platform::Place& place,
                   const framework::ProgramDesc& program_desc,
                   const std::vector<framework::Scope*>& micro_scope_list,
                   const std::shared_ptr<RuntimeGraph>& runtime_graph);
  void WaitCondVarToExit();
  void CopyParametersFromRoot(
      const framework::ProgramDesc& program,
      const std::vector<std::string>& inference_root_scope_vars);
  void CreateSourceAndSink(
      int64_t max_run_times,
      const std::vector<std::vector<TaskNode*>>& task_nodes,
      const platform::Place& place,
      paddle::framework::ProgramDesc* source_program);

  FleetExecutorDesc exe_desc_;
  std::vector<std::shared_ptr<RuntimeGraph>> runtime_graph_;
  std::unique_ptr<Interceptor> source_interceptor_;
  std::unique_ptr<Interceptor> sink_interceptor_;
  std::vector<std::unique_ptr<Carrier>> carriers_;
  std::mutex mutex_;
  std::condition_variable cond_var_;
  std::unique_ptr<TaskLoopThreadPool> thread_pool_;
  std::vector<framework::Scope*> microbatch_scopes_;
  framework::Scope* root_scope_{nullptr};
  framework::Scope* minibatch_scope_{nullptr};
};

}  // namespace distributed
}  // namespace paddle
