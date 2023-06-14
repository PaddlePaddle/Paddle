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
#include "paddle/fluid/framework/new_executor/standalone_executor.h"

#include "paddle/fluid/framework/new_executor/feed_fetch_utils.h"
#include "paddle/fluid/framework/new_executor/interpreter/interpreter_util.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"

namespace paddle {
namespace framework {
StandaloneExecutor::StandaloneExecutor(const platform::Place& place,
                                       const interpreter::Plan& plan,
                                       Scope* scope)
    : place_(place), plan_(plan), scope_(scope) {
  int64_t micro_batch_num = plan_.MicroBatchNum();
  for (int64_t i = 0; i < micro_batch_num; ++i) {
    micro_batch_scopes_.emplace_back(&scope->NewScope());
  }

  std::stringstream ss;
  ss << "Create " << micro_batch_num << " micro_batch_scopes for scope "
     << scope_ << " : ";
  for (Scope* scope : micro_batch_scopes_) {
    ss << scope << ", ";
  }
  VLOG(6) << ss.str();

  const auto& jobs = plan_.JobList();
  for (const auto& job : jobs) {
    const std::string& job_type = job->Type();
    std::shared_ptr<ProgramDesc> program =
        std::make_shared<ProgramDesc>(*(plan_.Program(job_type)));
    SetColAttrForFetchOps(*job, program);

    int64_t micro_batch_id = job->MicroBatchId();
    PADDLE_ENFORCE(
        micro_batch_id >= 0 && micro_batch_id < micro_batch_num,
        phi::errors::Unavailable("The micro batch id (%lld) out of bound, "
                                 "which should be in the range of [0, %lld].",
                                 micro_batch_id,
                                 micro_batch_num));

    interpreter::ExecutionConfig execution_config;
    execution_config.create_local_scope = false;
    // TODO(Ruibiao): hack skip gc all vars for multiple jobs, improve it later
    if (jobs.size() > 1) {
      for (VarDesc* var : program->Block(0).AllVars()) {
        execution_config.skip_gc_vars.insert(var->Name());
      }
    }

    interpretercores_.emplace_back(
        std::make_unique<InterpreterCore>(place_,
                                          program->Block(0),
                                          micro_batch_scopes_[micro_batch_id],
                                          execution_config));
    interpretercores_.back()->SetCopyProgram(program);
  }
}

paddle::framework::FetchList StandaloneExecutor::Run(
    const std::vector<std::string>& feed_names) {
  platform::RecordEvent record_event(
      "StandaloneExecutor::run", platform::TracerEventType::UserDefined, 1);

  if (plan_.MicroBatchNum() > 1) {
    PADDLE_ENFORCE_EQ(feed_names.size(),
                      0,
                      phi::errors::Unimplemented(
                          "Unsupported feed data for multiple micro_batch, "
                          "please use non-iterative DataLoader for now."));
  }

  const auto& jobs = plan_.JobList();
  for (size_t job_idx = 0; job_idx < jobs.size(); ++job_idx) {
    const auto& job = jobs[job_idx];
    const std::string& job_type = job->Type();

    VLOG(6) << "Run job (" << job_idx << "), type = " << job_type
            << ", micro_batch_id =" << job->MicroBatchId();

    interpretercores_[job_idx]->Run(feed_names, /*need_fetch = */ false);
  }

  // return Fetch Tensors
  auto* fetch_var = scope_->FindVar(interpreter::kFetchVarName);
  if (fetch_var) {
    return std::move(*fetch_var->GetMutable<framework::FetchList>());
  } else {
    return {};
  }
}

}  // namespace framework
}  // namespace paddle
