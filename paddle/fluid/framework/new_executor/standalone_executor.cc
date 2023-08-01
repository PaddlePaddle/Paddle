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

#include "paddle/fluid/ir/transforms/pd_op_to_kernel_pass.h"

#include "paddle/fluid/ir_adaptor/translator/translate.h"

PHI_DECLARE_bool(enable_new_ir_in_executor);

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
    execution_config.skip_gc_vars = job->SkipGcVars();

    // TODO(phlrain) we only support cpu for now
    if (FLAGS_enable_new_ir_in_executor) {
      VLOG(6) << "begin to translate" << std::endl;
      auto base_program = paddle::TranslateLegacyProgramToProgram(*program);
      auto kernel_program =
          paddle::dialect::PdOpLowerToKernelPass(base_program.get(), place);
      interpretercores_.emplace_back(std::make_shared<InterpreterCore>(
          place_, std::move(kernel_program), scope_, execution_config));
    } else {
      interpretercores_.emplace_back(
          std::make_shared<InterpreterCore>(place_,
                                            program->Block(0),
                                            micro_batch_scopes_[micro_batch_id],
                                            execution_config));
      interpretercores_.back()->SetCopyProgram(program);
    }
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

  std::map<std::string, size_t> type_to_first_id;
  if (!is_interpretercore_build_result_shared_) {
    type_to_first_id[jobs[0]->Type()] = 0;
    for (size_t job_idx = 1; job_idx < jobs.size(); ++job_idx) {
      interpretercores_[job_idx]->ShareWorkQueueFrom(interpretercores_[0]);
      // TODO(Ruibiao): Share other build result, e.g., kernel choosing, data
      // transfer, op dependency, thread scheduling, GC, event analyzer, and so
      // on.
      if (type_to_first_id.count(jobs[job_idx]->Type()) == 0) {
        type_to_first_id[jobs[job_idx]->Type()] = job_idx;
      }
    }
    is_interpretercore_build_result_shared_ = true;
  }

  for (size_t job_idx = 0; job_idx < jobs.size(); ++job_idx) {
    const auto& job = jobs[job_idx];
    const std::string& job_type = job->Type();
    platform::RecordEvent record_event(
        job_type + "-" + std::to_string(job->MicroBatchId()),
        platform::TracerEventType::UserDefined,
        1);

    VLOG(6) << "Run job (" << job_idx << "), type = " << job_type
            << ", micro_batch_id =" << job->MicroBatchId();
    // Note(sonder): Share build results don't work for new IR now.
    if (type_to_first_id.count(job_type) != 0 &&
        !FLAGS_enable_new_ir_in_executor) {
      interpretercores_[job_idx]->ShareBuildResultsFrom(
          interpretercores_[type_to_first_id[job_type]]);
    }
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
