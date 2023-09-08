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
#include "paddle/fluid/framework/new_executor/program_interpreter.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"
#include "paddle/phi/core/flags.h"

#include "paddle/fluid/ir/transforms/pd_op_to_kernel_pass.h"

#include "paddle/fluid/ir/transforms/inplace_pass.h"
#include "paddle/fluid/ir_adaptor/translator/translate.h"
#include "paddle/ir/core/program.h"
#include "paddle/ir/pass/pass.h"
#include "paddle/ir/pass/pass_manager.h"

PHI_DECLARE_bool(enable_new_ir_in_executor);
PHI_DECLARE_bool(enable_new_ir_api);
PHI_DECLARE_bool(new_ir_apply_inplace_pass);

namespace paddle {
namespace framework {
StandaloneExecutor::StandaloneExecutor(const platform::Place& place,
                                       const interpreter::Plan& plan,
                                       Scope* scope)
    : place_(place), plan_(plan), scope_(scope) {
  int64_t micro_batch_num = plan_.MicroBatchNum();
  vec_force_events_to_wait_.resize(micro_batch_num);
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
    std::shared_ptr<ProgramDesc> program = nullptr;
    std::shared_ptr<::ir::Program> ir_program = nullptr;
    if (FLAGS_enable_new_ir_api) {
      ir_program = plan_.IrProgram(job_type);
    } else {
      program = std::make_shared<ProgramDesc>(*(plan_.Program(job_type)));
    }

    int64_t micro_batch_id = job->MicroBatchId();
    PADDLE_ENFORCE(
        micro_batch_id >= 0 && micro_batch_id < micro_batch_num,
        phi::errors::Unavailable("The micro batch id (%lld) out of bound, "
                                 "which should be in the range of [0, %lld].",
                                 micro_batch_id,
                                 micro_batch_num));

    if (micro_batch_num > 1 && !FLAGS_enable_new_ir_api) {
      SetColAttrForFeedFetchOps(program, micro_batch_num, micro_batch_id);
    }

    interpreter::ExecutionConfig execution_config;
    execution_config.create_local_scope = false;
    execution_config.skip_gc_vars = job->SkipGcVars();

    // TODO(phlrain) we only support cpu for now
    if (FLAGS_enable_new_ir_in_executor) {
      std::shared_ptr<::ir::Program> base_program = ir_program;
      if (!FLAGS_enable_new_ir_api) {
        VLOG(6) << "begin to translate" << std::endl;
        base_program = paddle::TranslateLegacyProgramToProgram(*program);
      }
      auto block = base_program->block();
      for (auto it = block->begin(); it != block->end(); ++it) {
        if ((*it)->name() == "pd.fetch") {
          size_t index = (*it)
                             ->attributes()
                             .at("col")
                             .dyn_cast<ir::Int32Attribute>()
                             .data();

          if (fetch_var_names_.size() < index + 1) {
            fetch_var_names_.resize(index + 1);
          }

          fetch_var_names_[index] = (*it)
                                        ->attributes()
                                        .at("name")
                                        .dyn_cast<ir::StrAttribute>()
                                        .AsString() +
                                    "@fetch";
        }
      }
      auto kernel_program =
          paddle::dialect::PdOpLowerToKernelPass(base_program.get(), place);

      if (FLAGS_new_ir_apply_inplace_pass) {
        ir::PassManager pm(ir::IrContext::Instance(), 3);
        pm.AddPass(ir::CreateInplacePass());
        pm.Run(kernel_program.get());
      }

      interpretercores_.emplace_back(
          std::make_shared<InterpreterCore>(place_,
                                            fetch_var_names_,
                                            std::move(kernel_program),
                                            scope_,
                                            execution_config));
    } else {
      interpretercores_.emplace_back(
          std::make_shared<InterpreterCore>(place_,
                                            program->Block(0),
                                            micro_batch_scopes_[micro_batch_id],
                                            execution_config));
      interpretercores_.back()->SetCopyProgram(program);

      // Note(lizhiyu): Add mannual event info
      auto prog_inter = const_cast<ProgramInterpreter*>(
          static_cast<const ProgramInterpreter*>(
              interpretercores_.back()->Impl()));
      prog_inter->SetForceEventsToWaitInfo(
          &(vec_force_events_to_wait_[micro_batch_id]));

      // NOTE(lizhiyu): Now we only check backward subprogram. After static
      // build strategy is completely, we should
      //                check all the program in the PP strategy.
      if (job_type == "backward" && jobs.size() > 1) {
        PADDLE_ENFORCE_EQ(static_cast<const ProgramInterpreter*>(
                              interpretercores_.back()->Impl())
                              ->IsStaticBuild(),
                          true,
                          phi::errors::InvalidArgument(
                              "When using pipeline strategy in auto "
                              "prarallelism with new executor, "
                              "the backward subprogram must be builded in real "
                              "static build mode, but it can not "
                              "be staticly builded in this case. You can "
                              "enable 'GLOG_v=1' to obtain log information."));
      }
    }
  }
}

paddle::framework::FetchList StandaloneExecutor::Run(
    const std::vector<std::string>& feed_names) {
  platform::RecordEvent record_event(
      "StandaloneExecutor::run", platform::TracerEventType::UserDefined, 1);

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
    // TODO(zhaoyinglia): use a more general method
    if (jobs.size() > 1 && job_type != "forward") {
      const std::vector<std::string> tmp_feed_names = {};
      interpretercores_[job_idx]->Run(tmp_feed_names, /*need_fetch = */ false);
    } else {
      interpretercores_[job_idx]->Run(feed_names, /*need_fetch = */ false);
    }
  }

  // return Fetch Tensors
  if (FLAGS_enable_new_ir_in_executor) {
    framework::FetchList fetch_res;
    for (auto& var_name : fetch_var_names_) {
      auto* var = scope_->FindVar(var_name);
      fetch_res.push_back(var->Get<phi::DenseTensor>());
    }

    return fetch_res;
  } else {
    auto* fetch_var = scope_->FindVar(interpreter::kFetchVarName);
    if (fetch_var) {
      return std::move(*fetch_var->GetMutable<framework::FetchList>());
    } else {
      return {};
    }
  }
}

}  // namespace framework
}  // namespace paddle
