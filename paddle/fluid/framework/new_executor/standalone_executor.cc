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

#include "paddle/fluid/framework/new_executor/interpreter/interpreter_util.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"

#include "paddle/fluid/ir/pass/pd_op_to_kernel_pass.h"

#include "paddle/fluid/ir_adaptor/translator/translate.h"

PHI_DECLARE_bool(enable_new_ir_in_executor);

namespace paddle {
namespace framework {
StandaloneExecutor::StandaloneExecutor(const platform::Place& place,
                                       const std::vector<ProgramDesc>& programs)
    : place_(place), programs_(programs) {
  std::cerr << "construct stand alone" << std::endl;
  if (FLAGS_enable_new_ir_in_executor) {
    for (size_t i = 0; i < programs_.size(); ++i) {
      VLOG(6) << "begin to translate" << std::endl;
      auto base_progrm = paddle::TranslateLegacyProgramToProgram(programs_[i]);

      base_progrm->Print(std::cout);
      auto kernel_program =
          paddle::dialect::PdOpLowerToKernelPass(base_progrm.get());

      kernel_program->Print(std::cout);
      ir_programs_.emplace_back(std::move(kernel_program));
    }
  }
}

paddle::framework::FetchList StandaloneExecutor::Run(
    Scope* scope,
    const std::vector<std::string>& feed_names,
    const std::vector<std::string>& fetch_names) {
  platform::RecordEvent record_event(
      "StandaloneExecutor::run", platform::TracerEventType::UserDefined, 1);

  // TODO(Ruibiao): Unified single and multiple program run
  if (programs_.size() == 1) {  // run single program
    VLOG(6) << "Run single program";
    auto core = GetInterpreterCore(scope,
                                   programs_.at(0),
                                   feed_names,
                                   fetch_names,
                                   0,
                                   interpreter::ExecutionConfig());
    VLOG(4) << "StandaloneExecutor: " << this << ", InterpreterCore: " << core;

    return core->Run(feed_names);
  } else {  // run multiple programs
    VLOG(6) << "Run multiple program, programs_.size() " << programs_.size();
    FetchList merged_fetch_list;
    for (size_t program_idx = 0; program_idx < programs_.size();
         ++program_idx) {
      const ProgramDesc& program = programs_[program_idx];

      interpreter::ExecutionConfig execution_config;
      execution_config.create_local_scope = false;
      // TODO(Ruibiao): hack skip gc for all vars, improve it later
      std::set<std::string> skip_gc_vars;
      for (VarDesc* var : program.Block(0).AllVars()) {
        execution_config.skip_gc_vars.insert(var->Name());
      }

      // TODO(Ruibiao): ONLY support feeds data in the first program for now
      const std::vector<std::string>& real_feed_names =
          (program_idx == 0 ? feed_names : std::vector<std::string>());
      auto core = GetInterpreterCore(scope,
                                     program,
                                     real_feed_names,
                                     fetch_names,
                                     program_idx,
                                     execution_config);
      const FetchList& fetch_list = core->Run(real_feed_names);
      std::move(fetch_list.begin(),
                fetch_list.end(),
                std::back_inserter(merged_fetch_list));
    }
    return merged_fetch_list;
  }
}

std::shared_ptr<InterpreterCore> StandaloneExecutor::GetInterpreterCore(
    Scope* scope,
    const ProgramDesc& program,
    const std::vector<std::string>& feed_names,
    const std::vector<std::string>& fetch_names,
    size_t program_idx,
    interpreter::ExecutionConfig execution_config) {
  std::ostringstream oss;
  oss << "prog_idx:" << program_idx << ",";
  oss << "feed:";
  for (auto& feedname : feed_names) {
    oss << feedname << ",";
  }
  oss << "fetch:";
  for (auto& fetchname : fetch_names) {
    oss << fetchname << ",";
  }
  oss << "scope:" << scope;

  auto iter = interpretercores_.find(oss.str());

  if (iter == interpretercores_.end()) {
    VLOG(3) << "create interpreter_core for " << oss.str() << " on place "
            << place_;
    std::shared_ptr<InterpreterCore> core = nullptr;
    if (FLAGS_enable_new_ir_in_executor) {
      core = std::make_shared<InterpreterCore>(place_,
                                               program.Block(0),
                                               scope,
                                               ir_programs_[program_idx].get(),
                                               execution_config);
    } else {
      core = std::make_shared<InterpreterCore>(
          place_, program.Block(0), scope, execution_config);
    }
    interpretercores_.emplace(oss.str(), core);
    return core;
  } else {
    return iter->second;
  }
}

}  // namespace framework
}  // namespace paddle
