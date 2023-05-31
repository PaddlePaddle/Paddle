// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/framework/parallel_compiler.h"

#include <algorithm>
#include <fstream>
#include <thread>

#include "paddle/cinn/backends/codegen_cuda_dev.h"
#include "paddle/cinn/backends/codegen_cuda_host.h"
#include "paddle/cinn/backends/codegen_cuda_util.h"
#include "paddle/cinn/backends/compiler.h"
#include "paddle/cinn/backends/llvm/codegen_x86.h"
#include "paddle/cinn/backends/llvm/runtime_symbol_registry.h"
#include "paddle/cinn/backends/nvrtc/nvrtc_util.h"
#include "paddle/cinn/common/context.h"
#include "paddle/cinn/hlir/framework/pass.h"
#include "paddle/cinn/ir/module.h"

DECLARE_int32(cinn_parallel_compile_size);
DECLARE_int32(cinn_parallel_compile_thread);

namespace cinn {
namespace hlir {
namespace framework {
static constexpr int DebugLogMaxLen = 30000;

std::vector<std::unique_ptr<Instruction>> ParallelCompiler::operator()() {
  if (graph_->fusion_groups.size() == 0) {
    hlir::framework::ApplyPasses(graph_.get(), {"BuildNonFusedGroupsPass"});
  }
  // Task Spilt
  SplitTask();
  // launch task
  LaunchTask();
  // merge instruction
  return MergeResult();
}

OpPatternKind GetOpKind(const framework::Node* node) {
  auto& op_pattern_dict =
      framework::Operator::GetAttrs<OpPatternKind>("OpPattern");
  CHECK(op_pattern_dict.Find(node->op()))
      << "Don't find the pattern of op : " << node->id();
  auto kind = op_pattern_dict[node->op()];

  if (kind == framework::kBroadcast) {
    // As binary op was defined as broadcast, actually it should be
    // element-wise.
    if (node->op()->name != "broadcast_to") {
      return framework::kElementWise;
    }
  }

  return kind;
}

void ParallelCompiler::SplitTask() {
  CHECK(graph_->fusion_groups.size());
  CHECK(graph_->fusion_groups.size() == option_.lowered_funcs.size() ||
        option_.lowered_funcs.size() == 0);
  // split task
  int max_task_num = FLAGS_cinn_parallel_compile_thread > 0
                         ? FLAGS_cinn_parallel_compile_thread
                         : graph_->fusion_groups.size();

  int group_per_task = graph_->fusion_groups.size();
  if (max_task_num > 1) {
    group_per_task = FLAGS_cinn_parallel_compile_size > 0
                         ? FLAGS_cinn_parallel_compile_size
                         : ((graph_->fusion_groups.size() + max_task_num - 1) /
                            max_task_num);
  }

  for (int idx = 0; idx < graph_->fusion_groups.size(); idx += group_per_task) {
    tasks_.emplace_back(this, scope_, graph_, option_, target_);
  }
  VLOG(2) << "Split task to " << tasks_.size() << " sub-task!";
}

void RunTask(ParallelCompiler::Task* task) {
  VLOG(2) << "Stark run sub-task, Thread Id : " << std::this_thread::get_id();
  VLOG(4) << "Start Lowering";
  task->Lowering();
  VLOG(4) << "Start CodegenAndJit";
  task->CodegenAndJit();
  VLOG(4) << "Start BuildInstruction";
  task->BuildInstruction();
  VLOG(2) << "Finish run sub-task, Thread Id : " << std::this_thread::get_id();
}

void ParallelCompiler::LaunchTask() {
  // start sub-task.
  std::vector<std::thread> threads;
  for (int idx = 1; idx < tasks_.size(); ++idx) {
    threads.emplace_back(RunTask, &tasks_[idx]);
  }

  RunTask(&tasks_[0]);
  // syncthreads.
  for (auto& worker : threads) {
    worker.join();
  }
}

std::vector<std::unique_ptr<Instruction>> ParallelCompiler::MergeResult() {
  std::vector<std::unique_ptr<Instruction>> res(graph_->fusion_groups.size());
  for (auto& task : tasks_) {
    for (int idx = 0; idx < task.gidx.size(); ++idx) {
      res[task.gidx[idx]] = std::move(task.instructions[idx]);
    }
  }
  return std::move(res);
}

void ParallelCompiler::Task::Lowering() {
  if (options.lowered_funcs.size()) {
    CHECK_EQ(options.lowered_funcs.size(), graph->fusion_groups.size());
  }
  auto& dtype_dict =
      graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>(
          "inferdtype");
  auto& shape_dict =
      graph->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>(
          "infershape");

  OpLowerer op_lowerer(dtype_dict, shape_dict, target);
  while (true) {
    int idx = compiler->GetGroupIdx();
    if (idx < 0) {
      break;
    }

    gidx.push_back(idx);
    if (options.lowered_funcs.size()) {
      lowered_funcs.push_back(options.lowered_funcs[idx]);
      continue;
    }
    auto& group = graph->fusion_groups[idx];
    VLOG(1) << "Start Lowering Group " << idx << " at "
            << std::this_thread::get_id() << " :\n"
            << "Group " << idx << " {\n"
            << graph->DebugGroupedGraph(group->CollectNodes()) << "}\n";
    lowered_funcs.emplace_back(std::move(op_lowerer.Lower(group)));
    CHECK_EQ(lowered_funcs.back().size(), 1)
        << "Lowerd Function Is Not Equal 1!";
  }
}

void ParallelCompiler::Task::CodegenAndJit() {
  VLOG(2) << "Start Codegen and JIT with Group ["
          << cinn::utils::Join(this->gidx, ", ") << "] at "
          << std::this_thread::get_id();
  // build module
  ir::Module::Builder builder(common::UniqName("module"), target);
  for (auto& func : lowered_funcs) {
    CHECK_EQ(func.size(), 1);
    builder.AddFunction(func[0]);
  }

  auto ir_module = builder.Build();
  // codegen compile
  if (target == common::DefaultNVGPUTarget()) {
#ifdef CINN_WITH_CUDA
    auto splited_module = backends::SplitCudaAndHostModule(ir_module);
    auto hmodule = std::get<0>(splited_module);
    auto dmodule = std::get<1>(splited_module);

    VLOG(3) << "Host Code:\n" << hmodule;
    VLOG(3) << "Device Code:\n" << dmodule;
    backends::CodeGenCUDA_Dev codegen(target);
    auto cuda_c = codegen.Compile(dmodule);
    CHECK(!cuda_c.empty()) << "Compile CUDA C code failed from device module:\n"
                           << dmodule;

    cinn::backends::SourceCodePrint::GetInstance()->write(cuda_c);
    graph->SaveSourceCode(cuda_c);

    using runtime::cuda::CUDAModule;
    backends::nvrtc::Compiler compiler;
    auto ptx = compiler(cuda_c);
    CHECK(!ptx.empty()) << "Compile PTX failed from source code:\n" << cuda_c;
    graph->SavePTXCode(ptx);

    // load cumodule
    cumodule.reset(new CUDAModule(ptx,
                                  compiler.compile_to_cubin()
                                      ? CUDAModule::Kind::CUBIN
                                      : CUDAModule::Kind::PTX));
    // register kernel
    backends::RuntimeSymbols symbols;
    for (auto& fn : dmodule.functions()) {
      auto cufunc = cumodule->GetFunction(0, fn->name);
      CHECK(cufunc);
      symbols.RegisterVar(fn->name + "_ptr_", reinterpret_cast<void*>(cufunc));
    }
    engine = backends::ExecutionEngine::Create(backends::ExecutionOptions(),
                                               std::move(symbols));
    engine->Link<backends::CodeGenCUDA_Host>(hmodule);
#endif
  } else {
    engine = backends::ExecutionEngine::Create(backends::ExecutionOptions());
    engine->Link<backends::CodeGenX86>(ir_module);
  }
}

void ParallelCompiler::Task::BuildInstruction() {
  // create instruction.
  for (int idx : gidx) {
    VLOG(2) << "Start BuildInstruction of Group " << idx << " at "
            << std::this_thread::get_id();
    auto& group = graph->fusion_groups[idx];
    CHECK(group->input_names.size() > 0 || group->output_names.size() > 0);
    auto instr =
        std::unique_ptr<Instruction>(new Instruction(target,
                                                     scope.get(),
                                                     group->input_names,
                                                     group->output_names,
                                                     group->GetFuncName()));

    auto fn_ptr = engine->Lookup(group->GetFuncName());
    CHECK(fn_ptr) << "Can't find jit function : " << group->GetFuncName();
    instr->SetLoweredFunc(reinterpret_cast<void*>(fn_ptr),
                          group->GetFuncName());

    instr->Finalize();
    instructions.push_back(std::move(instr));
  }
}

int ParallelCompiler::GetGroupIdx() {
  std::lock_guard<std::mutex> lock(mtx_);
  if (index < graph_->fusion_groups.size()) {
    return index++;
  } else {
    return -1;
  }
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
