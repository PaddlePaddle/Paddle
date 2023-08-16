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
#include "paddle/cinn/runtime/flags.h"

DECLARE_int32(cinn_parallel_compile_thread);

namespace cinn {
namespace hlir {
namespace framework {

CompilationResult ParallelCompiler::operator()() {
  if (context_->graph->fusion_groups.empty()) {
    hlir::framework::ApplyPasses(context_->graph.get(),
                                 {"BuildNonFusedGroupsPass"});
  }
  // Task Spilt
  SplitTask();
  // launch task
  LaunchTask();
  // merge instruction
  return MergeResult();
}

void ParallelCompiler::SplitTask() {
  CHECK(!context_->graph->fusion_groups.empty());
  CHECK(context_->graph->fusion_groups.size() ==
            context_->lowered_funcs.size() ||
        context_->lowered_funcs.empty());
  // Assign fusion_group to each task.
  // The maximum number of tasks is determined by the number of threads.
  // Fusion_group is assigned to tasks in order and continuous.
  int fusion_group_size = context_->graph->fusion_groups.size();
  int thread_size = FLAGS_cinn_parallel_compile_thread > 0
                        ? FLAGS_cinn_parallel_compile_thread
                        : 1;
  int group_per_task =
      (context_->graph->fusion_groups.size() + thread_size - 1) / thread_size;
  for (int idx = 0; idx < context_->graph->fusion_groups.size();
       idx += group_per_task) {
    Task task(this, context_);
    task.start_gidx = idx;
    task.stop_gidx =
        (idx + group_per_task > fusion_group_size ? fusion_group_size
                                                  : idx + group_per_task);
    tasks_.emplace_back(std::move(task));
  }
  VLOG(2) << "Split task to " << tasks_.size() << " sub-task!";
}

void ParallelCompiler::RunTask(ParallelCompiler::Task* task) {
  VLOG(2) << "Stark run sub-task, Thread Id : " << std::this_thread::get_id();
  VLOG(4) << "Start Lowering";
  task->Lowering();
  if (context_->stage == CompilationStage::LOWERING) {
    VLOG(4) << "Just lowering, finish sub task on thread: "
            << std::this_thread::get_id();
    return;
  }
  VLOG(4) << "Start CodegenAndJit";
  task->CodegenAndJit();
  if (context_->stage == CompilationStage::CODEGEN_AND_JIT) {
    VLOG(4) << "Just codegen and jit, finish sub task on thread: "
            << std::this_thread::get_id();
    return;
  }
  VLOG(4) << "Start BuildInstruction";
  task->BuildInstruction();
  if (context_->stage == CompilationStage::BUILD_INSTRUCTION) {
    VLOG(4) << "Just build instruction, finish sub task on thread: "
            << std::this_thread::get_id();
    return;
  }
  VLOG(2) << "Finish run sub-task, Thread Id : " << std::this_thread::get_id();
}

void ParallelCompiler::LaunchTask() {
  // start sub-task.
  std::vector<std::thread> threads;
  for (int idx = 1; idx < tasks_.size(); ++idx) {
    threads.emplace_back(&ParallelCompiler::RunTask, this, &tasks_[idx]);
  }

  RunTask(&tasks_[0]);
  // syncthreads.
  for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));
}

CompilationResult ParallelCompiler::MergeResult() {
  CompilationResult res;
  for (int i = 0; i < tasks_.size(); ++i) {
    auto& task = tasks_[i];
    if (task.status != CompilationStatus::SUCCESS) {
      LOG(WARNING) << "Compile fail on task " << i << ", corresponds to group "
                   << task.start_gidx << " - " << task.stop_gidx;
      res.status = task.status;
      res.message = task.message;
      return std::move(res);
    }
    for (auto& lowered_func : task.lowered_funcs) {
      res.lowered_funcs.emplace_back(lowered_func);
    }
    for (auto& source_code : task.source_codes) {
      res.source_codes.emplace_back(source_code);
    }
    for (auto& source_ptx : task.source_ptxs) {
      res.source_ptxs.emplace_back(source_ptx);
    }
    for (auto& instruction : task.instructions) {
      res.instructions.emplace_back(std::move(instruction));
    }
  }
  return res;
}

void ParallelCompiler::Task::Lowering() {
  if (!context->lowered_funcs.empty()) {
    CHECK_EQ(context->lowered_funcs.size(),
             context->graph->fusion_groups.size());
  }
  auto& dtype_dict =
      context->graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>(
          "inferdtype");
  auto& shape_dict =
      context->graph
          ->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>(
              "infershape");

  OpLowerer op_lowerer(dtype_dict, shape_dict, context->target);
  for (int idx = start_gidx; idx < stop_gidx; ++idx) {
    if (!context->lowered_funcs.empty()) {
      lowered_funcs.push_back(context->lowered_funcs[idx]);
      continue;
    }
    auto& group = context->graph->fusion_groups[idx];
    VLOG(1) << "Start Lowering Group " << idx << " at "
            << std::this_thread::get_id() << " :\n"
            << "Group " << idx << " {\n"
            << context->graph->DebugGroupedGraph(group->CollectNodes())
            << "}\n";
    auto lowered_group = op_lowerer.Lower(group);
    CHECK_EQ(lowered_group.size(), 1) << "Lowerd Function Is Not Equal 1!";
    lowered_funcs.emplace_back(std::move(lowered_group));
  }
}

void ParallelCompiler::Task::CodegenAndJit() {
  VLOG(2) << "Start Codegen and JIT with Group [" << start_gidx << "-"
          << stop_gidx << ") at thread" << std::this_thread::get_id();
  // build module
  ir::Module::Builder builder(common::UniqName("module"), context->target);
  for (auto& func : lowered_funcs) {
    CHECK_EQ(func.size(), 1);
    builder.AddFunction(func[0]);
  }

  auto ir_module = builder.Build();
  if (context->target == common::DefaultNVGPUTarget()) {
#ifdef CINN_WITH_CUDA
    auto splited_module = backends::SplitCudaAndHostModule(ir_module);
    auto hmodule = std::get<0>(splited_module);
    auto dmodule = std::get<1>(splited_module);

    VLOG(3) << "Host Code:\n" << hmodule;
    VLOG(3) << "Device Code:\n" << dmodule;
    backends::CodeGenCUDA_Dev codegen(context->target);
    auto cuda_c = codegen.Compile(dmodule);
    CHECK(!cuda_c.empty()) << "Compile CUDA C code failed from device module:\n"
                           << dmodule;
    source_codes.emplace_back(cuda_c);

    cinn::backends::SourceCodePrint::GetInstance()->write(cuda_c);

    using runtime::cuda::CUDAModule;
    backends::nvrtc::Compiler compiler;
    auto ptx = compiler(cuda_c);
    CHECK(!ptx.empty()) << "Compile PTX failed from source code:\n" << cuda_c;
    source_ptxs.emplace_back(ptx);
    // load cumodule
    cumodule = std::make_unique<CUDAModule>(ptx,
                                            compiler.compile_to_cubin()
                                                ? CUDAModule::Kind::CUBIN
                                                : CUDAModule::Kind::PTX);

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
  for (int idx = start_gidx; idx < stop_gidx; ++idx) {
    VLOG(2) << "Start BuildInstruction of Group " << idx << " at "
            << std::this_thread::get_id();
    auto& group = context->graph->fusion_groups[idx];
    CHECK(!group->input_names.empty() || !group->output_names.empty());
    auto instr = std::make_unique<Instruction>(context->target,
                                               context->scope.get(),
                                               group->input_names,
                                               group->output_names,
                                               group->GetFuncName());

    auto fn_ptr = engine->Lookup(group->GetFuncName());
    CHECK(fn_ptr) << "Can't find jit function : " << group->GetFuncName();
    instr->SetLoweredFunc(reinterpret_cast<void*>(fn_ptr),
                          group->GetFuncName());

    instr->Finalize();
    instructions.push_back(std::move(instr));
  }
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
