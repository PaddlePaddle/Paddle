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
#include "paddle/cinn/hlir/framework/graph_compiler_util.h"
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
  // init compilation result
  result_.InitCompilationResult(context_->graph->fusion_groups.size());
  // task spilt
  SplitTask();
  // launch task
  LaunchTask();
  // return compilation result
  return std::move(result_);
}

void ParallelCompiler::SplitTask() {
  CHECK(!context_->graph->fusion_groups.empty());
  CHECK(context_->lowered_funcs.empty() ||
        context_->graph->fusion_groups.size() ==
            context_->lowered_funcs.size());
  for (int i = 0; i < context_->graph->fusion_groups.size(); ++i) {
    tasks_.emplace_back(i, this, context_);
  }
}

void ParallelCompiler::RunTask() {
  while (true) {
    int idx = GetTaskIdx();
    if (idx < 0) {
      return;
    }
    VLOG(4) << "Start run task " << idx
            << " on thread: " << std::this_thread::get_id();
    VLOG(4) << "Start lowering on task " << idx;
    tasks_[idx].Lowering();
    if (context_->stage == CompilationStage::LOWERING) {
      VLOG(4) << "Just lowering, finish task " << idx
              << " on thread: " << std::this_thread::get_id();
      return;
    }
    VLOG(4) << "Start CodegenAndJit";
    tasks_[idx].CodegenAndJit();
    if (context_->stage == CompilationStage::CODEGEN_AND_JIT) {
      VLOG(4) << "Just codegen and jit, finish task " << idx
              << " on thread: " << std::this_thread::get_id();
      return;
    }
    VLOG(4) << "Start BuildInstruction";
    tasks_[idx].BuildInstruction();
    if (context_->stage == CompilationStage::BUILD_INSTRUCTION) {
      VLOG(4) << "Just build instruction, finish task " << idx
              << " on thread: " << std::this_thread::get_id();
      return;
    }
    VLOG(4) << "Finish task " << idx
            << " on thread: " << std::this_thread::get_id();
  }
}

void ParallelCompiler::LaunchTask() {
  // multi thread compilation
  std::vector<std::thread> threads;
  VLOG(4) << "Compile with " << FLAGS_cinn_parallel_compile_thread
          << " threads";
  for (int idx = 1; idx < FLAGS_cinn_parallel_compile_thread; ++idx) {
    threads.emplace_back(&ParallelCompiler::RunTask, this);
  }

  RunTask();
  // syncthreads.
  for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));
}

void ParallelCompiler::Task::Lowering() {
  if (!context->lowered_funcs.empty()) {
    CHECK_EQ(context->lowered_funcs.size(),
             context->graph->fusion_groups.size());
    pcompiler->result_.lowered_funcs[group_id] =
        context->lowered_funcs[group_id];
  } else {
    auto& dtype_dict =
        context->graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>(
            "inferdtype");
    auto& shape_dict =
        context->graph
            ->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>(
                "infershape");
    OpLowerer op_lowerer(dtype_dict, shape_dict, context->target);
    auto& group = context->graph->fusion_groups[group_id];
    VLOG(4) << "Start Lowering Group " << group_id << " at "
            << std::this_thread::get_id() << " :\n"
            << "Group " << group_id << " {\n"
            << context->graph->DebugGroupedGraph(group->CollectNodes())
            << "}\n";
    auto lowered_group = op_lowerer.Lower(group);
    CHECK_EQ(lowered_group.size(), 1) << "Lowerd Function Is Not Equal 1!";
    pcompiler->result_.lowered_funcs[group_id] = std::move(lowered_group);
  }
  backends::CompilationInfoDumper::DumpLoweredFuncByGroupIndex(
      pcompiler->result_.lowered_funcs[group_id].front(), group_id);
}

void ParallelCompiler::Task::CodegenAndJit() {
  VLOG(2) << "Start Codegen and JIT on Group " << group_id
          << " at thread: " << std::this_thread::get_id();
  // build module
  ir::Module::Builder builder(common::UniqName("module"), context->target);
  for (auto& func : pcompiler->result_.lowered_funcs[group_id]) {
    builder.AddFunction(func);
  }

  auto ir_module = builder.Build();
  if (context->target == common::DefaultNVGPUTarget()) {
#ifdef CINN_WITH_CUDA
    auto splited_module = backends::SplitCudaAndHostModule(ir_module);
    auto hmodule = std::get<0>(splited_module);
    auto dmodule = std::get<1>(splited_module);

    VLOG(4) << "Host Code:\n" << hmodule;
    VLOG(4) << "Device Code:\n" << dmodule;
    std::string cuda_c;
    if (context->attached_source_code.empty()) {
      backends::CodeGenCUDA_Dev codegen(context->target);
      cuda_c = codegen.Compile(dmodule);
    } else {
      VLOG(4) << "Codegen and jit with attached source code.";
      cuda_c = context->attached_source_code;
    }
    CHECK(!cuda_c.empty()) << "Compile CUDA C code failed from device module:\n"
                           << dmodule;
    backends::CompilationInfoDumper::DumpSourceCodeByGroupIndex(cuda_c,
                                                                group_id);
    pcompiler->result_.source_codes[group_id] = cuda_c;

    cinn::backends::SourceCodePrint::GetInstance()->write(cuda_c);

    using runtime::cuda::CUDAModule;
    backends::nvrtc::Compiler compiler;
    auto ptx = compiler(cuda_c);
    CHECK(!ptx.empty()) << "Compile PTX failed from source code:\n" << cuda_c;
    backends::CompilationInfoDumper::DumpPtxCodeByGroupIndex(ptx, group_id);
    pcompiler->result_.source_ptxs[group_id] = ptx;
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
  VLOG(4) << "Start BuildInstruction of Group " << group_id
          << " at thread: " << std::this_thread::get_id();
  auto& group = context->graph->fusion_groups[group_id];
  CHECK(!group->input_names.empty() || !group->output_names.empty());
  auto instr = std::make_unique<Instruction>(context->target,
                                             context->scope.get(),
                                             group->input_names,
                                             group->output_names,
                                             group->GetFuncName());

  auto fn_ptr = engine->Lookup(group->GetFuncName());
  CHECK(fn_ptr) << "Can't find jit function : " << group->GetFuncName();
  instr->SetLoweredFunc(reinterpret_cast<void*>(fn_ptr), group->GetFuncName());

  instr->Finalize();
  backends::CompilationInfoDumper::DumpInstructionByGroupIndex(instr, group_id);
  pcompiler->result_.instructions[group_id] = std::move(instr);
}

int ParallelCompiler::GetTaskIdx() {
  std::lock_guard<std::mutex> lock(mtx_);
  if (task_idx_ < tasks_.size()) {
    return task_idx_++;
  } else {
    return -1;
  }
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
