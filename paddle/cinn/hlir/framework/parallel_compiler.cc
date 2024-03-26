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
#include "paddle/cinn/hlir/framework/compile_error.h"
#include "paddle/cinn/hlir/framework/graph_compiler_util.h"
#include "paddle/cinn/hlir/framework/op_lowering.h"
#include "paddle/cinn/hlir/framework/pass.h"
#include "paddle/cinn/ir/module.h"
#include "paddle/cinn/runtime/flags.h"

#ifdef CINN_WITH_SYCL
#include "paddle/cinn/backends/sycl/codegen_sycl_dev.h"
#include "paddle/cinn/backends/sycl/compiler_sycl.h"
#endif

#ifdef CINN_WITH_ROCM
#include "paddle/cinn/backends/hip/codegen_hip_dev.h"
#include "paddle/cinn/backends/hip/compiler_hip.h"
#endif

PD_DECLARE_int32(cinn_parallel_compile_thread);

namespace cinn {
namespace hlir {
namespace framework {

/** \brief A macro that guards the beginning of each step of compiling
 */
#define CINN_COMPILE_STEP_BEGIN() try {
/**
 * \brief A macro that pairs with `CINN_COMPILE_STEP_BEGIN`, handling potential
 * errors and error message recording.
 * @param err_msg_level A ScheduleErrorMessageLevel enum, level of error message
 * printing
 */
#define CINN_COMPILE_STEP_END(err_msg_level, idx)                              \
  }                                                                            \
  catch (const CompileErrorHandler& err_handler) {                             \
    std::string err_msg = err_handler.FormatErrorMessage(err_msg_level);       \
    err_msg =                                                                  \
        "Group Idx: " + std::to_string(idx) + ",  Compile Error.\n" + err_msg; \
    LOG(WARNING) << "\n" << err_msg;                                           \
    result_.SetMessage(idx, err_msg);                                          \
    result_.SetStatus(idx, err_handler.Status());                              \
    continue;                                                                  \
  }

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
  int device_id = 0;
  if (context_->target.arch_is_gpu()) {
    using cinn::runtime::BackendAPI;
    device_id = BackendAPI::get_backend(context_->target)->get_device();
  }
  // set max_group_num_of_one_task
  int max_group_num_of_one_task = 1;
  if (context_->target.language == Target::Language::sycl) {
    max_group_num_of_one_task = 100;
  }
  for (int start_group_id = 0;
       start_group_id < context_->graph->fusion_groups.size();
       start_group_id += max_group_num_of_one_task) {
    int groups_num =
        std::min(max_group_num_of_one_task,
                 static_cast<int>(context_->graph->fusion_groups.size() -
                                  start_group_id));
    std::vector<int> group_ids;
    for (int i = 0; i < groups_num; i++) {
      group_ids.push_back(start_group_id + i);
    }
    for (int group_id : group_ids) {
      tasks_.emplace_back(device_id, group_id, this, context_);
    }
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
    CINN_COMPILE_STEP_BEGIN();
    tasks_[idx].Lowering();
    CINN_COMPILE_STEP_END(err_msg_level_, idx);
    if (context_->stage == CompilationStage::LOWERING) {
      VLOG(4) << "Just lowering, finish task " << idx
              << " on thread: " << std::this_thread::get_id();
      continue;
    }
    VLOG(4) << "Start CodegenAndJit";
    CINN_COMPILE_STEP_BEGIN();
    tasks_[idx].CodegenAndJit();
    CINN_COMPILE_STEP_END(err_msg_level_, idx);
    if (context_->stage == CompilationStage::CODEGEN_AND_JIT) {
      VLOG(4) << "Just codegen and jit, finish task " << idx
              << " on thread: " << std::this_thread::get_id();
      continue;
    }
    VLOG(4) << "Start BuildInstruction";
    CINN_COMPILE_STEP_BEGIN();
    tasks_[idx].BuildInstruction();
    CINN_COMPILE_STEP_END(err_msg_level_, idx);
    if (context_->stage == CompilationStage::BUILD_INSTRUCTION) {
      VLOG(4) << "Just build instruction, finish task " << idx
              << " on thread: " << std::this_thread::get_id();
      continue;
    }
    VLOG(4) << "Finish task " << idx
            << " on thread: " << std::this_thread::get_id();
  }
}

void ParallelCompiler::LaunchTask() {
  int device_id = 0;
  if (context_->target.arch_is_gpu()) {
    using cinn::runtime::BackendAPI;
    device_id = BackendAPI::get_backend(context_->target)->get_device();
  }
  int num_threads = FLAGS_cinn_parallel_compile_thread;
#if defined(PADDLE_WITH_DISTRIBUTE)
  if (device_id > 0) {
    num_threads = 1;
  }
#endif
  // multi thread compilation
  std::vector<std::thread> threads;
  VLOG(4) << "Compile with " << num_threads << " threads";
  for (int idx = 1; idx < num_threads; ++idx) {
    threads.emplace_back(&ParallelCompiler::RunTask, this);
  }

  RunTask();
  // syncthreads.
  for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));
}

void ParallelCompiler::Task::Lowering() {
  if (!context->lowered_funcs.empty()) {
    if (context->lowered_funcs.size() != context->graph->fusion_groups.size()) {
      std::ostringstream err_msg;
      err_msg << "The number of LoweredFuncs attached differs from the number "
                 "of Groups, LoweredFuncs size = "
              << context->lowered_funcs.size()
              << " Groups size = " << context->graph->fusion_groups.size()
              << "\n";
      std::ostringstream detail_info;
      detail_info << "LoweredFuncs:\n";
      for (const std::vector<ir::LoweredFunc>& funcs : context->lowered_funcs) {
        detail_info << funcs[0] << "\n";
      }
      detail_info << "Groups:\n";
      for (const std::shared_ptr<Graph::Group>& group :
           context->graph->fusion_groups) {
        detail_info << context->graph->DebugGroupedGraph(group->CollectNodes())
                    << "\n";
      }
      throw CompileErrorHandler(CompilationStatus::LOWERING_FAIL,
                                err_msg.str(),
                                detail_info.str(),
                                __FILE__,
                                __LINE__);
    }
    pcompiler->result_.SetLoweredFuncs(group_id,
                                       context->lowered_funcs[group_id]);
  } else {
    auto& dtype_dict =
        context->graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>(
            "inferdtype");
    auto& shape_dict =
        context->graph
            ->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>(
                "infershape");
    auto op_lowerer = CreateOpLowerer(dtype_dict, shape_dict, context->target);
    auto& group = context->graph->fusion_groups[group_id];
    VLOG(4) << "Start Lowering Group " << group_id << " at "
            << std::this_thread::get_id() << " :\n"
            << "Group " << group_id << " {\n"
            << context->graph->DebugGroupedGraph(group->CollectNodes())
            << "}\n";
    // std::cout <<"group "<< group_id << "\n";
    // std::cout << group->CollectNodes()[0]->op()->name << ":\n";
    // LOG(INFO) << group->CollectNodes()[0]->op()->name;
    auto lowered_funcs = op_lowerer.Lower(group);
    // std::cout << lowered_funcs[0]->body << "\n";
    // LOG(INFO) << lowered_funcs[0]->body;
    if (lowered_funcs.size() != 1) {
      std::ostringstream err_msg;
      err_msg << "Lowering Group: " << group_id
              << ", the number of LoweredFuncs is not equal 1, but "
              << lowered_funcs.size()
              << "\nOur current principle is to generate 1 LoweredFunc for "
                 "each Group"
              << "\n";
      std::ostringstream detail_info;
      detail_info << "LoweredFuncs:\n";
      for (const ir::LoweredFunc& func : lowered_funcs) {
        detail_info << func << "\n";
      }
      detail_info << "Group:\n";
      detail_info << context->graph->DebugGroupedGraph(group->CollectNodes())
                  << "\n";
      throw CompileErrorHandler(CompilationStatus::LOWERING_FAIL,
                                err_msg.str(),
                                detail_info.str(),
                                __FILE__,
                                __LINE__);
    }
    pcompiler->result_.SetLoweredFuncs(group_id, lowered_funcs);
  }

  backends::CompilationInfoDumper::DumpLoweredFuncByGroupIndex(
      pcompiler->result_.LoweredFuncs(group_id).front(), group_id, device_id);
}

void ParallelCompiler::Task::CodegenAndJit() {
  // build module
  ir::Module::Builder builder(common::UniqName("module"), context->target);
  for (auto& func : pcompiler->result_.LoweredFuncs(group_id)) {
    builder.AddFunction(func);
  }
  auto ir_module = builder.Build();
  if (context->target == cinn::common::DefaultNVGPUTarget()) {
#ifdef CINN_WITH_CUDA
    auto splited_module =
        backends::SplitDeviceAndHostModule(ir_module, context->target);
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
    // backends::CompilationInfoDumper::DumpSourceCodeByGroupIndex(
    //     cuda_c, group_id, device_id);
    // pcompiler->result_.SetSourceCode(group_id, cuda_c);

    cinn::backends::SourceCodePrint::GetInstance()->write(cuda_c);
    VLOG(4) << "[CUDA]:\n" << cuda_c;
    using runtime::cuda::CUDAModule;
    backends::nvrtc::Compiler compiler;
    auto ptx = compiler(cuda_c);
    CHECK(!ptx.empty()) << "Compile PTX failed from source code:\n" << cuda_c;
    // backends::CompilationInfoDumper::DumpPtxCodeByGroupIndex(
    //     ptx, group_id, device_id);
    // pcompiler->result_.SetSourcePtx(group_id, ptx);
    //  load cumodule
    cumodule = std::make_unique<CUDAModule>(ptx,
                                            compiler.compile_to_cubin()
                                                ? CUDAModule::Kind::CUBIN
                                                : CUDAModule::Kind::PTX);

    // register kernel
    backends::RuntimeSymbols symbols;
    for (auto& fn : dmodule.functions()) {
      auto cufunc = cumodule->GetFunction(device_id, fn->name);
      CHECK(cufunc);
      symbols.RegisterVar(fn->name + "_ptr_", reinterpret_cast<void*>(cufunc));
    }
    engine = backends::ExecutionEngine::Create(backends::ExecutionOptions(),
                                               std::move(symbols));
    engine->Link<backends::CodeGenCUDA_Host>(hmodule);
#endif
  } else if (context->target.language == Target::Language::sycl) {
#ifdef CINN_WITH_SYCL
    auto splited_module =
        backends::SplitDeviceAndHostModule(ir_module, context->target);
    auto host_module = std::get<0>(splited_module);
    auto device_module = std::get<1>(splited_module);
    backends::CodeGenSYCL_Dev codegen(context->target);
    std::string source_code = codegen.Compile(device_module);
    CHECK(!source_code.empty())
        << "Compile SYCL code failed from device module:\n"
        << device_module;
    VLOG(4) << "[SYCL]:\n" << source_code;
    cinn::backends::SourceCodePrint::GetInstance()->write(source_code);
    using runtime::Sycl::SYCLModule;
    backends::syclrtc::Compiler compiler;
    std::string share_library = compiler(source_code, context->target.arch);
    CHECK(!share_library.empty())
        << "Compile SYCL code failed from source code" << source_code;
    sycl_module = std::make_unique<SYCLModule>(
        source_code, share_library, SYCLModule::Kind::so);
    // register kernel
    backends::RuntimeSymbols symbols;
    for (auto& fn : device_module.functions()) {
      auto cufunc = sycl_module->GetFunction(fn->name);
      CHECK(cufunc);
      symbols.RegisterVar(fn->name + "_ptr_", reinterpret_cast<void*>(cufunc));
    }
    engine = backends::ExecutionEngine::Create(backends::ExecutionOptions(),
                                               std::move(symbols));
    engine->Link<backends::CodeGenCUDA_Host>(host_module);
#endif
  } else if (context->target.language == Target::Language::hip) {
#ifdef CINN_WITH_ROCM
    auto splited_module =
        backends::SplitDeviceAndHostModule(ir_module, context->target);
    auto host_module = std::get<0>(splited_module);
    auto device_module = std::get<1>(splited_module);
    VLOG(4) << "Host Code:\n" << host_module;
    VLOG(4) << "Device Code:\n" << device_module;
    std::string hip_c;
    if (context->attached_source_code.empty()) {
      backends::CodeGenHIP_Dev codegen(context->target);
      hip_c = codegen.Compile(device_module);
    } else {
      VLOG(4) << "Codegen and jit with attached source code.";
      hip_c = context->attached_source_code;
    }
    CHECK(!hip_c.empty()) << "Compile HIP C code failed from device module:\n"
                          << device_module;

    cinn::backends::SourceCodePrint::GetInstance()->write(hip_c);
    VLOG(4) << "[HIP]:\n" << hip_c;
    using runtime::hip::HIPModule;
    backends::hiprtc::Compiler compiler;
    auto hsaco = compiler(hip_c);
    CHECK(!hsaco.empty()) << "Compile hsaco failed from source code:\n"
                          << hip_c;
    // backends::CompilationInfoDumper::DumpPtxCodeByGroupIndex(
    //     ptx, group_id, device_id);
    // pcompiler->result_.SetSourcePtx(group_id, ptx);
    //  load cumodule
    hip_module = std::make_unique<HIPModule>(hsaco,
                                             compiler.compile_to_hipbin()
                                                 ? HIPModule::Kind::HIPBIN
                                                 : HIPModule::Kind::GCN);
    // register kernel
    backends::RuntimeSymbols symbols;
    for (auto& fn : device_module.functions()) {
      auto hip_func = hip_module->GetFunction(device_id, fn->name);
      CHECK(hip_func);
      symbols.RegisterVar(fn->name + "_ptr_",
                          reinterpret_cast<void*>(hip_func));
    }
    engine = backends::ExecutionEngine::Create(backends::ExecutionOptions(),
                                               std::move(symbols));
    engine->Link<backends::CodeGenCUDA_Host>(host_module);
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
  backends::CompilationInfoDumper::DumpInstructionByGroupIndex(
      instr, group_id, device_id);
  pcompiler->result_.SetInstruction(group_id, std::move(instr));
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
