// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/backends/compiler.h"

#include <fstream>

#include "paddle/cinn/backends/llvm/runtime_symbol_registry.h"
#include "paddle/cinn/common/context.h"
#include "paddle/cinn/hlir/framework/graph_compiler_util.h"
#include "paddle/cinn/hlir/framework/visualize_helper.h"
#include "paddle/cinn/ir/utils/ir_printer.h"
#ifdef CINN_WITH_CUDA
#include "paddle/cinn/backends/codegen_cuda_dev.h"
#include "paddle/cinn/backends/codegen_cuda_host.h"
#include "paddle/cinn/backends/codegen_cuda_util.h"
#include "paddle/cinn/backends/nvrtc/nvrtc_util.h"
#include "paddle/cinn/runtime/cuda/cuda_module.h"
#include "paddle/cinn/runtime/cuda/cuda_util.h"
#include "paddle/cinn/runtime/flags.h"
#endif

PD_DECLARE_string(cinn_source_code_save_path);
PD_DECLARE_string(cinn_dump_group_lowered_func);
PD_DECLARE_string(cinn_dump_group_source_code);
PD_DECLARE_string(cinn_dump_group_ptx);
PD_DECLARE_string(cinn_dump_group_instruction);

namespace cinn {
namespace backends {
using ir::Module;
using CompilationStatus = hlir::framework::CompilationStatus;

static constexpr int DebugLogMaxLen = 30000;

void CompilationInfoDumper::DumpLoweredFuncByGroupIndex(
    const ir::LoweredFunc& lowered_func, const int gidx, const int device_id) {
  if (FLAGS_cinn_dump_group_lowered_func.empty() ||
      lowered_func.get() == nullptr) {
    return;
  }
  std::stringstream content;
  content << lowered_func;
  Dump(FLAGS_cinn_dump_group_lowered_func,
       gidx,
       device_id,
       "lowered_function.txt",
       content.str());
}

void CompilationInfoDumper::DumpSourceCodeByGroupIndex(
    const std::string& source_code, const int gidx, const int device_id) {
  if (FLAGS_cinn_dump_group_source_code.empty()) {
    return;
  }
  Dump(FLAGS_cinn_dump_group_source_code,
       gidx,
       device_id,
       "source_code.cu",
       source_code);
}

void CompilationInfoDumper::DumpPtxCodeByGroupIndex(
    const std::string& source_ptx, const int gidx, const int device_id) {
  if (FLAGS_cinn_dump_group_ptx.empty()) {
    return;
  }
  Dump(
      FLAGS_cinn_dump_group_ptx, gidx, device_id, "source_ptx.ptx", source_ptx);
}

void CompilationInfoDumper::DumpInstructionByGroupIndex(
    const std::unique_ptr<cinn::hlir::framework::Instruction>& instr,
    const int gidx,
    const int device_id) {
  if (FLAGS_cinn_dump_group_instruction.empty() || instr.get() == nullptr) {
    return;
  }
  Dump(FLAGS_cinn_dump_group_instruction,
       gidx,
       device_id,
       "instruction.txt",
       instr->DumpInstruction());
}

void CompilationInfoDumper::DumpLoweredFunc() {
  if (FLAGS_cinn_dump_group_lowered_func.empty()) {
    return;
  }
  for (int idx = 0; idx < info_.Size(); ++idx) {
    std::stringstream content;
    if (info_.Status(idx) > CompilationStatus::LOWERING_FAIL) {
      content << info_.LoweredFuncs(idx).front();
    } else {
      content << "[No lowered func generated]\n\n" << info_.Message(idx);
    }
    Dump(FLAGS_cinn_dump_group_lowered_func,
         idx,
         device_id_,
         "lowered_function.txt",
         content.str());
  }
}

void CompilationInfoDumper::DumpSourceCode() {
  if (FLAGS_cinn_dump_group_source_code.empty()) {
    return;
  }
  for (int idx = 0; idx < info_.Size(); ++idx) {
    std::string dump_str;
    if (info_.Status(idx) > CompilationStatus::CODEGEN_JIT_FAIL) {
      dump_str = info_.SourceCode(idx);
    } else {
      dump_str = "[No source code generated]\n\n" + info_.Message(idx);
    }
    Dump(FLAGS_cinn_dump_group_source_code,
         idx,
         device_id_,
         "source_code.cu",
         dump_str);
  }
}

void CompilationInfoDumper::DumpPtxCode() {
  if (FLAGS_cinn_dump_group_ptx.empty()) {
    return;
  }
  for (int idx = 0; idx < info_.Size(); ++idx) {
    std::string dump_str;
    if (info_.Status(idx) > CompilationStatus::CODEGEN_JIT_FAIL) {
      dump_str = info_.SourcePtx(idx);
    } else {
      dump_str = "[No source ptxs generated]\n\n" + info_.Message(idx);
    }
    Dump(
        FLAGS_cinn_dump_group_ptx, idx, device_id_, "source_ptx.ptx", dump_str);
  }
}

void CompilationInfoDumper::DumpInstruction() {
  if (FLAGS_cinn_dump_group_instruction.empty()) {
    return;
  }
  for (int idx = 0; idx < info_.RuntimeInstructions().size(); ++idx) {
    std::string dump_str;
    if (info_.RuntimeInstruction(idx).get() != nullptr) {
      dump_str = info_.RuntimeInstruction(idx)->DumpInstruction();
    } else {
      dump_str = "[No instruction generated]\n\n" + info_.Message(idx);
    }
    Dump(FLAGS_cinn_dump_group_instruction,
         idx,
         device_id_,
         "instruction.txt",
         dump_str);
  }
}

void CompilationInfoDumper::Dump(const std::string& base_path,
                                 const int idx,
                                 const int device_id,
                                 const std::string& file_name,
                                 const std::string& content) {
  auto dump_path = utils::StringFormat(
      "%s/device_%d/fusion_group_%d", base_path.c_str(), device_id, idx);
  if (!hlir::framework::MakeDirectory(
          dump_path, S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH)) {
    LOG(WARNING) << "Failed to make directory: \"" << dump_path
                 << "\", the instruction for this group will not dump.";
  } else {
    auto dump_file =
        utils::StringFormat("%s/%s", dump_path.c_str(), file_name.c_str());
    VLOG(7) << "Dump instruction to: " << dump_file;
    std::ofstream of(dump_file, std::ios_base::out);
    if (of.is_open()) {
      of << content;
      of.close();
    } else {
      LOG(WARNING) << "Failed to open file: " << dump_file
                   << ", please check your path.";
    }
  }
}

SourceCodePrint::SourceCodePrint() {
  if (!FLAGS_cinn_source_code_save_path.empty()) {
    LOG(INFO)
        << "The CINN auto generated source code will writing into file: \""
        << FLAGS_cinn_source_code_save_path << "\"";
    of.open(FLAGS_cinn_source_code_save_path, std::ios_base::out);
  }
}

SourceCodePrint::~SourceCodePrint() {
  if (of.is_open()) {
    of.close();
  }
}

void SourceCodePrint::write(const std::string& source_code) {
  std::lock_guard<std::mutex> guard(mtx_);
  if (of.is_open()) {
    of << source_code << std::endl;
  } else if (!FLAGS_cinn_source_code_save_path.empty()) {
    LOG(WARNING) << "Failed to open \"" << FLAGS_cinn_source_code_save_path
                 << "\", source code will print.";
    if (source_code.size() > DebugLogMaxLen) {
      LOG(INFO) << "[CUDA] source code-0:\n"
                << source_code.substr(0, DebugLogMaxLen);
      for (int i = 1; i * DebugLogMaxLen < source_code.size(); ++i) {
        LOG(INFO) << "[CUDA] source code-" << i << ":\n"
                  << source_code.substr(DebugLogMaxLen * i, DebugLogMaxLen);
      }
    } else {
      LOG(INFO) << "[CUDA] source code:\n" << source_code;
    }
  }
}

void Compiler::Build(const Module& module, const std::string& code) {
  if (target_.arch == Target::Arch::NVGPU) {
    CompileCudaModule(module, code);
  } else if (target_.arch == Target::Arch::X86) {
    CompileX86Module(module);
  } else {
    CINN_NOT_IMPLEMENTED
  }
}

std::string Compiler::GetSourceCode(const ir::Module& module) {
  if (target_.arch == Target::Arch::NVGPU) {
#ifdef CINN_WITH_CUDA
    auto _host_module_device_module_ =
        SplitCudaAndHostModule(module);  // NOLINT
    auto& host_module = std::get<0>(_host_module_device_module_);
    auto& device_module = std::get<1>(_host_module_device_module_);
    CodeGenCUDA_Dev codegen(target_);
    auto source_code = codegen.Compile(device_module);
    return source_code;
#else
    CINN_NOT_IMPLEMENTED
#endif
  } else {
    CINN_NOT_IMPLEMENTED
  }
}

void Compiler::BuildDefault(const Module& module) {
  if (target_.arch == Target::Arch::NVGPU) {
    CompileCudaModule(module);
  } else if (target_.arch == Target::Arch::X86) {
    CompileX86Module(module);
  } else {
    CINN_NOT_IMPLEMENTED
  }
}

void Compiler::CompileCudaModule(const Module& module,
                                 const std::string& code) {
#ifdef CINN_WITH_CUDA
  auto _host_module_device_module_ = SplitCudaAndHostModule(module);  // NOLINT
  auto& host_module = std::get<0>(_host_module_device_module_);
  auto& device_module = std::get<1>(_host_module_device_module_);
  VLOG(3) << "[CUDA] host module:\n" << host_module;

  VLOG(3) << "[CUDA] device module:\n" << device_module;
  std::string source_code;
  if (code.empty()) {
    CodeGenCUDA_Dev codegen(target_);
    source_code = codegen.Compile(device_module);
  } else {
    source_code = code;
  }
  CHECK(!source_code.empty())
      << "Compile CUDA C code failed from device module:\n"
      << device_module;
  VLOG(3) << "[CUDA] C:\n" << source_code;
  SourceCodePrint::GetInstance()->write(source_code);
  using runtime::cuda::CUDAModule;

  nvrtc::Compiler compiler;
  auto ptx = compiler(source_code);
  CHECK(!ptx.empty()) << "Compile PTX failed from source code:\n"
                      << source_code;
  cuda_module_.reset(new CUDAModule(ptx,
                                    compiler.compile_to_cubin()
                                        ? CUDAModule::Kind::CUBIN
                                        : CUDAModule::Kind::PTX));

  RuntimeSymbols symbols;
  for (auto& fn : device_module.functions()) {
    std::string kernel_fn_name = fn->name;
    auto fn_kernel = cuda_module_->GetFunction(0, kernel_fn_name);
    CHECK(fn_kernel);

    symbols.RegisterVar(kernel_fn_name + "_ptr_",
                        reinterpret_cast<void*>(fn_kernel));
  }

  engine_ = ExecutionEngine::Create(ExecutionOptions(), std::move(symbols));
  engine_->Link<CodeGenCUDA_Host>(host_module);

#else
  CINN_NOT_IMPLEMENTED
#endif
}

void Compiler::CompileX86Module(const Module& module) {
  engine_->Link<CodeGenX86>(module);
}

void Compiler::ExportObject(const std::string& path) {
  engine_->ExportObject(path);
}

void* Compiler::Lookup(absl::string_view fn_name) {
  CHECK(engine_);
  if (engine_->Lookup(fn_name) != nullptr) {
    return engine_->Lookup(fn_name);
  }
  return nullptr;
}

}  // namespace backends
}  // namespace cinn
