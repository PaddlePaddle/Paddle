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

#include "cinn/backends/compiler.h"

#include <fstream>

#include "cinn/backends/llvm/runtime_symbol_registry.h"
#include "cinn/common/context.h"
#ifdef CINN_WITH_CUDA
#include "cinn/backends/codegen_cuda_dev.h"
#include "cinn/backends/codegen_cuda_host.h"
#include "cinn/backends/codegen_cuda_util.h"
#include "cinn/backends/nvrtc/nvrtc_util.h"
#include "cinn/runtime/cuda/cuda_module.h"
#include "cinn/runtime/cuda/cuda_util.h"
#endif

DECLARE_string(cinn_source_code_save_path);

namespace cinn {
namespace backends {
using ir::Module;

static constexpr int DebugLogMaxLen = 30000;

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

  backends::nvrtc::Compiler compiler;

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
