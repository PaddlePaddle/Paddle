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

#pragma once

#include <absl/strings/string_view.h>
#include <llvm/AsmParser/Parser.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/LambdaResolver.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SmallVectorMemoryBuffer.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "paddle/cinn/backends/llvm/codegen_llvm.h"
#include "paddle/cinn/backends/llvm/llvm_util.h"
#include "paddle/cinn/backends/llvm/runtime_symbol_registry.h"
#include "paddle/cinn/ir/module.h"
#include "paddle/cinn/runtime/intrinsic.h"

namespace cinn {
namespace backends {

class SimpleJIT {
 public:
  static std::unique_ptr<SimpleJIT> Create() {
    return std::unique_ptr<SimpleJIT>(new SimpleJIT);
  }

  /**
   * Runtime link to a module.
   * @tparam CodeGenT a CodeGenLLVM implementation.
   * @param module a CINN module.
   * @param optimize whether to optimize.
   */
  template <typename CodeGenT = CodeGenLLVM>
  void Link(ir::Module module, bool optimize = true);

  void Link(llvm::orc::ThreadSafeModule m, bool optimize = true) {
    llvm::cantFail(jit_->addIRModule(std::move(m)));
  }

  llvm::JITTargetAddress Lookup(absl::string_view name) {
    return llvm::cantFail(jit_->lookup(AsStringRef(name))).getAddress();
  }

 private:
  void AddModule(std::unique_ptr<llvm::Module> module, bool optimize);

  llvm::LLVMContext &context() { return *context_.getContext(); }

  SimpleJIT();

  std::unique_ptr<llvm::orc::LLJIT> jit_;
  llvm::orc::ThreadSafeContext context_;
};

}  // namespace backends
}  // namespace cinn
