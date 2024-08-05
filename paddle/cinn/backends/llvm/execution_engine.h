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

#include <llvm/ADT/StringMap.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/ObjectCache.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/LambdaResolver.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SmallVectorMemoryBuffer.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>

#include <functional>
#include <memory>
#include <mutex>  // NOLINT
#include <optional>
#include <string>
#include <vector>

#include "paddle/cinn/backends/llvm/codegen_x86.h"
#include "paddle/cinn/backends/llvm/llvm_util.h"
#include "paddle/cinn/backends/llvm/runtime_symbol_registry.h"
#include "paddle/cinn/ir/module.h"

namespace cinn::backends {

class NaiveObjectCache : public llvm::ObjectCache {
 public:
  void notifyObjectCompiled(const llvm::Module *,
                            llvm::MemoryBufferRef) override;
  std::unique_ptr<llvm::MemoryBuffer> getObject(const llvm::Module *) override;

 private:
  llvm::StringMap<std::unique_ptr<llvm::MemoryBuffer>> cached_objects_;
};

struct ExecutionOptions {
  int opt_level{3};
  bool enable_debug_info{false};
  // TODO(fc500110)
  // int num_compile_threads{1};
  // bool enable_fast_math;
};

class ExecutionEngine {
 public:
  static std::unique_ptr<ExecutionEngine> Create(
      const ExecutionOptions &config);

  void *Lookup(absl::string_view name);

  template <typename CodeGenT = CodeGenLLVM>
  void Link(const ir::Module &module);

  void ExportObject(const std::string &path);

  bool AddModule(std::unique_ptr<llvm::Module> module,
                 std::unique_ptr<llvm::LLVMContext> context);

  void RegisterModuleRuntimeSymbols(RuntimeSymbols &&module_symbols);

  bool AddSelfModule();

 protected:
  explicit ExecutionEngine(bool enable_object_cache)
      : cache_(std::make_unique<NaiveObjectCache>()),
        ctx(std::make_unique<llvm::LLVMContext>()),
        b(std::make_unique<llvm::IRBuilder<>>(*ctx)) {}

  void RegisterGlobalRuntimeSymbols();

  bool SetupTargetTriple(llvm::Module *module);

  // This may not be a compatible implementation.
  friend std::unique_ptr<ExecutionEngine> std::make_unique<ExecutionEngine>(
      bool &&);

 private:
  mutable std::mutex mu_;
  llvm::SmallString<0> buffer_;
  std::unique_ptr<llvm::orc::LLJIT> jit_;
  std::unique_ptr<NaiveObjectCache> cache_;
  RuntimeSymbols module_symbols_;

  std::unique_ptr<llvm::LLVMContext> ctx;
  std::unique_ptr<llvm::Module> m;
  std::unique_ptr<llvm::IRBuilder<>> b;
};

}  // namespace cinn::backends
