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

// This header intentionally only depends on LLVM headers (no CINN / paddle
// headers). NaiveObjectCache subclasses llvm::ObjectCache and the compiler
// helper constructs llvm::orc::SimpleCompiler; both are polymorphic LLVM types.
// xtrans's libLLVM-15 is built with RTTI OFF and therefore does not export
// typeinfo for these classes. Definitions that emit vtables/typeinfo for them
// (NaiveObjectCache's out-of-line virtuals, the SimpleCompiler construction)
// live in object_cache_rtti_off.cc which is compiled with -fno-rtti under
// WITH_XPU_CADA, so no LLVM typeinfo is required at link time.

#include <llvm/ADT/StringMap.h>
#include <llvm/ExecutionEngine/ObjectCache.h>
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>

#include <memory>

namespace cinn::backends {

class NaiveObjectCache : public llvm::ObjectCache {
 public:
  void notifyObjectCompiled(const llvm::Module *,
                            llvm::MemoryBufferRef) override;
  std::unique_ptr<llvm::MemoryBuffer> getObject(const llvm::Module *) override;

 private:
  llvm::StringMap<std::unique_ptr<llvm::MemoryBuffer>> cached_objects_;
};

// Factory that builds a TargetMachine from `jtmb` and returns an
// llvm::orc::IRCompiler (a TMOwningSimpleCompiler) wired to `cache`.
// Defined in object_cache_rtti_off.cc so the SimpleCompiler vtable/typeinfo is
// emitted in a -fno-rtti translation unit under WITH_XPU_CADA.
llvm::Expected<std::unique_ptr<llvm::orc::IRCompileLayer::IRCompiler>>
CreateObjectCacheCompiler(llvm::orc::JITTargetMachineBuilder jtmb,
                          llvm::ObjectCache *cache);

}  // namespace cinn::backends
