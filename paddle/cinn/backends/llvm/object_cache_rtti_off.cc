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

// This translation unit holds the definitions that emit vtables/typeinfo for
// polymorphic LLVM classes (llvm::ObjectCache via NaiveObjectCache, and
// llvm::orc::SimpleCompiler via TMOwningSimpleCompiler). Under WITH_XPU_CADA it
// is compiled with -fno-rtti so that no LLVM typeinfo symbols are required when
// linking against xtrans's RTTI-OFF libLLVM-15. It deliberately includes ONLY
// LLVM headers (plus glog) and never pulls in paddle/utils/variant.h or other
// CINN headers, so it compiles cleanly with RTTI disabled.

#include "paddle/cinn/backends/llvm/object_cache.h"

#include <glog/logging.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>
// Completes llvm::Target (used via TargetMachine::getTarget() below). The header
// moved from Support/ to MC/ in LLVM 14.
#if LLVM_VERSION_MAJOR >= 14
#include <llvm/MC/TargetRegistry.h>
#else
#include <llvm/Support/TargetRegistry.h>
#endif
#include <llvm/Target/TargetMachine.h>

#include <memory>
#include <utility>

namespace cinn::backends {

void NaiveObjectCache::notifyObjectCompiled(const llvm::Module *m,
                                            llvm::MemoryBufferRef obj_buffer) {
  cached_objects_[m->getModuleIdentifier()] =
      llvm::MemoryBuffer::getMemBufferCopy(obj_buffer.getBuffer(),
                                           obj_buffer.getBufferIdentifier());
}

std::unique_ptr<llvm::MemoryBuffer> NaiveObjectCache::getObject(
    const llvm::Module *m) {
  auto it = cached_objects_.find(m->getModuleIdentifier());
  if (it == cached_objects_.end()) {
    VLOG(1) << "No object for " << m->getModuleIdentifier()
            << " in cache. Compiling.";
    return nullptr;
  }

  VLOG(3) << "Object for " << m->getModuleIdentifier() << " loaded from cache.";
  return llvm::MemoryBuffer::getMemBuffer(it->second->getMemBufferRef());
}

llvm::Expected<std::unique_ptr<llvm::orc::IRCompileLayer::IRCompiler>>
CreateObjectCacheCompiler(llvm::orc::JITTargetMachineBuilder jtmb,
                          llvm::ObjectCache *cache) {
  auto machine = llvm::cantFail(jtmb.createTargetMachine());
  VLOG(6) << "create llvm compile layer";
  VLOG(6) << "Target Name: " << machine->getTarget().getName();
  VLOG(6) << "Target CPU: " << machine->getTargetCPU().str() << std::endl;
  return std::make_unique<llvm::orc::TMOwningSimpleCompiler>(std::move(machine),
                                                             cache);
}

}  // namespace cinn::backends
