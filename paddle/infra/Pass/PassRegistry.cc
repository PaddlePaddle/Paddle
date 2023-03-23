// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "Pass/PassRegistry.h"
#include <memory>
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ManagedStatic.h"

namespace infra {

static llvm::ManagedStatic<llvm::StringMap<PassAllocatorFunction>> PassRegistry;

void RegisterPass(const PassAllocatorFunction& func) {
  std::unique_ptr<Pass> pass = func();

  PassRegistry->try_emplace(pass->GetPassInfo().name, func);
}

}  // namespace infra
