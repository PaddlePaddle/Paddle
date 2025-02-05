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
#include <llvm/ADT/StringRef.h>
#include <llvm/ExecutionEngine/MCJIT.h>
#include <llvm/IR/Argument.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Value.h>

#include <string>
#include <type_traits>
#include <utility>

#include "paddle/cinn/common/type.h"
#include "paddle/common/enforce.h"

namespace cinn {
namespace backends {

template <typename T>
std::string DumpToString(const T &entity) {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  entity.print(os);
  os.flush();
  return buffer;
  // return "\033[33m" + buffer + "\033[0m"; // Green
}

inline llvm::StringRef AsStringRef(absl::string_view str) {
  return llvm::StringRef(str.data(), str.size());
}

llvm::Type *CinnTypeToLLVMType(cinn::common::Type t,
                               llvm::Module *m,
                               bool is_vec = false);

template <typename T>
llvm::Type *llvm_type_of(llvm::Module *m);

}  // namespace backends
}  // namespace cinn
