// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>

#include <memory>
#include <string>
#include <utility>

#include "paddle/infrt/host_context/value.h"

namespace infrt {
namespace host_context {

class KernelRegistry;
class OpExecutable;
class OpExecutableBuilder;
class SymbolTable;

/**
 * CoreRuntime encapsulate the execution for a sequence of ops.
 * Each function call will bind to a CoreRuntime instance, push the argument
 * Values in to the argument-list, and get the
 * result Values from the return-list.
 */
class CoreRuntime : public std::enable_shared_from_this<CoreRuntime> {
 public:
  //! Execute a program.
  void Execute();

  //! Return the number of ops.
  size_t num_ops() const;

  //! Get the results of the execution.
  llvm::SmallVector<ValueRef, 4>  //
      GetResults(llvm::ArrayRef<std::string> arg_names);

  std::shared_ptr<CoreRuntime> getptr() {
    return std::shared_ptr<CoreRuntime>(this);
  }

  KernelRegistry* kernel_registry() const;

  ~CoreRuntime();

 protected:
  //! Get the symbol table.
  SymbolTable* symbol_table();

  class Impl;
  explicit CoreRuntime(Impl* impl);
  std::unique_ptr<Impl> impl_;
};

/**
 * The builder for CoreRuntime, help to construct a function.
 */
class CoreRuntimeBuilder : public CoreRuntime {
 public:
  explicit CoreRuntimeBuilder(KernelRegistry* kernel_registry);

  using CoreRuntime::symbol_table;

  void SetKernelRegistry(KernelRegistry* x);

  //! Feed the input arguments, each item is a pair of arg-name and arg-value.
  void FeedInArgs(llvm::ArrayRef<std::pair<std::string, ValueRef>> args);

  llvm::ArrayRef<const std::string&> attr_names() const;

  OpExecutableBuilder* NewOpExecutable(const std::string& op_name);
};

}  // namespace host_context
}  // namespace infrt
