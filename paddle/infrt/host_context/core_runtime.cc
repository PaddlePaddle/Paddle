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

#include "paddle/infrt/host_context/core_runtime.h"

#include <unordered_map>

#include <string>
#include <vector>

#include "paddle/infrt/host_context/kernel_registry.h"
#include "paddle/infrt/host_context/op_executable.h"
#include "paddle/infrt/host_context/symbol_table.h"

namespace infrt {
namespace host_context {

struct CoreRuntime::Impl {
  KernelRegistry* kernel_registry{};
  SymbolTable symbol_table;
  std::vector<OpExecutableBuilder> op_executables;

  mutable std::vector<ValueRef> results;
};

SymbolTable* CoreRuntime::symbol_table() { return &impl_->symbol_table; }

CoreRuntime::CoreRuntime(CoreRuntime::Impl* impl) : impl_(impl) { CHECK(impl); }

void CoreRuntime::Execute() {
  // std::cout << "CoreRuntime::Execute" << std::endl;
  int op_offset = 0;
  for (auto& op : impl_->op_executables) {
    VLOG(3) << "running op " << op_offset++ << " " << op.name();
    op.Execute();
  }
}

KernelRegistry* CoreRuntime::kernel_registry() const {
  return impl_->kernel_registry;
}

size_t CoreRuntime::num_ops() const { return impl_->op_executables.size(); }

CoreRuntimeBuilder::CoreRuntimeBuilder(KernelRegistry* kernel_registry)
    : CoreRuntime(new Impl) {
  impl_->kernel_registry =
      kernel_registry ? kernel_registry : GetCpuKernelRegistry();
}

OpExecutableBuilder* CoreRuntimeBuilder::NewOpExecutable(
    const std::string& op_name) {
  CHECK(impl_.get());
  impl_->op_executables.emplace_back(
      op_name, symbol_table(), impl_->kernel_registry);
  return &impl_->op_executables.back();
}

void CoreRuntimeBuilder::FeedInArgs(
    llvm::ArrayRef<std::pair<std::string, ValueRef>> args) {
  for (auto& item : args) {
    symbol_table()->Register(item.first, item.second);
  }
}

void CoreRuntimeBuilder::SetKernelRegistry(KernelRegistry* x) {
  CHECK(x);
  impl_->kernel_registry = x;
}

llvm::SmallVector<ValueRef, 4> CoreRuntime::GetResults(
    llvm::ArrayRef<std::string> arg_names) {
  llvm::SmallVector<ValueRef, 4> results;
  for (auto& name : arg_names) {
    results.push_back(ValueRef(symbol_table()->GetValue(name)));
  }

  return results;
}

CoreRuntime::~CoreRuntime() {}

}  // namespace host_context
}  // namespace infrt
