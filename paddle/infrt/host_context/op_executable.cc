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

#include "paddle/infrt/host_context/op_executable.h"

#include <string>

#include "paddle/infrt/host_context/kernel_frame.h"
#include "paddle/infrt/host_context/kernel_registry.h"
#include "paddle/infrt/host_context/mlir_function_executable.h"
#include "paddle/infrt/host_context/symbol_table.h"

namespace infrt::host_context {

struct OpExecutable::Impl {
  Impl(const std::string& op_name,
       SymbolTable* symbol_table,
       KernelRegistry* kernel_registry)
      : name(op_name),
        symbol_table(symbol_table),
        kernel_registry(kernel_registry ? kernel_registry
                                        : GetCpuKernelRegistry()) {
    CHECK(kernel_registry);
  }

  inline bool to_execute() const {
    return !run_once || run_once && !has_executed;
  }
  inline void MarkRun() { has_executed = true; }

  std::string name;
  SymbolTable* symbol_table{};
  KernelFrameBuilder frame;
  KernelRegistry* kernel_registry{};

  std::unique_ptr<MlirFunctionExecutable> mlir_function_executable;

  KernelImplementation kernel_impl{};

  //! Tell whether this Op should be executed only once.
  bool run_once{};
  //! Tell whether this op has been executed.
  bool has_executed{};
};

OpExecutable::OpExecutable(OpExecutable::Impl* impl) : impl_(impl) {}

const std::string& OpExecutable::name() const { return impl_->name; }

OpExecutableBuilder::OpExecutableBuilder(const std::string& op_name,
                                         SymbolTable* symbol_table,
                                         KernelRegistry* kernel_registry)
    : OpExecutable(new Impl(op_name, symbol_table, kernel_registry)) {
  CHECK(impl_);
  // CPU kernel registry is the default KernelRegistry.
  impl_->kernel_impl = impl_->kernel_registry->GetKernel(
      std::string(op_name.data(), op_name.size()));
  // TODO(Superjomn) support other device other than CPU.
  CHECK(impl_->kernel_impl) << "No CPU kernel called " << op_name;

  if (op_name == "dt.get_param") {
    impl_->run_once = true;
  }
}

void OpExecutableBuilder::AppendArgument(const std::string& name) {
  if (!impl_->symbol_table->GetValue(name)) {
    impl_->symbol_table->Register(name);
  }
  impl_->frame.AddArgument(impl_->symbol_table->GetValue(name));
}

void OpExecutableBuilder::AppendArgument(Value* value) {
  impl_->frame.AddArgument(value);
}

KernelFrame& OpExecutable::frame() { return impl_->frame; }
const KernelFrame& OpExecutable::frame() const { return impl_->frame; }

void OpExecutableBuilder::SetResults(llvm::ArrayRef<std::string> result_names) {
  llvm::SmallVector<Value*, 3> results;
  for (size_t result_id = 0; result_id < result_names.size(); result_id++) {
    Value* value = impl_->symbol_table->Register(result_names[result_id]);
    results.push_back(value);
  }
  impl_->frame.SetResults(results);
}

void OpExecutableBuilder::SetResults(llvm::ArrayRef<Value*> results) {
  impl_->frame.SetResults(results);
}

void OpExecutableBuilder::AppendAttribute(Value* value) {
  impl_->frame.AddAttribute(value);
}

OpExecutableBuilder::OpExecutableBuilder(OpExecutableBuilder&& other)
    : OpExecutable(other.impl_.release()) {}

MlirFunctionExecutable* OpExecutableBuilder::CreateFunctionExecutable(
    mlir::FuncOp op, MlirToRuntimeTranslator::function_defs_t* function_defs) {
  CHECK(!impl_->mlir_function_executable);
  impl_->mlir_function_executable.reset(
      new MlirFunctionExecutable(op, impl_->kernel_registry, *function_defs));
  return impl_->mlir_function_executable.get();
}

MlirFunctionExecutable* OpExecutableBuilder::CreateFunctionExecutable(
    mlir::Region* region,
    mlir::FunctionType func_type,
    function_defs_t* function_defs) {
  CHECK(!impl_->mlir_function_executable);
  impl_->mlir_function_executable.reset(new MlirFunctionExecutable(
      region, func_type, impl_->kernel_registry, *function_defs));
  return impl_->mlir_function_executable.get();
}

void OpExecutable::Execute() {
#ifndef NDEBUG
  VLOG(3) << "execute " << name()
          << " --- frame args: " << impl_->frame.GetNumArgs() << " results "
          << impl_->frame.GetNumResults() << " attributes "
          << impl_->frame.GetNumAttributes();
  for (int i = 0; i < impl_->frame.GetNumArgs(); i++) {
    VLOG(3) << "function arg: " << impl_->frame.GetArgAt(i);
  }
  for (int i = 0; i < impl_->frame.GetNumResults(); i++) {
    VLOG(3) << "function result: " << impl_->frame.GetResults()[i];
  }
#endif

  if (impl_->to_execute()) {
    impl_->kernel_impl(&impl_->frame);
    impl_->MarkRun();
  }
}

OpExecutable::~OpExecutable() {}

}  // namespace infrt::host_context
