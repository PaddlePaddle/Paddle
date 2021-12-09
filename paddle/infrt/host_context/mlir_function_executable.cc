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

#include "paddle/infrt/host_context/mlir_function_executable.h"

#include <glog/logging.h>

#include <string>  // NOLINT

#include "paddle/infrt/common/common.h"
#include "paddle/infrt/host_context/core_runtime.h"

namespace infrt {
namespace host_context {

template <typename T>
std::string DumpToString(T& op) {  // NOLINT
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  op.print(os);
  os.flush();
  return buffer;
}

MlirFunctionExecutable::MlirFunctionExecutable(
    mlir::FuncOp func_op,
    KernelRegistry* kernel_registry,
    MlirToRuntimeTranslator::function_defs_t& function_table)
    : Function(func_op.getName().str(),
               func_op.getNumArguments(),
               func_op.getNumResults()),
      MlirToRuntimeTranslator(&core_runtime_builder_),
      region_(&func_op.getRegion()),
      core_runtime_builder_(kernel_registry),
      function_table_(function_table) {}

MlirFunctionExecutable::MlirFunctionExecutable(
    mlir::Region* region,
    mlir::FunctionType func_type,
    KernelRegistry* kernel_registry,
    MlirToRuntimeTranslator::function_defs_t& function_table)
    : Function("", func_type.getNumInputs(), func_type.getNumResults()),
      MlirToRuntimeTranslator(&core_runtime_builder_),
      region_(region),
      core_runtime_builder_(kernel_registry),
      function_table_(function_table) {}

void MlirFunctionExecutable::BuildExecutables(
    llvm::ArrayRef<Value*> arguments,
    llvm::MutableArrayRef<ValueRef> results,
    bool is_region) {
  CHECK_EQ(arguments.size(), num_arguments());
  // We use the function call's arguments as op_executable's operands to avoid
  // copy.
  for (size_t i = 0; i < num_arguments(); i++) {
    AddValue(region_->getArgument(i), arguments[i]);
  }

  // build the program
  auto& blocks = region_->getBlocks();
  CHECK_EQ(blocks.size(), 1UL)
      << "function with more than one block is not supported yet";

  llvm::SmallVector<Value*, 3> runtime_results;
  for (auto& op : blocks.front()) {
    if (EmitConstantOp(&op)) continue;
    if (EmitBuildShapeOp(&op)) continue;

    llvm::SmallVector<mlir::Value, 3> mlir_results;
    if (EmitReturnOp(&op, &mlir_results)) {
      if (!is_region) {
        for (auto v : mlir_results) {
          runtime_results.push_back(GetValue(v));
        }
      }
      continue;
    }

    if (EmitCallOp(&op, &function_table_)) continue;

    if (EmitGeneralOp(&op)) continue;
    LOG(FATAL) << "Not supported op: " << DumpToString(op);
  }

  // after the block is built, we can get the result values of the whole
  // function call in the runtime_results.

  mlir::SmallVector<Value*, 3> results_copied;
  if (!is_region) {
    for (ValueRef& x : results) {
      results_copied.push_back(x.get());
    }
  }

  // set a lambda function to help copy the results from the runtime results in
  // the local function to outer program.
  CHECK_EQ(results_copied.size(), runtime_results.size());
  this->copy_res_fn_ = [results_copied, runtime_results] {
    VLOG(4) << "copy results to result";
    for (size_t i = 0; i < results_copied.size(); i++) {
      VLOG(4) << ".. copy " << runtime_results[i] << " to "
              << results_copied[i];
      CopyTo(*runtime_results[i], results_copied[i]);
    }
  };
}

void MlirFunctionExecutable::Execute(llvm::ArrayRef<Value*> arguments,
                                     llvm::MutableArrayRef<ValueRef> results,
                                     bool is_region) const {
  CHECK_EQ(arguments.size(), num_arguments());
  CHECK_EQ(results.size(), num_results());

  if (core_runtime_builder_.num_ops() == 0) {
    Reference(this).BuildExecutables(arguments, results, is_region);
  }

  Reference(&core_runtime_builder_).Execute();

  copy_res_fn_();
}

}  // namespace host_context
}  // namespace infrt
