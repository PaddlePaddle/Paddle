// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/new_executor/instruction/instruction_base.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"
#include "paddle/phi/api/ext/op_meta_info.h"

namespace pir {
class Operation;
}  // namespace pir

namespace paddle {
namespace framework {
class Scope;
class PythonFunctionInstruction : public InstructionBase {
 public:
  PythonFunctionInstruction(size_t id,
                            const phi::Place& place,
                            pir::Operation* op,
                            const ValueExecutionInfo& value_exec_info);

  pir::Operation* Operation() const override { return op_; }

  void Run() override;

  const std::string& Name() const override { return python_op_name_; }

  void clear();

 private:
  void BuildPythonFunctionContext(
      const paddle::dialect::OpYamlInfoParser& op_yaml_info);

  paddle::CustomOpKernelContext python_function_ctx_;
  paddle::KernelFunc kernel_func_ = nullptr;

  const paddle::PythonOperatorFunctionType* py_func_ptr_ = nullptr;
  const paddle::PythonOperatorInferMetaFunctionType* py_func_infer_meta_ptr_ =
      nullptr;  // Unused in runtime

  // use for update output
  std::vector<phi::DenseTensor*> cache_out_ptrs_;

  std::string python_op_name_;

  pir::Operation* op_{nullptr};  // not owned

  const paddle::OpMetaInfo* python_op_meta_;   // not owned
  const ValueExecutionInfo& value_exec_info_;  // not owned
};

}  // namespace framework
}  // namespace paddle
