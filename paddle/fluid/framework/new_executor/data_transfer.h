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
#include <string>

#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/new_executor/new_executor_defs.h"
#include "paddle/fluid/framework/op_kernel_type.h"

namespace paddle {
namespace framework {
namespace interpreter {

/*
 * A Helper class to implement data transform operation.
 * It will apply layout/dtype/device transfer by turns.
 */
class DataTranferHelper {
 public:
  DataTranferHelper(const platform::Place& place, VariableScope* var_scope)
      : place_(place), var_scope_(var_scope) {}

  bool apply(const OpKernelType& kernel_type_for_var,
             const OpKernelType& expected_kernel_key,
             const std::string& var_name, std::string* new_var_name,
             std::vector<OpFuncNode>* new_op_func_nodes, bool use_local_scope);

  void RunAndConstructShareNode(const std::string& src_var_name,
                                const std::string& dst_var_name,
                                std::vector<OpFuncNode>* op_func_nodes);

  void RunAndConstructOpFuncNode(const std::shared_ptr<OperatorBase>& op,
                                 const std::string& var_name,
                                 const std::string& new_var_name,
                                 std::vector<OpFuncNode>* op_func_nodes);

 private:
  platform::Place place_;
  VariableScope* var_scope_;
};

void ApplyDataTransform(const OpKernelType& expected_kernel_key,
                        const platform::Place& place,
                        VariableValueMap* ins_map_temp,
                        VariableScope* var_scope, OpFuncNode* op_func_node,
                        std::vector<OpFuncNode>* op_func_nodes,
                        bool use_local_scope = true);

void HandleComplexGradToRealGrad(const OpFuncNode& op_func_node,
                                 const platform::Place& place,
                                 const VariableNameMap& out_names,
                                 VariableValueMap* out_vars,
                                 VariableScope* var_scope,
                                 std::vector<OpFuncNode>* op_func_nodes,
                                 framework::Scope* local_scope);

std::string get_memcpy_type(const platform::Place& src_place,
                            const platform::Place& dst_place);

inline bool need_device_transform(const OpKernelType& kernel_type_for_var,
                                  const OpKernelType& expected_kernel_key) {
  auto& src_place = kernel_type_for_var.place_;
  auto& dst_place = expected_kernel_key.place_;
  if (platform::is_same_place(src_place, dst_place) ||
      (platform::is_cuda_pinned_place(src_place) &&
       platform::is_cpu_place(dst_place))) {
    return false;
  }
  return true;
}

inline bool need_dtype_transform(const OpKernelType& kernel_type_for_var,
                                 const OpKernelType& expected_kernel_key) {
  return framework::NeedTransformDataType(kernel_type_for_var,
                                          expected_kernel_key);
}

inline bool need_layout_transform(const OpKernelType& kernel_type_for_var,
                                  const OpKernelType& expected_kernel_key) {
  return framework::NeedTransformLayout(kernel_type_for_var.data_layout_,
                                        expected_kernel_key.data_layout_);
}

std::shared_ptr<OperatorBase> TransferLayout(const std::string& var_name,
                                             std::string* new_var_name,
                                             DataLayout in_layout,
                                             DataLayout out_layout,
                                             VariableScope* var_scope,
                                             framework::Scope* local_scope);

std::shared_ptr<OperatorBase> TransferDtype(const std::string& var_name,
                                            std::string* new_var_name,
                                            proto::VarType::Type in_dtype,
                                            proto::VarType::Type out_dtype,
                                            VariableScope* var_scope,
                                            framework::Scope* local_scope);

std::shared_ptr<OperatorBase> TransferDevice(const std::string& var_name,
                                             std::string* new_var_name,
                                             const platform::Place& src_place,
                                             const platform::Place& dst_place,
                                             VariableScope* var_scope,
                                             framework::Scope* local_scope);

}  // namespace interpreter
}  // namespace framework
}  // namespace paddle
