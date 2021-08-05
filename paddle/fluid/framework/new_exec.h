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

#include <iostream>
#include <string>

#include <chrono>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/executor_gc_helper.h"
#include "paddle/fluid/framework/garbage_collector.h"
#include "paddle/fluid/framework/new_exec_util.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/init.h"

namespace paddle {
namespace framework {

using std::cerr;
using std::endl;

using OpKernelComputeFunc = std::function<void(const ExecutionContext&)>;
using OpKernelMap =
    std::unordered_map<OpKernelType, OpKernelComputeFunc, OpKernelType::Hash>;

framework::ProgramDesc LoadFromFile(const std::string& file_name) {
  std::ifstream fin(file_name, std::ios::in | std::ios::binary);
  fin.seekg(0, std::ios::end);
  std::string buffer(fin.tellg(), ' ');
  fin.seekg(0, std::ios::beg);
  fin.read(&buffer[0], buffer.size());
  fin.close();

  ProgramDesc program_desc(buffer);
  return program_desc;
}

struct OpKernelFunc {
  OpKernelComputeFunc compute_func_;
  OperatorBase* operator_base_;
};

struct VariableMetaInfo {
  int var_ref_count_;
};

struct VariableScope {
  std::vector<Variable*> var_list;
  std::map<std::string, int> name2id;
  std::vector<VariableMetaInfo> vec_meta_info_;
};

struct NextInstruction {
  std::vector<size_t> direct_run_;
};

struct EventInter {};

struct InstructionInfo {
  std::vector<size_t> dependecy_count_;
};

struct EventRun {
  EventInter event_inter;
  std::vector<size_t> same_device_run_;
  std::vector<size_t> synchronized_run;
};

struct Instruction {
  OpKernelFunc kernel_func_;
  std::map<std::string, std::vector<int>> input_index_;
  std::map<std::string, std::vector<int>> output_index_;

  std::vector<size_t> gc_check_var_list;
  NextInstruction next_instruction_;
  std::vector<EventInter> vec_event_list_;
};

struct OpFuncNode {
  // int unsed;
  std::map<std::string, std::vector<int>> input_index;
  std::map<std::string, std::vector<int>> output_index;

  OpKernelComputeFunc kernel_func_;
};

void BuildVariableScopeFromOuterScope(const framework::ProgramDesc& pdesc,
                                      VariableScope* var_scope,
                                      Scope* outer_scope);

void BuildVariableScope(const framework::ProgramDesc& pdesc,
                        VariableScope* var_scope);

void BuildOpFuncList(const framework::ProgramDesc& pdesc,
                     std::vector<OperatorBase*>* op_list,
                     std::vector<OpFuncNode>* vec_func_list,
                     VariableScope* var_scope, const platform::Place& place);

class InterpreterCore {
 public:
  InterpreterCore(const platform::Place& place, const ProgramDesc& prog,
                  const ProgramDesc& startup_prog, Scope* scope);

  void Run(const std::vector<std::string>& vec_name,
           const std::vector<framework::Tensor>& vec_tensor,
           const std::vector<std::string>& vec_fetch_name,
           std::vector<framework::Tensor>* vec_out);

 private:
  void ConvertToInstrcutions();

  void RunInstruction(const Instruction& instr_node,
                      const VariableScope& var_scope,
                      const platform::Place& place);

  void RunInstructionList(const std::vector<Instruction>& vec_instr,
                          const VariableScope& var_scope,
                          const platform::Place& place);

  const platform::Place& place_;
  const ProgramDesc& prog_;
  paddle::framework::VariableScope global_scope;
  std::vector<paddle::framework::OpFuncNode> vec_func_list;
  std::vector<paddle::framework::OperatorBase*> op_list;

  bool is_build_;

  std::vector<Instruction> vec_instruction_;

  InstructionInfo instruction_info_;

  std::vector<size_t> dependecy_count_;
  std::vector<VariableMetaInfo> ref_coun_info;
  std::vector<std::vector<size_t>> input_var2op_info_;

  Scope* outer_scope_;
};
}  // namespace framework
}  // namespace paddle
