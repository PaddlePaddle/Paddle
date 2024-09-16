// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/jit/function_schema.h"

#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/jit/function_utils.h"
#include "paddle/fluid/pybind/pir_utils.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/pir/include/core/program.h"

namespace paddle::jit {

Argument::Argument(const std::string& name, bool is_out)
    : name_(name), is_output_(is_out) {}

const std::string& Argument::Name() const { return name_; }

const std::vector<std::string> FunctionSchema::InputArgNames() const {
  std::vector<std::string> input_arg_names;
  input_arg_names.reserve(input_args.size());
  for (auto& arg : input_args) {
    input_arg_names.emplace_back(arg.Name());
  }
  return input_arg_names;
}

const std::vector<std::string> FunctionSchema::OutputArgNames() const {
  std::vector<std::string> output_arg_names;
  output_arg_names.reserve(output_args.size());
  for (auto& arg : output_args) {
    output_arg_names.emplace_back(arg.Name() + "@fetch");
  }
  return output_arg_names;
}

void FunctionSchema::AddInputArg(const std::string& name) {
  input_args.emplace_back(name, false);
}

void FunctionSchema::AddOutputArg(const std::string& name) {
  output_args.emplace_back(name, true);
}

/* base function info*/
BaseFunctionInfo::BaseFunctionInfo(const std::string& func_name,
                                   const std::vector<std::string>& param_names)
    : func_name_(func_name), param_names_(param_names) {}
const std::string& BaseFunctionInfo::FunctionName() const { return func_name_; }

const std::vector<std::string>& BaseFunctionInfo::ParamNames() const {
  return param_names_;
}

const std::vector<std::string> BaseFunctionInfo::InputArgNames() const {
  return schema_.InputArgNames();
}

const std::vector<std::string> BaseFunctionInfo::OutputArgNames() const {
  return schema_.OutputArgNames();
}

const std::string& BaseFunctionInfo::ProgramFilePath() const {
  return prog_file_path_;
}

void BaseFunctionInfo::SetProgramFilePath(const std::string& path) {
  prog_file_path_ = path;
}

/* FunctionInfo */
FunctionInfo::FunctionInfo(const std::string& func_name,
                           const std::vector<std::string>& param_names,
                           const framework::ProgramDesc& program_desc)
    : BaseFunctionInfo(func_name, param_names) {
  program_desc_.reset(new framework::ProgramDesc(program_desc));
  // Parse FunctionSchema
  for (auto& in_name : program_desc_->GetFeedTargetNames()) {
    schema_.AddInputArg(in_name);
  }
  for (auto& out_name : program_desc_->GetFetchTargetNames()) {
    schema_.AddOutputArg(out_name);
  }
}

const framework::ProgramDesc& FunctionInfo::ProgramDesc() const {
  return *program_desc_.get();  // NOLINT
}

void FunctionInfo::RemoveDescFeedFetch() {
  utils::RemoveFeedFetch(program_desc_.get());
}

/* pirFunctionInfo*/
PirFunctionInfo::PirFunctionInfo(const std::string& func_name,
                                 const std::vector<std::string>& param_names,
                                 std::shared_ptr<pir::Program> program)
    : BaseFunctionInfo(func_name, param_names) {
  program_ = program;
  // Parse FunctionSchema
  for (auto& in_name : GetFeedTargetNames(program_.get())) {
    schema_.AddInputArg(in_name);
  }
  for (auto& out_name : GetFetchTargetNames(program_.get())) {
    schema_.AddOutputArg(out_name);
  }
}

std::shared_ptr<pir::Program> PirFunctionInfo::Program() const {
  return program_;  // NOLINT
}

}  // namespace paddle::jit
