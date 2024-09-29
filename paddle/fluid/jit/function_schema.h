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

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace pir {
class Program;
}

namespace paddle {

namespace framework {
class ProgramDesc;
}  // namespace framework

namespace jit {

class Argument {
 public:
  explicit Argument(const std::string& name, bool is_out = false);

  const std::string& Name() const;

 private:
  std::string name_;
  // paddle::optional<Variable> default_val_;
  bool is_output_;
};

class FunctionSchema {
 public:
  FunctionSchema() = default;

  const std::vector<std::string> InputArgNames() const;

  const std::vector<std::string> OutputArgNames() const;

  void AddInputArg(const std::string& name);

  void AddOutputArg(const std::string& name);

 private:
  // input_args and output_args are ordered
  std::vector<Argument> input_args;
  std::vector<Argument> output_args;
};
class BaseFunctionInfo {
 public:
  BaseFunctionInfo(const std::string& func_name,
                   const std::vector<std::string>& param_names);

  virtual ~BaseFunctionInfo() = default;

  const std::string& FunctionName() const;

  const std::vector<std::string>& ParamNames() const;

  const std::vector<std::string> InputArgNames() const;

  const std::vector<std::string> OutputArgNames() const;

  const std::string& ProgramFilePath() const;

  void SetProgramFilePath(const std::string& path);

 protected:
  std::string func_name_;
  std::vector<std::string> param_names_;
  FunctionSchema schema_;
  std::string prog_file_path_;
};

class FunctionInfo : public BaseFunctionInfo {
 public:
  FunctionInfo(const std::string& func_name,
               const std::vector<std::string>& param_names,
               const framework::ProgramDesc& program_desc);

  const framework::ProgramDesc& ProgramDesc() const;

  void RemoveDescFeedFetch();

 private:
  std::shared_ptr<framework::ProgramDesc> program_desc_;
};

class PirFunctionInfo : public BaseFunctionInfo {
 public:
  PirFunctionInfo(const std::string& func_name,
                  const std::vector<std::string>& param_names,
                  std::shared_ptr<pir::Program> program);

  std::shared_ptr<pir::Program> Program() const;

 private:
  std::shared_ptr<pir::Program> program_;
};

}  // namespace jit
}  // namespace paddle
