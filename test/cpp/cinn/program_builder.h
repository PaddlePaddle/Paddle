// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/common/type.h"
#include "paddle/cinn/frontend/net_builder.h"
#include "paddle/cinn/frontend/syntax.h"
#include "paddle/cinn/utils/type_defs.h"

namespace cinn {
namespace tests {

// This struct packing data fields for variable constructor
struct VariableInfo {
  std::string id;
  std::vector<int> shape;
  common::Type type;
  VariableInfo(std::string name,
               std::vector<int> shape,
               common::Type dtype = common::Float(32))
      : id(name), shape(shape), type(dtype) {}
};

// This class define a general interface to build a frontend::Program easily for
// test usage, developer can implement derived classes to build customized
// programs which may be reused by others through specifying the detail of input
// variables and attributes
class ProgramBuilder {
 public:
  explicit ProgramBuilder(const std::string& name) : builder_(name) {}

  /*
   * \brief Build a frontend::Program with the input variables info and
   * attributes
   * @param input_varinfo The detail data fields of each input variable, input
   * order should match their usage in override implement
   * @param attrs The detail value of each input attributes, input order should
   *                      match their usage in override implementdefinition
   * @return The built program
   */
  virtual frontend::Program Build(
      const std::vector<VariableInfo>& inputs_varinfo,
      const utils::AttributeMap& attrs) = 0;

  // return the output variables
  const std::vector<frontend::Variable>& GetOutputs() const { return outputs_; }

 protected:
  void AppendOutput(frontend::Variable var) { outputs_.emplace_back(var); }

  frontend::NetBuilder builder_;
  std::vector<frontend::Variable> outputs_;
};

// Build a frontend::Program which has only one operator
class OpBuilder final : public ProgramBuilder {
 public:
  /*
   * @param op_name name of the built operator, you should lookup the name from
   * its registry exactly
   */
  OpBuilder(const std::string& op_name);

  // the item order in `inputs_varinfo` and `attrs` should match their usage in
  // the underlying operator
  frontend::Program Build(const std::vector<VariableInfo>& inputs_varinfo,
                          const utils::AttributeMap& attrs = {}) override;

 private:
  std::string op_name_;
};

// Build a frontend::Program by loading paddle model from local files
class PaddleModelBuilder final : public ProgramBuilder {
 public:
  /*
   * @param model_path the path to local model files
   * @param target device type
   */
  PaddleModelBuilder(const std::string& model_path,
                     const common::Target& target);

  frontend::Program Build(const std::vector<VariableInfo>& inputs_varinfo,
                          const utils::AttributeMap& attrs = {}) override;

 private:
  std::string model_path_;
  common::Target target_;
};

}  // namespace tests
}  // namespace cinn
