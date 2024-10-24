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
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/phi/core/framework/pass_desc.pb.h"

namespace paddle {
namespace framework {
namespace ir {

// Generate a substitute pass from protobuf.
class GeneratePass : public Pass {
 public:
  // from binary_str
  explicit GeneratePass(const std::string& binary_str,
                        const std::string& pass_type = "");
  // from PassDesc/MultiPassDesc
  explicit GeneratePass(const proto::MultiPassDesc& multi_pass_desc,
                        const std::string& pass_type = "");

 protected:
  void ApplyImpl(Graph* graph) const override;

 private:
  GeneratePass() = delete;
  DISABLE_COPY_AND_ASSIGN(GeneratePass);
  // Verify desc
  void VerifyDesc() const;
  // Verify graph
  static bool VerifyGraph(const Graph& graph);

  proto::MultiPassDesc multi_pass_desc_;
};

namespace generate_pass {

class VarHelper;
class OpHelper;
class SubgraphHelper;

// VarHelper is used to represent a variable node.
class VarHelper {
 public:
  enum class Type { kInput, kOutput };

  explicit VarHelper(const char* name);
  VarHelper(const std::string& name, Type type);

  std::string name_;
  Type type_;
};

// OpHelper is used to represent a operator node.
class OpHelper {
 public:
  // Convert multiple inputs.
  struct Arguments {
    Arguments(const char* parameter, const VarHelper& var_helper);
    Arguments(const char* parameter,
              std::initializer_list<VarHelper> var_helpers);

    std::string parameter_;
    std::vector<VarHelper> var_helpers_;
  };

  OpHelper(const char* type, SubgraphHelper* subgraph_helper);

  OpHelper& operator()(const Arguments& input);
  OpHelper& operator()(std::initializer_list<Arguments> inputs);

  VarHelper Out(const char* name);

 private:
  OpHelper() = delete;
  DISABLE_COPY_AND_ASSIGN(OpHelper);

  const char* type_;
  proto::OpDesc* op_desc_;
  SubgraphHelper* subgraph_helper_;
};

/*
 * SubgraphHelper is used to define pattern/replace subgraphs.
 *
 * Use lambda expression to define subgraph like Python. SubgraphHelper
 * converts lambda expression to ProgramDesc.
 *
 * In order to define a subgraph, user need to use VarHelper and OpHelper.
 * Use the macros instead of class names, so user can develop better and
 * don't need to know too much about underlying implementation.
 *
 * An example of defining a subgraph as follows:
 *
 *   SUBGRAPH_(subgraph)([subgraph=&subgraph](VAR_(x), VAR_(y), VAR_(z)) {
 *     auto ewadd1 = OP_(elementwise_add)({{"X", x}, {"Y", y}}).Out("Out");
 *     auto ewadd2 = OP_(elementwise_add)({{"X", ewadd1}, {"Y", z}}).Out("Out");
 *     return ewadd2;
 *   });
 *
 */
class SubgraphHelper {
 public:
  SubgraphHelper() = default;
  // The lambda expression is a prvalue expression.
  template <typename T>
  SubgraphHelper& operator=(const T&& f) {
    proto::BlockDesc* block = program_desc_.add_blocks();
    block->set_idx(0);
    block->set_parent_idx(0);
    AddOutputVars(f());
    return *this;
  }

  proto::ProgramDesc* ProgramDesc();
  const proto::ProgramDesc& ProgramDesc() const;
  const std::vector<std::string>& InputVars() const;
  const std::vector<std::string>& OutputVars() const;

  void AddInputVar(const std::string& name);

  void AddOutputVars(const VarHelper& var_helper);

  template <size_t i,
            typename... Ts,
            std::enable_if_t<i + 1 < sizeof...(Ts)>* = nullptr>
  void AddOutputVars(const std::tuple<Ts...>& outputs) {
    AddOutputVars(std::get<i>(outputs));
    AddOutputVars<i + 1>(outputs);
  }

  template <size_t i,
            typename... Ts,
            std::enable_if_t<i + 1 == sizeof...(Ts)>* = nullptr>
  void AddOutputVars(const std::tuple<Ts...>& outputs) {
    AddOutputVars(std::get<i>(outputs));
  }

  template <typename... Ts>
  void AddOutputVars(const std::tuple<Ts...>& outputs) {
    AddOutputVars<0>(outputs);
  }

 private:
  DISABLE_COPY_AND_ASSIGN(SubgraphHelper);
  std::vector<std::string> input_vars_;
  std::vector<std::string> output_vars_;
  proto::ProgramDesc program_desc_;
};

}  // namespace generate_pass

class PassPairs {
 public:
  using SubgraphType = generate_pass::SubgraphHelper;

  PassPairs() = default;
  PassPairs(const SubgraphType& pattern, const SubgraphType& replace);

  void AddPassDesc(const SubgraphType& pattern, const SubgraphType& replace);

  const proto::MultiPassDesc& MultiPassDesc() const;

 private:
  proto::MultiPassDesc multi_pass_desc_;
};

// Use function to register in CC.
template <PassPairs (*Functor)(void)>
class MacroPassHelper : public GeneratePass {
 public:
  MacroPassHelper() : GeneratePass(Functor().MultiPassDesc()) {}
};

#define VAR_(name)                                         \
  ::paddle::framework::ir::generate_pass::VarHelper name = \
      ::paddle::framework::ir::generate_pass::VarHelper(#name)
#define OP_(type) \
  ::paddle::framework::ir::generate_pass::OpHelper(#type, subgraph)
#define SUBGRAPH_(name)                                        \
  ::paddle::framework::ir::generate_pass::SubgraphHelper name; \
  name

#define REGISTER_GENERATE_PASS(pass_type)                               \
  paddle::framework::ir::PassPairs register_##pass_type();              \
  REGISTER_PASS(                                                        \
      pass_type,                                                        \
      ::paddle::framework::ir::MacroPassHelper<&register_##pass_type>); \
  paddle::framework::ir::PassPairs register_##pass_type()

}  // namespace ir
}  // namespace framework
}  // namespace paddle
