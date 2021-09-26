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
#include "paddle/fluid/framework/pass_desc.pb.h"

namespace paddle {
namespace framework {
namespace ir {

// Generate a substitute pass from protobuf.
class GeneratePass : public Pass {
 public:
  // from binary_str
  explicit GeneratePass(const std::string& binary_str);
  // from PassDesc/MultiPassDesc
  explicit GeneratePass(const proto::MultiPassDesc& multi_pass_desc);

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

class VarHelper {
 public:
  enum class Type { kInput, kOutput, kIntermediate };

  explicit VarHelper(const char* name);
  VarHelper(const std::string& name, Type type);

  const std::string& Name() const;

  bool CheckIntermediate();

 private:
  // VarHelper() = delete;
  std::string name_;
  Type type_;
};

class OpHelper {
 public:
  OpHelper(const char* type, SubgraphHelper* subgraph_helper);

  OpHelper& operator()(std::pair<const char*, VarHelper> input);
  OpHelper& operator()(std::pair<const char*, std::vector<VarHelper>> input);
  OpHelper& operator()(std::vector<std::pair<const char*, VarHelper>> inputs);
  OpHelper& operator()(
      std::vector<std::pair<const char*, std::vector<VarHelper>>> inputs);

  VarHelper Out(const char* name);

 private:
  const char* type_;
  proto::OpDesc* op_desc_;
  SubgraphHelper* subgraph_helper_;
};

class SubgraphHelper {
 public:
  template <typename T>
  explicit SubgraphHelper(const T&& f) {
    proto::BlockDesc* block = program_desc_.add_blocks();
    block->set_idx(0);
    block->set_parent_idx(0);
    AddOutputVars(f());
  }

  proto::ProgramDesc* ProgramDesc();
  const proto::ProgramDesc& ProgramDesc() const;
  const std::vector<std::string>& InputVars() const;
  const std::vector<std::string>& OutputVars() const;

  void AddInputVar(const std::string& name);

  void AddOutputVars(const VarHelper& var_helper);

  template <size_t i, typename... Ts,
            std::enable_if_t<i + 1 < sizeof...(Ts)>* = nullptr>
  void AddOutputVars(const std::tuple<Ts...>& outputs) {
    AddOutputVars(std::get<i>(outputs));
    AddOutputVars<i + 1>(outputs);
  }

  template <size_t i, typename... Ts,
            std::enable_if_t<i + 1 == sizeof...(Ts)>* = nullptr>
  void AddOutputVars(const std::tuple<Ts...>& outputs) {
    AddOutputVars(std::get<i>(outputs));
  }

  template <typename... Ts>
  void AddOutputVars(const std::tuple<Ts...>& outputs) {
    AddOutputVars<0>(outputs);
  }

 private:
  std::vector<std::string> input_vars_;
  std::vector<std::string> output_vars_;
  proto::ProgramDesc program_desc_;
};

}  // namespace generate_pass

class PassPairs {
 public:
  using SubgraphType = generate_pass::SubgraphHelper;
  PassPairs() = default;

  PassPairs(SubgraphType pattern, SubgraphType replace);

  void push_back(std::pair<SubgraphType, SubgraphType> pass_pair);

  proto::MultiPassDesc ToMultiPassDesc();

 private:
  std::vector<std::pair<SubgraphType, SubgraphType>> pass_pairs_;
};

// Use function to register in CC.
template <PassPairs (*Functor)(void)>
class CXXGeneratePass : public GeneratePass {
 public:
  CXXGeneratePass() : GeneratePass(Functor().ToMultiPassDesc()) {}
};

#define VAR_(name)                                         \
  ::paddle::framework::ir::generate_pass::VarHelper name = \
      ::paddle::framework::ir::generate_pass::VarHelper(#name)
#define OP_(type) \
  ::paddle::framework::ir::generate_pass::OpHelper(#type, subgraph)
#define SUBGRAPH_(name) \
  ::paddle::framework::ir::generate_pass::SubgraphHelper name

#define REGISTER_GENERATE_PASS(pass_type)                               \
  paddle::framework::ir::PassPairs register_##pass_type();              \
  REGISTER_PASS(                                                        \
      pass_type,                                                        \
      ::paddle::framework::ir::CXXGeneratePass<&register_##pass_type>); \
  paddle::framework::ir::PassPairs register_##pass_type()

}  // namespace ir
}  // namespace framework
}  // namespace paddle
