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
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
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

struct VarHelper;
struct OpHelper;
struct FunctionHelper;

struct VarHelper {
  VarHelper() = default;
  explicit VarHelper(const char*);
  // VarHelper(std::initializer_list<VarHelper>);
};

struct OpHelper {
  explicit OpHelper(const char*);

  VarHelper& operator()(std::pair<std::string, VarHelper>);
  VarHelper& operator()(std::pair<std::string, std::vector<VarHelper>>);
  VarHelper& operator()(
      std::initializer_list<std::pair<std::string, VarHelper>>);
  VarHelper& operator()(
      std::initializer_list<std::pair<std::string, std::vector<VarHelper>>>);
};

struct FunctionHelper {
  template <typename T>
  explicit FunctionHelper(const T&& f) {
    // using Var = GeneratePassRegister::VarHelper;
    // static_assert(std::is_convertible<T, std::function<Var(Var, Var,
    // Var)>>::value);
    // 获取可执行的函数
    // 执行
    // 获取desc
  }

  proto::ProgramDesc program_desc_;
};

}  // namespace generate_pass

struct PassPairs {
  PassPairs() = default;

  template <typename PT, typename RT>
  explicit PassPairs(std::pair<PT, RT> t) {
    // using Var = GeneratePassRegister::VarHelper;
    // static_assert(std::is_convertible<PT, std::function<Var(Var, Var,
    // Var)>>::value);
  }

  template <typename T1, typename T2>
  void push_back(std::pair<T1, T2> t) {}

  proto::MultiPassDesc ToMultiPassDesc();
};

// Use function to register in CC.
template <PassPairs (*Functor)(void)>
class CXXGeneratePass : public GeneratePass {
 public:
  CXXGeneratePass() : GeneratePass(Functor().ToMultiPassDesc()) {}
};

#define VAR_(name) ::paddle::framework::ir::generate_pass::VarHelper name
#define OP_(type) ::paddle::framework::ir::generate_pass::OpHelper(#type)
#define SUBGRAPH_(name) \
  ::paddle::framework::ir::generate_pass::FunctionHelper name

#define REGISTER_GENERATE_PASS(pass_type)                               \
  paddle::framework::ir::PassPairs register_##pass_type();              \
  REGISTER_PASS(                                                        \
      pass_type,                                                        \
      ::paddle::framework::ir::CXXGeneratePass<&register_##pass_type>); \
  paddle::framework::ir::PassPairs register_##pass_type()

}  // namespace ir
}  // namespace framework
}  // namespace paddle
