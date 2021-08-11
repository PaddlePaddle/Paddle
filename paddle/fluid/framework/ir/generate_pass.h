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
  // from serialized string
  explicit GeneratePass(const std::string& binary_str);
  // from PassDesc/MultiPassDesc
  explicit GeneratePass(const proto::PassDesc& pass_desc);
  explicit GeneratePass(const proto::MultiPassDesc& multi_pass_desc)
      : multi_pass_desc_(multi_pass_desc) {}
  virtual ~GeneratePass() {}

 protected:
  void ApplyImpl(Graph* graph) const override;

 private:
  GeneratePass() = delete;
  // for debug
  void ExportDotString() const {}
  // Verify pass desc
  void VerifyDesc() const {}
  // Verify graph
  void VerifyGraph() const {}
  // InitDetector
  void InitPattern(PDPattern* pattern, const proto::PassDesc& pass_desc) const;
  // Substitute
  GraphPatternDetector::handle_t Substitute(
      const PDPattern& pattern, const proto::PassDesc& pass_desc) const;

  proto::MultiPassDesc multi_pass_desc_;
};

struct Attr {
  template <typename T>
  void Set(T value) {}
};

struct Var {};

struct Op {
  explicit Op(const char* op_type) {}
  ir::Attr Attr(const char* attr_name) { return ir::Attr(); }
  Var Output(const char* var_name) { return Var(); }
  Op& SetInput(std::vector<std::pair<std::string, Var>> inputs) {
    return *this;
  }
  Op& SetOutput(std::pair<std::string, Var> inputs) { return *this; }
};

#define REGISTER_GENERATE_PASS(pass_name)     \
  std::pair<std::vector<Op>, std::vector<Op>> \
      generate_pass_create_##pass_name##_func()

}  // namespace ir
}  // namespace framework
}  // namespace paddle
