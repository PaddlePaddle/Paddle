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

}  // namespace ir
}  // namespace framework
}  // namespace paddle
