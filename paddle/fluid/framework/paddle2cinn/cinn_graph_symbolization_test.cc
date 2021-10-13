/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "gtest/gtest.h"

#include "paddle/fluid/framework/paddle2cinn/cinn_graph_symbolization.h"

namespace paddle {
namespace framework {
namespace paddle2cinn {

// only used for test CinnGraphSymbolization class
class CinnGraphSymbolizationTest : public ::testing::Test {
 public:
  void AddFeedVarIntoCinn(const OpMapperContext& ctx) const;

  using CinnOpDesc = ::cinn::frontend::paddle::cpp::OpDesc;
  std::vector<std::unique_ptr<CinnOpDesc>> TransformAllGraphOpToCinn() const;

  std::vector<Node*> TopoSortGraph() const;

  void RunOp(const CinnOpDesc& op_desc, const OpMapperContext& ctx) const;

  void RunGraph(const OpMapperContext& ctx) const;

 protected:
  void SetUp() override {
    // TODO(jiangcheng05): initial CinnGraphSymbolization
  }

 private:
  CinnGraphSymbolization cinn_symbol_;
};

TEST_F(CinnGraphSymbolizationTest, basic) {
  // TODO(jiangcheng05): fill the single test code
}

}  // namespace paddle2cinn
}  // namespace framework
}  // namespace paddle
