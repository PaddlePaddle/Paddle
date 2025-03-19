// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <sstream>

#include "paddle/cinn/ir/ir_analyzer/data_dependency_graph.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/stmt.h"

namespace cinn {
namespace ir {

namespace {
using ir::analyzer::DataDependencyGraph;
using ir::analyzer::DepKind;

class TestDataDependencyGraph : public ::testing::Test {
 public:
  void SetUp() override {
    const std::vector<ir::Expr> shape = {};
    tensor_a =
        ir::_Tensor_::Make(common::UniqName("A"), common::Bool(), shape, shape);
    tensor_b =
        ir::_Tensor_::Make(common::UniqName("B"), common::Bool(), shape, shape);
    tensor_c =
        ir::_Tensor_::Make(common::UniqName("C"), common::Bool(), shape, shape);
    tensor_d =
        ir::_Tensor_::Make(common::UniqName("D"), common::Bool(), shape, shape);

    tensor_a->WithBuffer("global", "_" + tensor_a->name + "_temp_buffer");
    tensor_b->WithBuffer("global", "_" + tensor_b->name + "_temp_buffer");
    tensor_c->WithBuffer("global", "_" + tensor_c->name + "_temp_buffer");
    tensor_d->WithBuffer("global", "_" + tensor_d->name + "_temp_buffer");

    var_x = ir::Var(ir::Expr(1), ir::Expr(INT32_MAX), "x");

    load_a = ir::Load::Make(ir::Expr(tensor_a), {});
    load_b = ir::Load::Make(ir::Expr(tensor_b), {});
    load_c = ir::Load::Make(ir::Expr(tensor_c), {});
  };

  ir::Var var_x;
  ir::Tensor tensor_a, tensor_b, tensor_c, tensor_d;
  ir::Expr load_a, load_b, load_c;
};

TEST_F(TestDataDependencyGraph, TensorDep) {
  // A[0] = B[0]
  auto store_a_b = ir::stmt::Store(ir::Expr(tensor_a), load_b, {});
  // B[0] = C[0]
  auto store_b_c = ir::stmt::Store(ir::Expr(tensor_b), load_c, {});
  // C[0] = A[0]
  auto store_c_a = ir::stmt::Store(ir::Expr(tensor_c), load_a, {});
  const std::vector<ir::stmt::StmtRef> stmts = {
      store_a_b, store_b_c, store_c_a};
  const auto &dep_graph = DataDependencyGraph(stmts);
  dep_graph.Print();
  EXPECT_EQ(dep_graph.HasDependency(stmts[0], stmts[1]), DepKind::DEP);
  EXPECT_EQ(dep_graph.HasDependency(stmts[0], stmts[2]), DepKind::DEP);
  EXPECT_EQ(dep_graph.HasDependency(stmts[1], stmts[2]), DepKind::DEP);
}

TEST_F(TestDataDependencyGraph, VarDep) {
  // x = A[0]
  auto let_x_a = ir::stmt::Let(ir::Expr(var_x), load_a);
  // B[0] = x
  auto store_b_x = ir::stmt::Store(ir::Expr(tensor_b), ir::Expr(var_x), {});
  const std::vector<ir::stmt::StmtRef> stmts = {let_x_a, store_b_x};
  const auto &dep_graph = DataDependencyGraph(stmts);
  dep_graph.Print();
  EXPECT_EQ(dep_graph.HasDependency(stmts[0], stmts[1]), DepKind::DEP);
}

TEST_F(TestDataDependencyGraph, TensorNoDep) {
  // A[0] = B[0]
  auto store_a_b = ir::stmt::Store(ir::Expr(tensor_a), load_b, {});
  // C[0] = B[0]
  auto store_c_b = ir::stmt::Store(ir::Expr(tensor_c), load_b, {});
  // D[0] = B[0]
  auto store_d_b = ir::stmt::Store(ir::Expr(tensor_d), load_b, {});
  const std::vector<ir::stmt::StmtRef> stmts = {
      store_a_b, store_c_b, store_d_b};
  const auto &dep_graph = DataDependencyGraph(stmts);
  dep_graph.Print();
  EXPECT_EQ(dep_graph.HasDependency(stmts[0], stmts[1]), DepKind::NO_DEP);
  EXPECT_EQ(dep_graph.HasDependency(stmts[0], stmts[2]), DepKind::NO_DEP);
  EXPECT_EQ(dep_graph.HasDependency(stmts[1], stmts[2]), DepKind::NO_DEP);
}

TEST_F(TestDataDependencyGraph, VarNoDep) {
  // x = A[0]
  auto let_x_a = ir::stmt::Let(ir::Expr(var_x), load_a);
  // B[0] = x
  auto store_b_x = ir::stmt::Store(ir::Expr(tensor_b), ir::Expr(var_x), {});
  // C[0] = x
  auto store_c_x = ir::stmt::Store(ir::Expr(tensor_c), ir::Expr(var_x), {});
  // D[0] = x
  auto store_d_x = ir::stmt::Store(ir::Expr(tensor_d), ir::Expr(var_x), {});
  const std::vector<ir::stmt::StmtRef> stmts = {
      let_x_a, store_b_x, store_c_x, store_d_x};
  const auto &dep_graph = DataDependencyGraph(stmts);
  dep_graph.Print();
  EXPECT_EQ(dep_graph.HasDependency(stmts[1], stmts[2]), DepKind::NO_DEP);
  EXPECT_EQ(dep_graph.HasDependency(stmts[1], stmts[3]), DepKind::NO_DEP);
  EXPECT_EQ(dep_graph.HasDependency(stmts[2], stmts[3]), DepKind::NO_DEP);
}

}  // namespace

}  // namespace ir
}  // namespace cinn
