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

#include <absl/container/flat_hash_map.h>
#include <gtest/gtest.h>
#include <vector>

#include "paddle/cinn/ast_gen_ius/tensor_group.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/cinn/lang/placeholder.h"

namespace cinn {
namespace ast_gen_ius {

using ir::Expr;
using ir::Tensor;
using ir::Var;
using lang::Compute;
using lang::Placeholder;

TEST(TensorGroup, Easy) {
  auto M = Expr(100);
  auto N = Expr(15);
  Placeholder<float> A("A", {M, N});

  Tensor B = Compute(
      {M, N}, [=](Var i, Var j) -> Expr { return A(i, j) + 1.f; }, "B");

  TensorGroup tensor_group({B});

  ASSERT_TRUE(tensor_group.Contain("A"));
  ASSERT_TRUE(tensor_group.Contain("B"));
  ASSERT_EQ(tensor_group.Get("B")->name, "B");
  ASSERT_EQ(tensor_group.Get("A")->name, "A");
  ASSERT_EQ(tensor_group.GetAllTensors().size(), 2UL);

  ASSERT_EQ(tensor_group.GetCtrlDepTensors("A").size(), 0UL);
  ASSERT_EQ(tensor_group.GetCtrlDepTensors("B").size(), 1UL);
  ASSERT_TRUE(tensor_group.GetCtrlDepTensors("B").count(A));

  std::vector<ir::Tensor> topo_tensors =
      tensor_group.GetGenFuncTopoOrder({A.tensor(), B});
  ASSERT_EQ(topo_tensors.size(), 1UL);
  ASSERT_EQ(topo_tensors[0]->name, "B");

  ASSERT_EQ(tensor_group.GetShareMemRootName("A"), "A");
  ASSERT_EQ(tensor_group.GetShareMemRootName("B"), "B");
  tensor_group.MarkShareMemBuffer(tensor_group.Get("A"), tensor_group.Get("B"));

  absl::flat_hash_map<std::string, ir::Tensor> buffered_tensors =
      tensor_group.AllocateBuffers();
  ASSERT_EQ(buffered_tensors["A"]->buffer->name,
            buffered_tensors["B"]->buffer->name);
}

TEST(TensorGroup, GraphTopo) {
  auto M = Expr(16);
  auto N = Expr(16);

  /*
   *    A   B
   *   / \ /
   *  C   D
   *   \ /
   *    E
   */

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  Tensor C = Compute(
      {M, N}, [=](Var i, Var j) -> Expr { return A(i, j) + 1.f; }, "C");

  Tensor D = Compute(
      {M, N}, [=](Var i, Var j) -> Expr { return A(i, j) + B(i, j); }, "D");

  Tensor E = Compute(
      {M, N}, [=](Var i, Var j) -> Expr { return C(i, j) / D(i, j); }, "E");

  TensorGroup tensor_group({C, D, E});

  std::vector<std::string> check_names = {"A", "B", "C", "D", "E"};
  ASSERT_EQ(tensor_group.GetAllTensors().size(), check_names.size());
  for (const std::string& name : check_names) {
    ASSERT_TRUE(tensor_group.Contain(name));
    ASSERT_EQ(tensor_group.Get(name)->name, name);
  }

  ASSERT_TRUE(tensor_group.GetCtrlDepTensors("E").count(D));
  ASSERT_TRUE(tensor_group.GetCtrlDepTensors("E").count(C));
  ASSERT_TRUE(tensor_group.GetCtrlDepTensors("D").count(A));
  ASSERT_TRUE(tensor_group.GetCtrlDepTensors("D").count(B));
  ASSERT_TRUE(tensor_group.GetCtrlDepTensors("C").count(A));

  std::vector<ir::Tensor> topo_tensors = tensor_group.GetGenFuncTopoOrder();
  ASSERT_EQ(topo_tensors.size(), check_names.size());
  for (size_t i = 0; i < check_names.size(); ++i) {
    ASSERT_EQ(topo_tensors[i]->name, check_names[i]);
  }

  std::vector<ir::Tensor> topo_except_argu =
      tensor_group.GetGenFuncTopoOrder({A.tensor(), B.tensor()});
  ASSERT_EQ(topo_except_argu.size(), 3);
  for (int i = 0; i < 3; ++i) {
    ASSERT_EQ(topo_except_argu[i]->name, check_names[i + 2]);
  }

  for (size_t i = 0; i < check_names.size(); ++i) {
    ASSERT_EQ(tensor_group.GetShareMemRootName(check_names[i]), check_names[i]);
  }
  tensor_group.MarkShareMemBuffer(tensor_group.Get("A"), tensor_group.Get("B"));
  tensor_group.MarkShareMemBuffer(tensor_group.Get("B"), tensor_group.Get("C"));
  tensor_group.MarkShareMemBuffer(tensor_group.Get("C"), tensor_group.Get("D"));

  ASSERT_EQ(tensor_group.GetShareMemRootName("A"),
            tensor_group.GetShareMemRootName("D"));
  absl::flat_hash_map<std::string, ir::Tensor> buffered_tensors =
      tensor_group.AllocateBuffers();
  ASSERT_EQ(buffered_tensors["A"]->buffer->name,
            buffered_tensors["D"]->buffer->name);
}

}  // namespace ast_gen_ius
}  // namespace cinn
