// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/lang/lower_impl.h"

#include <gtest/gtest.h>

#include "paddle/cinn/cinn.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace lang {
namespace detail {

#define CREATE_GNODE(k__) auto* n##k__ = graph->RetrieveNode(#k__);
#define ASSERT_LINKED(a__, b__) ASSERT_TRUE(n##a__->IsLinkedTo(n##b__));

TEST(CreateCompGraph, single_layer) {
  Expr M(100);
  Expr N(200);

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  auto C = Compute(
      {M, N}, [&](Expr i, Expr j) { return A(i, j) + B(i, j); }, "C");

  LOG(INFO) << C->expr_fields().size();
  for (auto* e : C->expr_fields()) {
    LOG(INFO) << "e: " << *e;
  }

  auto stages = CreateStages({C});
  auto graph = CreateCompGraph({A, B, C}, stages);

  LOG(INFO) << "graph:\n" << graph->Visualize();

  /* generated graph
    digraph G {
       node_0[label="A"]
       node_1[label="B"]
       node_2[label="C"]
       node_0->node_2
       node_1->node_2
    } // end G
  */

  CREATE_GNODE(A)
  CREATE_GNODE(B)
  CREATE_GNODE(C)

  ASSERT_TRUE(nA->IsLinkedTo(nC));
  ASSERT_TRUE(nB->IsLinkedTo(nC));
}

TEST(CreateCompGraph, multi_layers) {
  Expr M(100);
  Expr N(200);

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  // A->C
  // B->C
  auto C = Compute(
      {M, N}, [&](Expr i, Expr j) { return A(i, j) + B(i, j); }, "C");

  // C->D
  // B->D
  auto D = Compute(
      {M, N}, [&](Expr i, Expr j) { return C(i, j) + B(i, j); }, "D");

  // A->E
  // B->E
  // C->E
  // D->E
  auto E = Compute(
      {M, N},
      [&](Expr i, Expr j) { return A(i, j) + B(i, j) + C(i, j) + D(i, j); },
      "E");

  auto stages = CreateStages({C, D, E});
  auto graph = CreateCompGraph({A, B, E}, stages);

  LOG(INFO) << "graph:\n" << graph->Visualize();

  /*
   digraph G {
     node_0[label="A"]
     node_1[label="B"]
     node_3[label="C"]
     node_4[label="D"]
     node_2[label="E"]
     node_0->node_2
     node_0->node_3
     node_1->node_2
     node_1->node_4
     node_1->node_3
     node_3->node_2
     node_3->node_4
     node_4->node_2
   } // end G
  */

  CREATE_GNODE(A)
  CREATE_GNODE(B)
  CREATE_GNODE(C)
  CREATE_GNODE(D)
  CREATE_GNODE(E)

  ASSERT_EQ(graph->num_nodes(), 5);

  ASSERT_LINKED(A, C)
  ASSERT_LINKED(B, C)

  ASSERT_LINKED(C, D)
  ASSERT_LINKED(B, D)

  ASSERT_LINKED(A, E)
  ASSERT_LINKED(B, E)
  ASSERT_LINKED(C, E)
  ASSERT_LINKED(D, E)
}

TEST(CreateCompGraph, multi_layers_with_extra_deps) {
  Expr M(100);
  Expr N(200);

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  // A->C
  auto C = Compute(
      {M, N}, [&](Expr i, Expr j) { return A(i, j) + 1.f; }, "C");

  // B->D
  auto D = Compute(
      {M, N}, [&](Expr i, Expr j) { return B(i, j) + 1.f; }, "D");

  // A->E
  auto E = Compute(
      {M, N}, [&](Expr i, Expr j) { return A(i, j) + 1.f; }, "E");

  auto F = Compute(
      {M, N}, [&](Expr i, Expr j) { return C(i, j) + D(i, j) + E(i, j); }, "F");

  auto stages = CreateStages({C, D, E, F});
  // C->D
  stages[D]->CtrlDepend(C);
  // C->E
  stages[E]->CtrlDepend(C);

  auto graph = CreateCompGraph({A, B, F}, stages);

  LOG(INFO) << "graph:\n" << graph->Visualize();

  /*
   digraph G {
      node_0[label="A"]
      node_1[label="B"]
      node_3[label="C"]
      node_4[label="D"]
      node_5[label="E"]
      node_2[label="F"]
      node_0->node_5
      node_0->node_3
      node_1->node_4
      node_3->node_2
      node_3->node_5
      node_3->node_4
      node_4->node_2
      node_5->node_2
   } // end G
   */

  CREATE_GNODE(A)
  CREATE_GNODE(B)
  CREATE_GNODE(C)
  CREATE_GNODE(D)
  CREATE_GNODE(E)
  CREATE_GNODE(F)

  ASSERT_LINKED(B, D)
  ASSERT_LINKED(A, C)
  ASSERT_LINKED(A, E)
  ASSERT_LINKED(C, E)
  ASSERT_LINKED(C, F)
  ASSERT_LINKED(C, D)
  ASSERT_LINKED(D, F)
}

TEST(CreateCompGraph, inline_compatible) {
  Expr M(100);
  Expr N(200);

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  // A->C
  // B->C
  auto C = Compute(
      {M, N}, [&](Expr i, Expr j) { return A(i, j) + B(i, j); }, "C");

  // C->D
  // B->D
  auto D = Compute(
      {M, N}, [&](Expr i, Expr j) { return C(i, j) + B(i, j); }, "D");

  // A->E
  // B->E
  // C->E
  // D->E
  auto E = Compute(
      {M, N},
      [&](Expr i, Expr j) { return A(i, j) + B(i, j) + C(i, j) + D(i, j); },
      "E");

  auto stages = CreateStages({C, D, E});
  stages[D]->ComputeInline();

  auto graph = CreateCompGraph({A, B, E}, stages, true);

  LOG(INFO) << "graph:\n" << graph->Visualize();

  /*
    digraph G {
    node_0[label="A"]
    node_1[label="B"]
    node_3[label="C"]
    node_2[label="E"]
    node_0->node_2
    node_0->node_3
    node_1->node_2
    node_1->node_3
    node_3->node_2
    } // end G
  */

  CREATE_GNODE(A)
  CREATE_GNODE(B)
  CREATE_GNODE(C)
  CREATE_GNODE(E)

  ASSERT_EQ(graph->num_nodes(), 4);
  ASSERT_TRUE(nA->IsLinkedTo(nC));
  ASSERT_TRUE(nA->IsLinkedTo(nE));
  ASSERT_TRUE(nB->IsLinkedTo(nC));
  ASSERT_TRUE(nB->IsLinkedTo(nE));
  ASSERT_TRUE(nA->IsLinkedTo(nC));
  ASSERT_TRUE(nB->IsLinkedTo(nE));
}

TEST(CreateCompGraph, inline_compatible1) {
  Expr M(100);
  Expr N(200);

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  // A->C
  // B->C
  auto C = Compute(
      {M, N}, [&](Expr i, Expr j) { return A(i, j) + B(i, j); }, "C");

  // C->D
  // B->D
  auto D = Compute(
      {M, N}, [&](Expr i, Expr j) { return C(i, j) + B(i, j); }, "D");

  // A->E
  // B->E
  // C->E
  // D->E
  auto E = Compute(
      {M, N},
      [&](Expr i, Expr j) { return A(i, j) + B(i, j) + C(i, j) + D(i, j); },
      "E");

  auto stages = CreateStages({C, D, E});
  stages[C]->ComputeInline();

  auto graph = CreateCompGraph({A, B, E}, stages, true);

  LOG(INFO) << "graph:\n" << graph->Visualize();

  /*
  digraph G {
     node_0[label="A"]
     node_1[label="B"]
     node_3[label="D"]
     node_2[label="E"]
     node_0->node_2
     node_1->node_2
     node_1->node_3
     node_3->node_2
  } // end G
  */

  CREATE_GNODE(A)
  CREATE_GNODE(B)
  CREATE_GNODE(D)
  CREATE_GNODE(E)

  ASSERT_EQ(graph->num_nodes(), 4);

  ASSERT_TRUE(nA->IsLinkedTo(nE));
  ASSERT_TRUE(nD->IsLinkedTo(nE));
  ASSERT_TRUE(nB->IsLinkedTo(nE));
  ASSERT_TRUE(nB->IsLinkedTo(nD));
  ASSERT_TRUE(nD->IsLinkedTo(nE));
}

}  // namespace detail
}  // namespace lang
}  // namespace cinn
