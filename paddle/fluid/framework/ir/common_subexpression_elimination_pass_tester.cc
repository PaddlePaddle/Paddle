/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>

#include <functional>
#include <sstream>
#include <string>
#include <unordered_set>

#include "paddle/fluid/framework/ir/common_subexpression_elimination_pass.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
namespace ir {

TEST(CommonSubexpressionEliminationPass, basic_test) {
  // inputs                           operator            output
  // --------------------------------------------------------------------
  // (relu(a), b)                  elementwise_add ->      d
  // (relu(a), c)                  elementwise_add ->      e
  // (d, e)                        elementwise_add ->      f

  Layers layers;
  auto* a = layers.data("a", {1024, 768});
  auto* b = layers.data("b", {1024, 768});
  auto* c = layers.data("c", {1024, 768});
  auto* d = layers.elementwise_add(layers.relu(a), b);
  auto* e = layers.elementwise_add(layers.relu(a), c);
  auto* f = layers.data("f", {1024, 768});
  layers.elementwise_add(d, e, f, 0);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  auto pass =
      PassRegistry::Instance().Get("common_subexpression_elimination_pass");
  graph.reset(pass->Apply(graph.release()));
  int num_nodes_after = GetNumOpNodes(graph, "relu");
  PADDLE_ENFORCE_EQ(num_nodes_after,
                    1,
                    platform::errors::InvalidArgument(
                        "Before the common subexpression elimination pass, "
                        "there should be 1 "
                        "relu op, but the result is %d",
                        num_nodes_after));
}

TEST(CommonSubexpressionEliminationPass, commutative_operator_test) {
  // inputs                           operator            output
  // --------------------------------------------------------------------
  // (a, b)                        elementwise_add ->      e
  // (b, a)                        elementwise_add ->      f
  // (e, c)                        elementwise_add ->      g
  // (f, d)                        elementwise_add ->      h

  Layers layers;
  auto* a = layers.data("a", {1024, 768});
  auto* b = layers.data("b", {1024, 768});
  auto* c = layers.data("c", {1024, 768});
  auto* d = layers.data("d", {1024, 768});
  auto* e = layers.data("e", {1024, 768});
  auto* f = layers.data("f", {1024, 768});
  auto* g = layers.data("g", {1024, 768});
  auto* h = layers.data("h", {1024, 768});

  layers.elementwise_add(a, b, e, 0);
  layers.elementwise_add(b, a, f, 0);
  layers.elementwise_add(e, c, g, 0);
  layers.elementwise_add(f, d, h, 0);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  auto pass =
      PassRegistry::Instance().Get("common_subexpression_elimination_pass");
  graph.reset(pass->Apply(graph.release()));
  int num_nodes_after = GetNumOpNodes(graph, "elementwise_add");
  PADDLE_ENFORCE_EQ(num_nodes_after,
                    3,
                    platform::errors::InvalidArgument(
                        "Before the common subexpression elimination pass, "
                        "there should be 3 "
                        "elementwise_add op, but the result is %d",
                        num_nodes_after));
}

TEST(CommonSubexpressionEliminationPass, nondeterministic_operator_test) {
  // inputs                           operator            output
  // --------------------------------------------------------------------
  // (dropout(a), b)               elementwise_add ->      d
  // (dropout(a), c)               elementwise_add ->      e
  // (d, e)                        elementwise_add ->      f

  Layers layers;
  auto* a = layers.data("a", {1024, 768});
  auto* b = layers.data("b", {1024, 768});
  auto* c = layers.data("c", {1024, 768});
  auto* d =
      layers.elementwise_add(layers.dropout(a, 0.5, "downgrade_in_infer"), b);
  auto* e =
      layers.elementwise_add(layers.dropout(a, 0.5, "downgrade_in_infer"), c);
  auto* f = layers.data("f", {1024, 768});
  layers.elementwise_add(d, e, f, 0);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  auto pass =
      PassRegistry::Instance().Get("common_subexpression_elimination_pass");
  graph.reset(pass->Apply(graph.release()));
  int num_nodes_after = GetNumOpNodes(graph, "dropout");
  PADDLE_ENFORCE_EQ(num_nodes_after,
                    2,
                    platform::errors::InvalidArgument(
                        "After the common subexpression elimination pass, "
                        "there should still be 2 "
                        "dropout op, but the result is %d",
                        num_nodes_after));
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(common_subexpression_elimination_pass);
