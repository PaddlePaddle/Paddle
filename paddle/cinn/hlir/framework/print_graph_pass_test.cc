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

#include <absl/types/any.h>
#include <gtest/gtest.h>

#include <string>

#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/hlir/framework/node.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/pass.h"
#include "paddle/cinn/hlir/op/use_ops.h"
#include "paddle/cinn/lang/packed_func.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace hlir {
namespace framework {

void PrintGraphPass(Graph* src) {
  std::string res;
  auto store_node = std::get<0>(src->topological_order());
  int index = 0;
  for (auto& i : store_node) {
    if (i->is_type<Node>()) {
      res += std::to_string(index) + ":";
      res += i->safe_as<Node>()->attrs.node_name;
      res += "(" + i->id() + ")\n";
      index++;
    }
  }
  src->attrs["print_graph"] = std::make_shared<absl::any>(res);
}

CINN_REGISTER_PASS(PrintGraph)
    .describe(
        "This pass just save the visulization Graph to "
        "g.attrs[\"print_graph\"].")
    .set_change_structure(false)
    .provide_graph_attr("print_graph")
    .set_body(PrintGraphPass);

TEST(Operator, GetAttrs) {
  frontend::Program prog;
  frontend::Variable a("A");
  frontend::Variable b("B");
  Type t = Float(32);
  a->type = t;
  b->type = t;
  auto c = prog.add(a, b);
  auto d = prog.add(c, b);
  auto e = prog.add(c, d);
  ASSERT_EQ(prog.size(), 3);
  Graph* g = new Graph(prog, common::DefaultHostTarget());
  ApplyPass(g, "PrintGraph");
  auto s = g->GetAttrs<std::string>("print_graph");
  LOG(INFO) << s;
  std::string target_str = R"ROC(
0:elementwise_add(elementwise_add_0)
1:elementwise_add(elementwise_add_1)
2:elementwise_add(elementwise_add_2)
)ROC";
  ASSERT_EQ(utils::Trim(s), utils::Trim(target_str));
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
