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

#include "paddle/fluid/framework/ir/generate_pass.h"
#include "gtest/gtest.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"

namespace paddle {
namespace framework {
namespace ir {

void init_test_pass_desc(proto::PassDesc* pass_desc) {
  pass_desc->set_name("test_generate_pass");
  // pattern
  proto::PassDesc::Op* mul = pass_desc->add_pattern_op();
  mul->set_type("mul");
  mul->add_input()->set_name("X");
  mul->add_input()->set_name("Y");
  mul->mutable_input(1)->set_persistable(true);
  mul->add_output()->set_name("out");
  proto::PassDesc::Op* elementwise_add = pass_desc->add_pattern_op();
  elementwise_add->set_type("elementwise_add");
  elementwise_add->add_input()->set_name("X");
  elementwise_add->mutable_input(0)->set_from_op_type("mul");
  elementwise_add->mutable_input(0)->set_from_op_var("out");
  elementwise_add->add_input()->set_name("bias");
  elementwise_add->mutable_input(1)->set_persistable(true);
  elementwise_add->add_output()->set_name("out");
  // algebra
  proto::PassDesc::Op* fc = pass_desc->add_algebra_op();
  fc->set_type("fc");
  fc->add_input()->set_name("Input");
  fc->mutable_input(0)->set_from_op_type("mul");
  fc->mutable_input(0)->set_from_op_var("X");
  fc->add_input()->set_name("W");
  fc->mutable_input(1)->set_from_op_type("mul");
  fc->mutable_input(1)->set_from_op_var("Y");
  fc->add_input()->set_name("Bias");
  fc->mutable_input(2)->set_from_op_type("elementwise_add");
  fc->mutable_input(2)->set_from_op_var("bias");
  fc->add_output()->set_name("Out");
  fc->mutable_output(0)->set_from_op_type("elementwise_add");
  fc->mutable_output(0)->set_from_op_var("out");
}

void AddVarToScope(Scope* param_scope, const std::string& name,
                   const DDim& dims) {
  auto* tensor = param_scope->Var(name)->GetMutable<LoDTensor>();
  tensor->Resize(dims);
  tensor->mutable_data<float>(platform::CPUPlace());
}

Scope* CreateParamScope() {
  auto param_scope = new Scope();
  AddVarToScope(param_scope, "weights_0", {});
  AddVarToScope(param_scope, "bias_1", {});
  return param_scope;
}

TEST(GeneatePass, basic) {
  proto::MultiPassDesc multi_pass_desc;
  proto::PassDesc* pass_desc = multi_pass_desc.add_pass_desc();
  init_test_pass_desc(pass_desc);

  GeneratePass pass(multi_pass_desc);
  Layers layers;
  auto* a = layers.data("a");
  auto* relu_out_0 = layers.relu(a);
  auto* weights_0 = layers.data("weights_0", {}, true);
  auto* mul_out_0 = layers.mul(relu_out_0, weights_0);
  auto* bias_1 = layers.data("bias_1", {}, true);
  layers.elementwise_add(mul_out_0, bias_1);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  graph->Set("__param_scope__", CreateParamScope());
  int num_nodes_before = graph->Nodes().size();
  int num_mul_nodes_before = GetNumOpNodes(graph, "mul");
  VLOG(3) << DebugString(graph);

  graph.reset(pass.Apply(graph.release()));
  int num_nodes_after = graph->Nodes().size();
  int num_fc_nodes_after = GetNumOpNodes(graph, "fc");
  VLOG(3) << DebugString(graph);

  PADDLE_ENFORCE_EQ(num_nodes_before, num_nodes_after + 6,
                    platform::errors::InvalidArgument(
                        "num_nodes_before=%d, num_nodes_after=%d.",
                        num_nodes_before, num_nodes_after));
  PADDLE_ENFORCE_EQ(num_fc_nodes_after, 1,
                    platform::errors::InvalidArgument("num_fc_nodes_after=%d.",
                                                      num_fc_nodes_after));
  PADDLE_ENFORCE_EQ(num_mul_nodes_before, num_fc_nodes_after,
                    platform::errors::InvalidArgument(
                        "num_mul_nodes_before=%d, num_fc_nodes_after=%d.",
                        num_mul_nodes_before, num_fc_nodes_after));
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
