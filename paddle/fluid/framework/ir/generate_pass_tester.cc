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

template <proto::MultiPassDesc (*Functor)(void)>
class CXXGeneratePass : public GeneratePass {
 public:
  CXXGeneratePass() : GeneratePass(Functor()) {}
};

#define REGISTER_GENERATE_PASS(pass_type, function) \
  REGISTER_PASS(pass_type, ::paddle::framework::ir::CXXGeneratePass<&function>)

proto::MultiPassDesc generate_fc_fuse() {
  proto::MultiPassDesc multi_pass_desc;
  for (bool with_relu : {true, false}) {
    proto::PassDesc* pass_desc = multi_pass_desc.add_pass_descs();
    proto::BlockDesc* pattern = pass_desc->mutable_pattern()->add_blocks();
    pattern->set_idx(0);
    pattern->set_parent_idx(0);
    proto::OpDesc* mul = pattern->add_ops();
    mul->set_type("mul");
    proto::OpDesc::Var* mul_x = mul->add_inputs();
    mul_x->set_parameter("X");
    mul_x->add_arguments()->assign("x");
    proto::OpDesc::Var* mul_y = mul->add_inputs();
    mul_y->set_parameter("Y");
    mul_y->add_arguments()->assign("w");
    proto::OpDesc::Var* mul_out = mul->add_outputs();
    mul_out->set_parameter("Out");
    mul_out->add_arguments()->assign("mul_out");
    proto::OpDesc* ewadd = pattern->add_ops();
    ewadd->set_type("elementwise_add");
    proto::OpDesc::Var* ewadd_x = ewadd->add_inputs();
    ewadd_x->set_parameter("X");
    ewadd_x->add_arguments()->assign("mul_out");
    proto::OpDesc::Var* ewadd_y = ewadd->add_inputs();
    ewadd_y->set_parameter("Y");
    ewadd_y->add_arguments()->assign("b");
    proto::OpDesc::Var* ewadd_out = ewadd->add_outputs();
    ewadd_out->set_parameter("Out");
    ewadd_out->add_arguments()->assign("ewadd_out");
    proto::OpDesc* relu = nullptr;
    proto::BlockDesc* replace = pass_desc->mutable_replace()->add_blocks();
    replace->set_idx(0);
    replace->set_parent_idx(0);
    proto::OpDesc* fc = replace->add_ops();
    fc->set_type("fc");
    proto::OpDesc::Var* fc_x = fc->add_inputs();
    fc_x->set_parameter("Input");
    fc_x->add_arguments()->assign("x");
    proto::OpDesc::Var* fc_w = fc->add_inputs();
    fc_w->set_parameter("W");
    fc_w->add_arguments()->assign("w");
    proto::OpDesc::Var* fc_b = fc->add_inputs();
    fc_b->set_parameter("Bias");
    fc_b->add_arguments()->assign("b");
    proto::OpDesc::Var* fc_out = fc->add_outputs();
    fc_out->set_parameter("Out");
    fc_out->add_arguments()->assign("fc_out");
    for (const char* var : {"x", "w", "b", "fc_out"}) {
      proto::PassDesc::VarMap* var_map = pass_desc->add_var_maps();
      var_map->set_pattern_var(var);
      var_map->set_replace_var(var);
    }
    proto::PassDesc::AttrMap* attr_map = pass_desc->add_attr_maps();
    attr_map->set_pattern_op_idx(0);
    attr_map->set_pattern_name("x_num_col_dims");
    attr_map->set_replace_op_idx(0);
    attr_map->set_replace_name("in_num_col_dims");
    if (with_relu) {
      relu = pattern->add_ops();
      relu->set_type("relu");
      proto::OpDesc::Var* relu_x = relu->add_inputs();
      relu_x->set_parameter("X");
      relu_x->add_arguments()->assign("ewadd_out");
      proto::OpDesc::Var* relu_out = relu->add_outputs();
      relu_out->set_parameter("Out");
      relu_out->add_arguments()->assign("relu_out");
      pass_desc->mutable_var_maps(3)->set_pattern_var("relu_out");
      proto::OpDesc::Attr* attr = fc->add_attrs();
      attr->set_name("activation_type");
      attr->set_type(proto::AttrType::STRING);
      attr->set_s("relu");
    } else {
      pass_desc->mutable_var_maps(3)->set_pattern_var("ewadd_out");
    }
  }
  return multi_pass_desc;
}

proto::MultiPassDesc generate_multi_add_to_addn() {
  proto::MultiPassDesc multi_pass_desc;
  proto::PassDesc* pass_desc = multi_pass_desc.add_pass_descs();
  proto::BlockDesc* pattern = pass_desc->mutable_pattern()->add_blocks();
  proto::OpDesc* ewadd_0 = pattern->add_ops();
  ewadd_0->set_type("elementwise_add");
  proto::OpDesc::Var* ewadd_0_x = ewadd_0->add_inputs();
  ewadd_0_x->set_parameter("X");
  ewadd_0_x->add_arguments()->assign("a");
  proto::OpDesc::Var* ewadd_0_y = ewadd_0->add_inputs();
  ewadd_0_y->set_parameter("Y");
  ewadd_0_y->add_arguments()->assign("b");
  proto::OpDesc::Var* ewadd_0_out = ewadd_0->add_outputs();
  ewadd_0_out->set_parameter("Out");
  ewadd_0_out->add_arguments()->assign("ewadd_out_0");
  proto::OpDesc* ewadd_1 = pattern->add_ops();
  ewadd_1->set_type("elementwise_add");
  proto::OpDesc::Var* ewadd_1_x = ewadd_1->add_inputs();
  ewadd_1_x->set_parameter("X");
  ewadd_1_x->add_arguments()->assign("ewadd_out_0");
  proto::OpDesc::Var* ewadd_1_y = ewadd_1->add_inputs();
  ewadd_1_y->set_parameter("Y");
  ewadd_1_y->add_arguments()->assign("c");
  proto::OpDesc::Var* ewadd_1_out = ewadd_1->add_outputs();
  ewadd_1_out->set_parameter("Out");
  ewadd_1_out->add_arguments()->assign("ewadd_out_1");
  proto::BlockDesc* replace = pass_desc->mutable_replace()->add_blocks();
  proto::OpDesc* addn = replace->add_ops();
  addn->set_type("add_n");
  proto::OpDesc::Var* addn_x = addn->add_inputs();
  addn_x->set_parameter("X");
  addn_x->add_arguments()->assign("a");
  addn_x->add_arguments()->assign("b");
  addn_x->add_arguments()->assign("c");
  proto::OpDesc::Var* addn_out = addn->add_outputs();
  addn_out->set_parameter("Out");
  addn_out->add_arguments()->assign("addn_out");
  for (const char* var : {"a", "b", "c", "ewadd_out_1"}) {
    proto::PassDesc::VarMap* var_map = pass_desc->add_var_maps();
    var_map->set_pattern_var(var);
    var_map->set_replace_var(var);
  }
  pass_desc->mutable_var_maps(3)->set_replace_var("addn_out");
  return multi_pass_desc;
}

proto::MultiPassDesc generate_combine_matmul() {
  proto::MultiPassDesc multi_pass_desc;
  proto::PassDesc* pass_desc = multi_pass_desc.add_pass_descs();
  proto::BlockDesc* pattern = pass_desc->mutable_pattern()->add_blocks();
  proto::OpDesc* matmul_0 = pattern->add_ops();
  matmul_0->set_type("matmul");
  proto::OpDesc::Var* matmul_0_x = matmul_0->add_inputs();
  matmul_0_x->set_parameter("X");
  matmul_0_x->add_arguments()->assign("a");
  proto::OpDesc::Var* matmul_0_y = matmul_0->add_inputs();
  matmul_0_y->set_parameter("Y");
  matmul_0_y->add_arguments()->assign("b");
  proto::OpDesc::Var* matmul_0_out = matmul_0->add_outputs();
  matmul_0_out->set_parameter("Out");
  matmul_0_out->add_arguments()->assign("matmul_out_0");
  proto::OpDesc* matmul_1 = pattern->add_ops();
  matmul_1->set_type("matmul");
  proto::OpDesc::Var* matmul_1_x = matmul_1->add_inputs();
  matmul_1_x->set_parameter("X");
  matmul_1_x->add_arguments()->assign("a");
  proto::OpDesc::Var* matmul_1_y = matmul_1->add_inputs();
  matmul_1_y->set_parameter("Y");
  matmul_1_y->add_arguments()->assign("c");
  proto::OpDesc::Var* matmul_1_out = matmul_1->add_outputs();
  matmul_1_out->set_parameter("Out");
  matmul_1_out->add_arguments()->assign("matmul_out_1");
  proto::BlockDesc* replace = pass_desc->mutable_replace()->add_blocks();
  proto::OpDesc* concat = replace->add_ops();
  concat->set_type("concat");
  proto::OpDesc::Var* concat_x = concat->add_inputs();
  concat_x->set_parameter("X");
  concat_x->add_arguments()->assign("b");
  concat_x->add_arguments()->assign("c");
  proto::OpDesc::Var* concat_out = concat->add_outputs();
  concat_out->set_parameter("Out");
  concat_out->add_arguments()->assign("concat_out");
  proto::OpDesc* matmul = replace->add_ops();
  matmul->set_type("matmul");
  proto::OpDesc::Var* matmul_x = matmul->add_inputs();
  matmul_x->set_parameter("X");
  matmul_x->add_arguments()->assign("a");
  proto::OpDesc::Var* matmul_y = matmul->add_inputs();
  matmul_y->set_parameter("Y");
  matmul_y->add_arguments()->assign("concat_out");
  proto::OpDesc::Var* matmul_out = matmul->add_outputs();
  matmul_out->set_parameter("Out");
  matmul_out->add_arguments()->assign("matmul_out");
  proto::OpDesc* slice_0 = replace->add_ops();
  slice_0->set_type("slice");
  proto::OpDesc::Var* slice_0_x = slice_0->add_inputs();
  slice_0_x->set_parameter("X");
  slice_0_x->add_arguments()->assign("matmul_out");
  proto::OpDesc::Var* slice_0_out = slice_0->add_outputs();
  slice_0_out->set_parameter("Out");
  slice_0_out->add_arguments()->assign("slice_out_0");
  proto::OpDesc* slice_1 = replace->add_ops();
  slice_1->set_type("slice");
  proto::OpDesc::Var* slice_1_x = slice_1->add_inputs();
  slice_1_x->set_parameter("X");
  slice_1_x->add_arguments()->assign("matmul_out");
  proto::OpDesc::Var* slice_1_out = slice_1->add_outputs();
  slice_1_out->set_parameter("Out");
  slice_1_out->add_arguments()->assign("slice_out_1");
  for (const char* var : {"a", "b", "c", "matmul_out_0", "matmul_out_1"}) {
    proto::PassDesc::VarMap* var_map = pass_desc->add_var_maps();
    var_map->set_pattern_var(var);
    var_map->set_replace_var(var);
  }
  pass_desc->mutable_var_maps(3)->set_replace_var("slice_out_0");
  pass_desc->mutable_var_maps(4)->set_replace_var("slice_out_1");
  return multi_pass_desc;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_GENERATE_PASS(generate_fc_fuse,
                       paddle::framework::ir::generate_fc_fuse);
REGISTER_GENERATE_PASS(generate_multi_add_to_addn,
                       paddle::framework::ir::generate_multi_add_to_addn);
REGISTER_GENERATE_PASS(generate_combine_matmul,
                       paddle::framework::ir::generate_combine_matmul);

namespace paddle {
namespace framework {
namespace ir {

TEST(GeneratePass, construct_with_string) {
  std::string binary_str;
  generate_fc_fuse().SerializeToString(&binary_str);
  GeneratePass generate_pass(binary_str);
}

TEST(GeneratePass, generate_fc_fuse) {
  // inputs                     operator            output
  // --------------------------------------------------------
  // (a, filters_0 bias_0)      conv2d           -> conv2d_out
  // conv2d_out                 relu             -> relu_out_0
  // (relu_out_0, weights_0)    mul              -> mul_out_0
  // (mul_out_0, bias_1)        elementwise_add  -> add_out_0
  // add_out_0                  relu             -> relu_out_1
  // (relu_out_1, weights_1)    mul              -> mul_out_1
  // (mul_out_1, bias_2)        elementwise_add  -> add_out_1
  Layers layers;
  auto* a = layers.data("a");
  auto* filters_0 = layers.data("conv2d_filters_0", {}, true);
  auto* bias_0 = layers.data("conv2d_bias_0", {}, true);
  auto* conv2d_out = layers.conv2d(a, filters_0, bias_0, false);
  auto* relu_out_0 = layers.relu(conv2d_out);
  auto* weights_0 = layers.data("weights_0", {}, true);
  auto* mul_out_0 = layers.mul(relu_out_0, weights_0);
  auto* bias_1 = layers.data("bias_1", {}, true);
  auto* add_out_0 = layers.elementwise_add(mul_out_0, bias_1, nullptr, 1);
  auto* relu_out_1 = layers.relu(add_out_0);
  auto* weights_1 = layers.data("weights_1", {}, true);
  auto* mul_out_1 = layers.mul(relu_out_1, weights_1);
  auto* bias_2 = layers.data("bias_2", {}, true);
  auto* add_out_1 = layers.elementwise_add(mul_out_1, bias_2, nullptr, 1);
  VLOG(4) << add_out_1;

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  auto pass = PassRegistry::Instance().Get("generate_fc_fuse");
  int num_nodes_before = graph->Nodes().size();
  int num_mul_nodes_before = GetNumOpNodes(graph, "mul");
  VLOG(3) << DebugString(graph);

  graph.reset(pass->Apply(graph.release()));
  int num_nodes_after = graph->Nodes().size();
  int num_fc_nodes_after = GetNumOpNodes(graph, "fc");
  VLOG(3) << DebugString(graph);

  PADDLE_ENFORCE_EQ(num_nodes_before, num_nodes_after + 6,
                    platform::errors::InvalidArgument(
                        "num_nodes_before=%d, num_nodes_after=%d.",
                        num_nodes_before, num_nodes_after));
  PADDLE_ENFORCE_EQ(num_fc_nodes_after, 2,
                    platform::errors::InvalidArgument("num_fc_nodes_after=%d.",
                                                      num_fc_nodes_after));
  PADDLE_ENFORCE_EQ(num_mul_nodes_before, num_fc_nodes_after,
                    platform::errors::InvalidArgument(
                        "num_mul_nodes_before=%d, num_fc_nodes_after=%d.",
                        num_mul_nodes_before, num_fc_nodes_after));
}

TEST(GeneratePass, generate_multi_add_to_addn) {
  // inputs                     operator            output
  // --------------------------------------------------------
  // (a, b)                     elementwise_add  -> add_out_0
  // (add_out_0, c)             elementwise_add  -> add_out_1
  Layers layers;
  auto* a = layers.data("a");
  auto* b = layers.data("b");
  auto* c = layers.data("c");
  auto* add_out_0 = layers.elementwise_add(a, b);
  layers.elementwise_add(add_out_0, c);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  auto pass = PassRegistry::Instance().Get("generate_multi_add_to_addn");
  int num_nodes_before = graph->Nodes().size();
  int num_add_nodes_before = GetNumOpNodes(graph, "elementwise_add");
  VLOG(3) << DebugString(graph);

  graph.reset(pass->Apply(graph.release()));
  int num_nodes_after = graph->Nodes().size();
  int num_addn_nodes_after = GetNumOpNodes(graph, "add_n");
  VLOG(3) << DebugString(graph);

  PADDLE_ENFORCE_EQ(num_nodes_before, num_nodes_after + 2,
                    platform::errors::InvalidArgument(
                        "num_nodes_before=%d, num_nodes_after=%d.",
                        num_nodes_before, num_nodes_after));
  PADDLE_ENFORCE_EQ(num_addn_nodes_after, 1,
                    platform::errors::InvalidArgument(
                        "num_addn_nodes_after=%d.", num_addn_nodes_after));
  PADDLE_ENFORCE_EQ(num_add_nodes_before, num_addn_nodes_after + 1,
                    platform::errors::InvalidArgument(
                        "num_add_nodes_before=%d, num_addn_nodes_after=%d.",
                        num_add_nodes_before, num_addn_nodes_after));
}

TEST(GeneratePass, generate_combine_matmul) {
  // inputs                     operator            output
  // --------------------------------------------------------
  // (a, b)                     matmul           -> matmul_out_0
  // (a, c)                     matmul           -> matmul_out_1
  Layers layers;
  auto* a = layers.data("a");
  auto* b = layers.data("b");
  auto* c = layers.data("c");
  layers.matmul(a, b);
  layers.matmul(a, c);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  auto pass = PassRegistry::Instance().Get("generate_combine_matmul");
  int num_nodes_before = graph->Nodes().size();
  int num_matmul_nodes_before = GetNumOpNodes(graph, "matmul");
  VLOG(3) << DebugString(graph);

  graph.reset(pass->Apply(graph.release()));
  int num_nodes_after = graph->Nodes().size();
  int num_matmul_nodes_after = GetNumOpNodes(graph, "matmul");
  VLOG(3) << DebugString(graph);

  PADDLE_ENFORCE_EQ(num_nodes_before, num_nodes_after - 4,
                    platform::errors::InvalidArgument(
                        "num_nodes_before=%d, num_nodes_after=%d.",
                        num_nodes_before, num_nodes_after));
  PADDLE_ENFORCE_EQ(num_matmul_nodes_after, 1,
                    platform::errors::InvalidArgument(
                        "num_matmul_nodes_after=%d.", num_matmul_nodes_after));
  PADDLE_ENFORCE_EQ(
      num_matmul_nodes_before, num_matmul_nodes_after + 1,
      platform::errors::InvalidArgument(
          "num_matmul_nodes_before=%d, num_matmul_nodes_after=%d.",
          num_matmul_nodes_before, num_matmul_nodes_after));
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
