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
    proto::BlockDesc* replace = pass_desc->mutable_pattern()->add_blocks();
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
    fc_out->set_parameter("Output");
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
  return multi_pass_desc;
}

proto::MultiPassDesc generate_combine_matmul() {
  proto::MultiPassDesc multi_pass_desc;
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

void AddVarToScope(Scope* param_scope, const std::string& name,
                   const DDim& dims) {
  auto* tensor = param_scope->Var(name)->GetMutable<LoDTensor>();
  tensor->Resize(dims);
  tensor->mutable_data<float>(platform::CPUPlace());
}

Scope* CreateParamScope() {
  auto param_scope = new Scope();
  AddVarToScope(param_scope, "conv2d_filters_0", {});
  AddVarToScope(param_scope, "conv2d_bias_0", {});
  AddVarToScope(param_scope, "weights_0", {});
  AddVarToScope(param_scope, "weights_1", {});
  AddVarToScope(param_scope, "bias_1", {});
  AddVarToScope(param_scope, "bias_2", {});
  return param_scope;
}

TEST(FCFusePass, basic) {
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

}  // namespace ir
}  // namespace framework
}  // namespace paddle
