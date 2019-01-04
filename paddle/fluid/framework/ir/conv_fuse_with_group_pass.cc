// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/conv_fuse_with_group_pass.h"
#include <algorithm>
#include <functional>
#include <numeric>
#include <string>
#include <vector>
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/strided_memcpy.h"
#include "paddle/fluid/platform/enforce.h"

#define GROUP_COUNT 7

namespace paddle {
namespace framework {
namespace ir {

const Node* ConvFuseWithGroupPass::IsSameSingleInput(std::vector<Node*>& nodes,
                                                     std::string type) const {
  const Node* input_node = nullptr;

  for (auto it = nodes.begin(); it != nodes.end(); ++it) {
    auto node = *it;
    if (node->IsOp() && (node->inputs.size() == 1) &&
        (!node->Name().compare(type))) {
      if (input_node == nullptr) {
        input_node = node->inputs[0];
      } else {
        if (node->inputs[0] != input_node) {
          input_node = nullptr;
          break;
        }
      }
    }
  }

  // Return the same input node.
  return input_node;
}

void ConvFuseWithGroupPass::GetSpeicalOpNodes(
    std::vector<Node*>& nodes, std::string type,
    std::vector<Node*>* dst_nodes) const {
  for (auto it = nodes.begin(); it != nodes.end(); ++it) {
    auto node = *it;
    if (node->IsOp() && (!node->Name().compare(type))) {
      dst_nodes->push_back(node);
    }
  }
}

const Node* ConvFuseWithGroupPass::GetSpeicalVarNode(std::vector<Node*>& nodes,
                                                     std::string name) const {
  const Node* out_node = nullptr;

  for (auto it = nodes.begin(); it != nodes.end(); ++it) {
    auto node = *it;
    if (node->IsVar() && (!node->Name().compare(name))) {
      out_node = node;
      break;
    }
  }

  return out_node;
}

// Sort the nodes by the name.
void ConvFuseWithGroupPass::SortNodes(std::vector<Node*>& nodes) const {
  std::sort(nodes.begin(), nodes.end(), [](Node* node1, Node* node2) {
    return node1->outputs[0]->Name().compare(node2->outputs[0]->Name()) < 0;
  });
}

// Obtain the weights and biases nodes from convlution nodes.
const Node* ConvFuseWithGroupPass::GetConvWeightBiasNodes(
    const std::vector<Node*>& nodes, std::vector<const Node*>& weights_node,
    std::vector<const Node*>& biases_node, int block, int group, int layer,
    std::string type) const {
  const Node* conv_node = nullptr;

  for (size_t item = 1; item <= GROUP_COUNT; item++) {
    size_t index = item % GROUP_COUNT;
    std::string weight_name, bias_name;

    // Check the convlution type if it is as "conv" or "project" and set the
    // weight and bias name with special block, group and layer.
    if (!type.compare("conv")) {
      if (block == 0) {
        weight_name = "patch_" + std::to_string(index) + "_" + type +
                      std::to_string(block) + "_weights";
        bias_name = "patch_" + std::to_string(index) + "_" + type +
                    std::to_string(block) + "_biases";
      } else {
        weight_name = "patch_" + std::to_string(index) + "_" + type +
                      std::to_string(block) + "-" + std::to_string(group) +
                      "_" + std::to_string(layer) + "_weights";
        bias_name = "patch_" + std::to_string(index) + "_" + type +
                    std::to_string(block) + "-" + std::to_string(group) + "_" +
                    std::to_string(layer) + "_biases";
      }
    } else {
      weight_name = "patch_" + std::to_string(index) + "_" + type +
                    std::to_string(block) + "_weights";
      bias_name = "patch_" + std::to_string(index) + "_" + type +
                  std::to_string(block) + "_biases";
    }

    for (auto it = nodes.begin(); it != nodes.end(); ++it) {
      const Node* node = *it;
      if (node->IsOp() && (node->inputs.size() == 3)) {
        for (auto input_node : node->inputs) {
          if (!input_node->Name().compare(weight_name)) {
            // To get the first conv node for using the attribute's duplication.
            if (conv_node == nullptr) {
              for (auto biasNode : node->inputs) {
                if (!biasNode->Name().compare(bias_name)) {
                  conv_node = node;
                }
              }
            }

            // Push back the conv node into weights vector.
            weights_node.push_back(input_node);
          } else if (!input_node->Name().compare(bias_name)) {
            biases_node.push_back(input_node);
          }
        }
      }
    }
  }
  return conv_node;
}

// Create the variable Node
const Node* ConvFuseWithGroupPass::CreateVarNode(
    const std::unique_ptr<ir::Graph>& graph, Scope* scope, std::string name,
    DDim dims, bool persistable) const {
  const Node* node = nullptr;
  VarDesc desc(patterns::PDNodeName(name_scope_, name));
  if (persistable) {
    desc.SetPersistable(true);
  }
  node = graph->CreateVarNode(&desc);

  if (persistable) {
    // If the variable is persistable, then it need be allocate the memory and
    // set up the dimsion.
    auto* tensor = scope->Var(node->Name())->GetMutable<LoDTensor>();
    tensor->Resize(dims);
    tensor->mutable_data<float>(platform::CPUPlace());
  }

  return node;
}

const Node* ConvFuseWithGroupPass::CreateConvVarNode(
    const std::unique_ptr<ir::Graph>& graph, Scope* scope,
    const std::vector<const Node*>& nodes, int block, int group, int layer,
    bool persistable, std::string type, std::string usage, int64_t axis) const {
  // To speicalize the variable name of convlution node.
  std::string name = type + "_" + usage + "_" + std::to_string(block) + "_" +
                     std::to_string(group) + "_" + std::to_string(layer);
  const Node* node = nullptr;

  if (persistable && !nodes.empty()) {
    // If the persistable attribute is configured, it need update the data by
    // group size.
    auto* tensor = scope->FindVar(nodes[0]->Name())->GetMutable<LoDTensor>();

    // Update the dimsion information
    auto dims = tensor->dims();
    framework::set(dims, axis, framework::get(dims, axis) * nodes.size());

    node = CreateVarNode(graph, scope, name, dims, persistable);
    tensor = scope->Var(node->Name())->GetMutable<LoDTensor>();

    // Update the data, such as weights or biases.
    for (size_t index = 0, out_offset = 0; index < nodes.size(); index++) {
      auto input_tensor =
          scope->Var(nodes[index]->Name())->GetMutable<LoDTensor>();
      auto out_stride = framework::stride_numel(dims);
      auto in_stride = framework::stride_numel(input_tensor->dims());

      paddle::operators::StridedNumelCopyWithAxis<float>(
          paddle::platform::CPUDeviceContext(), axis,
          tensor->data<float>() + out_offset, out_stride,
          (const float*)(input_tensor->data<float>()), in_stride,
          in_stride[axis]);
      out_offset += in_stride[axis];
    }
  } else {
    // To create the variable with temporary attribute for operator's output
    // usage.
    node = CreateVarNode(graph, scope, name);
  }

  return node;
}

const Node* ConvFuseWithGroupPass::CreateConvOpWithGroup(
    const std::unique_ptr<ir::Graph>& graph, Scope* scope,
    const Node* input_mode, const Node* conv_node,
    const std::vector<const Node*>& weights_node,
    const std::vector<const Node*>& biases_node, int block, int group,
    int layer, std::string type) const {
  const Node* output_node = nullptr;

  // To create weight and bias variable node for convlution node.
  auto weight_node = CreateConvVarNode(graph, scope, weights_node, block, group,
                                       layer, true, type, "weight");
  auto bias_node = CreateConvVarNode(graph, scope, biases_node, block, group,
                                     layer, true, type, "bias");
  // Notice:
  //  When created the temporary variable, the nodes parameters are unused.
  //  So it will be special as any value, such as "biases_node" or
  //  "weights_node".
  output_node = CreateConvVarNode(graph, scope, biases_node, block, group,
                                  layer, false, type, "output");

  OpDesc desc;

  // Configure the Input and output nodes.
  desc.SetInput("Input", std::vector<std::string>({input_mode->Name()}));
  desc.SetInput("Filter", std::vector<std::string>({weight_node->Name()}));
  desc.SetInput("Bias", std::vector<std::string>({bias_node->Name()}));
  desc.SetOutput("Output", std::vector<std::string>({output_node->Name()}));
  desc.SetType("conv2d");

  // duplicate the original attributes and update the groups attribute for the
  // new convlution operator.
  for (auto& attr : conv_node->Op()->GetAttrMap()) {
    if (attr.first == "groups") {
      desc.SetAttr(attr.first, static_cast<int>(weights_node.size()));
    } else {
      desc.SetAttr(attr.first, attr.second);
    }
  }

  // To create convlution operator node.
  auto conv_out_node = graph->CreateOpNode(&desc);

  // Link variable and operator nodes.
  IR_NODE_LINK_TO(const_cast<Node*>(input_mode),
                  const_cast<Node*>(conv_out_node));
  IR_NODE_LINK_TO(const_cast<Node*>(weight_node),
                  const_cast<Node*>(conv_out_node));
  IR_NODE_LINK_TO(const_cast<Node*>(bias_node),
                  const_cast<Node*>(conv_out_node));
  IR_NODE_LINK_TO(const_cast<Node*>(conv_out_node),
                  const_cast<Node*>(output_node));

  return output_node;
}

const Node* ConvFuseWithGroupPass::GetKeyEltwiseAddOpNode(
    const std::vector<Node*>& nodes, int block, int group) const {
  const Node* eltwise_add_node = nullptr;
  std::string eltwise_add_name;

  // Specialize the eltwise_add operator's match string.
  eltwise_add_name = "patch_0_shortcut" + std::to_string(block) + "-" +
                     std::to_string(group) + ".add.output";

  // Enumerate all nodes and get the eltwise_add operator.
  for (auto it = nodes.begin(); it != nodes.end(); ++it) {
    const Node* node = *it;
    if (node->IsOp() && (node->outputs.size() == 1)) {
      for (auto output_node : node->outputs) {
        if (std::string::npos != output_node->Name().find(eltwise_add_name)) {
          eltwise_add_node = node;
        }
      }
    }
  }

  return eltwise_add_node;
}

const Node* ConvFuseWithGroupPass::CreateEltwiseAddOp(
    const std::unique_ptr<ir::Graph>& graph, Scope* scope,
    const Node* input_x_mode, const Node* input_y_mode,
    const Node* eltwise_add_node, int block, int group) const {
  std::string name =
      "eltwise_add_out_" + std::to_string(block) + "_" + std::to_string(group);

  // To create the eltwise_add output variable node.
  const Node* output_node = CreateVarNode(graph, scope, name);

  // To allocate the eltwise_add operator and link all related nodes.
  OpDesc desc;

  desc.SetInput("X", std::vector<std::string>({input_x_mode->Name()}));
  desc.SetInput("Y", std::vector<std::string>({input_y_mode->Name()}));
  desc.SetOutput("Out", std::vector<std::string>({output_node->Name()}));
  desc.SetType("elementwise_add");

  for (auto& attr : eltwise_add_node->Op()->GetAttrMap()) {
    desc.SetAttr(attr.first, attr.second);
  }
  auto eltwise_add_out_node = graph->CreateOpNode(&desc);

  IR_NODE_LINK_TO(const_cast<Node*>(input_x_mode),
                  const_cast<Node*>(eltwise_add_out_node));
  IR_NODE_LINK_TO(const_cast<Node*>(input_y_mode),
                  const_cast<Node*>(eltwise_add_out_node));
  IR_NODE_LINK_TO(const_cast<Node*>(eltwise_add_out_node),
                  const_cast<Node*>(output_node));

  return output_node;
}

const Node* ConvFuseWithGroupPass::CreateResiualNetWithGroup(
    const std::unique_ptr<ir::Graph>& graph, Scope* scope,
    const std::vector<Node*>& conv_nodes,
    const std::vector<Node*>& eltwise_add_nodes, const Node* input_mode,
    int block, int group, bool is_project) const {
  const Node* output_node = nullptr;

  output_node = input_mode;

  // To create two convlution operators and connect them.
  for (size_t layer = 1; layer < 3; layer++) {
    std::vector<const Node*> weights_node;
    std::vector<const Node*> biases_node;
    const Node* conv_node = GetConvWeightBiasNodes(
        conv_nodes, weights_node, biases_node, block, group, layer);
    PADDLE_ENFORCE_EQ(weights_node.size(), biases_node.size());
    PADDLE_ENFORCE(conv_node != nullptr);

    output_node =
        CreateConvOpWithGroup(graph, scope, output_node, conv_node,
                              weights_node, biases_node, block, group, layer);
    weights_node.clear();
    biases_node.clear();
  }

  // Check if the "project" status is enabled, then new the convlution node for
  // it.
  if (is_project) {
    std::vector<const Node*> weights_node;
    std::vector<const Node*> biases_node;
    const Node* conv_node = GetConvWeightBiasNodes(
        conv_nodes, weights_node, biases_node, block - 1, group, 1, "project");
    PADDLE_ENFORCE_EQ(weights_node.size(), biases_node.size());
    PADDLE_ENFORCE(conv_node != nullptr);

    input_mode =
        CreateConvOpWithGroup(graph, scope, input_mode, conv_node, weights_node,
                              biases_node, block - 1, group, 1, "project");
    weights_node.clear();
    biases_node.clear();
  }

  // To get the typical eltwise_add node for duplicating the attributes usage
  // and create "eltwise_add" operator node.
  const Node* eltwise_add_node =
      GetKeyEltwiseAddOpNode(eltwise_add_nodes, block, group);
  output_node = CreateEltwiseAddOp(graph, scope, input_mode, output_node,
                                   eltwise_add_node, block, group);

  return output_node;
}

int ConvFuseWithGroupPass::GetConvOutputChannelsNum(Scope* scope) const {
  int channels_num = 0;

  // Get the weight channels of last convlution operator node.
  auto* tensor =
      scope->FindVar("patch_0_conv4-3_2_weights")->GetMutable<LoDTensor>();
  channels_num = framework::get(tensor->dims(), 0);

  return channels_num;
}

void ConvFuseWithGroupPass::RedirectSplitPoolOpNodes(
    const std::unique_ptr<ir::Graph>& graph, Scope* scope,
    std::vector<Node*>& split_nodes, std::vector<Node*>& pool_nodes,
    const Node* conv_mode, int conv_channels_num) const {
  // Sort the split and pool nodes to locate the special node easy.
  SortNodes(split_nodes);
  SortNodes(pool_nodes);

  PADDLE_ENFORCE_EQ(split_nodes.size(), pool_nodes.size());

  int length = std::min(split_nodes.size(), pool_nodes.size());
  for (size_t index = 0; index < (size_t)length; index++) {
    Node* split_node = split_nodes[index];
    Node* pool_node = pool_nodes[index];

    // To update the "Split" node's attributes.
    for (auto& attr : split_node->Op()->GetAttrMap()) {
      if (attr.first == "sections") {
        std::vector<int> sections =
            boost::get<std::vector<int>>(split_node->Op()->GetAttr("sections"));
        int sum = std::accumulate(sections.begin(), sections.end(), 0);
        int channels = conv_channels_num * length;

        std::for_each(sections.begin(), sections.end(),
                      [&](int& value) { value = value * channels / sum; });
        split_node->Op()->SetAttr(attr.first, sections);
      } else {
        split_node->Op()->SetAttr(attr.first, attr.second);
      }
    }

    // To break down the link between split and convlution(without group
    // attribute) node and connect split and pool.
    for (auto var_node : split_node->outputs) {
      for (auto op_node : var_node->outputs) {
        if (!op_node->Name().compare("conv2d")) {
          const std::vector<std::string> x_inputs = pool_node->Op()->Input("X");
          if (!x_inputs.empty()) {
            for (auto it_pool = pool_node->inputs.begin();
                 it_pool != pool_node->inputs.end();) {
              auto input_node = *it_pool;
              if (std::find(x_inputs.begin(), x_inputs.end(),
                            input_node->Name()) != x_inputs.end()) {
                for (auto it = input_node->outputs.begin();
                     it != input_node->outputs.end();) {
                  if (*it == pool_node) {
                    it = input_node->outputs.erase(it);
                    it_pool = pool_node->inputs.erase(it_pool);
                    break;
                  } else {
                    it++;
                  }
                }
              } else {
                it_pool++;
              }
            }
          }

          for (auto it = op_node->inputs.begin();
               it != op_node->inputs.end();) {
            if (*it == var_node) {
              it = op_node->inputs.erase(it);
            } else {
              it++;
            }
          }
          var_node->outputs.clear();
          pool_node->Op()->SetInput(
              "X", std::vector<std::string>({var_node->Name()}));
          IR_NODE_LINK_TO(const_cast<Node*>(var_node),
                          const_cast<Node*>(pool_node));
          break;
        }
      }
    }

    // To connect the split and convlution(with group) operator.
    for (auto node : split_node->inputs) {
      for (auto it = node->outputs.begin(); it != node->outputs.end();) {
        if (*it == split_node) {
          it = node->outputs.erase(it);
          break;
        } else {
          it++;
        }
      }
    }
    split_node->inputs.clear();
    split_node->Op()->SetInput("X",
                               std::vector<std::string>({conv_mode->Name()}));
    IR_NODE_LINK_TO(const_cast<Node*>(conv_mode),
                    const_cast<Node*>(split_node));
  }
}

std::unique_ptr<ir::Graph> ConvFuseWithGroupPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  PADDLE_ENFORCE(graph.get());
  FusePassBase::Init(name_scope_, graph.get());

  auto* scope = param_scope();
  PADDLE_ENFORCE(scope);

  // To obtain all operator nodes.
  std::vector<Node*> nodes = TopologySortOperations(*graph);

  // To obtain all split nodes.
  std::vector<Node*> split_nodes;
  GetSpeicalOpNodes(nodes, "split", &split_nodes);

  // Check all split nodes' input nodes are same.
  const Node* input_node = IsSameSingleInput(split_nodes, "split");
  if (input_node == nullptr || split_nodes.size() != GROUP_COUNT) {
    return graph;
  }

  // To obtain all convlution nodes.
  std::vector<Node*> conv_nodes;
  GetSpeicalOpNodes(nodes, "conv2d", &conv_nodes);

  // To obtain all eltwise_add nodes.
  std::vector<Node*> eltwise_add_nodes;
  GetSpeicalOpNodes(nodes, "elementwise_add", &eltwise_add_nodes);

  int block_index;

  // Block 0
  std::vector<const Node*> weights_node;
  std::vector<const Node*> biases_node;
  block_index = 0;

  const Node* conv_node = GetConvWeightBiasNodes(conv_nodes, weights_node,
                                                 biases_node, block_index);

  PADDLE_ENFORCE_EQ(weights_node.size(), biases_node.size());
  PADDLE_ENFORCE(conv_node != nullptr);

  input_node = CreateConvOpWithGroup(graph, scope, input_node, conv_node,
                                     weights_node, biases_node);

  weights_node.clear();
  biases_node.clear();

  // block 1
  block_index = 1;

  for (size_t group_index = 1; group_index < 4; group_index++) {
    input_node =
        CreateResiualNetWithGroup(graph, scope, conv_nodes, eltwise_add_nodes,
                                  input_node, block_index, group_index);
  }

  // block 2
  block_index = 2;
  for (size_t group_index = 1; group_index < 5; group_index++) {
    input_node = CreateResiualNetWithGroup(
        graph, scope, conv_nodes, eltwise_add_nodes, input_node, block_index,
        group_index, group_index == 1);
  }

  // block 3
  block_index = 3;
  for (size_t group_index = 1; group_index < 7; group_index++) {
    input_node = CreateResiualNetWithGroup(
        graph, scope, conv_nodes, eltwise_add_nodes, input_node, block_index,
        group_index, group_index == 1);
  }

  // block 4
  block_index = 4;
  for (size_t group_index = 1; group_index < 4; group_index++) {
    input_node = CreateResiualNetWithGroup(
        graph, scope, conv_nodes, eltwise_add_nodes, input_node, block_index,
        group_index, group_index == 1);
  }

  // To obtain all pool nodes.
  std::vector<Node*> pool_nodes;
  GetSpeicalOpNodes(nodes, "pool2d", &pool_nodes);

  // To redirect the split and pool nodes.
  int conv_channels_num = GetConvOutputChannelsNum(scope);
  RedirectSplitPoolOpNodes(graph, scope, split_nodes, pool_nodes, input_node,
                           conv_channels_num);

  // To remove all remnant nodes.
  std::unordered_set<const Node*> remove_nodes;
  std::for_each(conv_nodes.begin(), conv_nodes.end(), [&](Node* node) {
    remove_nodes.insert(std::make_move_iterator(node->inputs.begin()),
                        std::make_move_iterator(node->inputs.end()));
    remove_nodes.insert(std::make_move_iterator(node->outputs.begin()),
                        std::make_move_iterator(node->outputs.end()));
  });
  remove_nodes.insert(std::make_move_iterator(conv_nodes.begin()),
                      std::make_move_iterator(conv_nodes.end()));
  std::for_each(
      eltwise_add_nodes.begin(), eltwise_add_nodes.end(), [&](Node* node) {
        remove_nodes.insert(std::make_move_iterator(node->inputs.begin()),
                            std::make_move_iterator(node->inputs.end()));
        remove_nodes.insert(std::make_move_iterator(node->outputs.begin()),
                            std::make_move_iterator(node->outputs.end()));
      });
  remove_nodes.insert(std::make_move_iterator(eltwise_add_nodes.begin()),
                      std::make_move_iterator(eltwise_add_nodes.end()));
  GraphSafeRemoveNodes(graph.get(), remove_nodes);

  return graph;
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle
REGISTER_PASS(conv_fuse_with_group_pass,
              paddle::framework::ir::ConvFuseWithGroupPass);
