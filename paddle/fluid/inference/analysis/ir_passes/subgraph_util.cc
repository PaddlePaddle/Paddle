/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

/*
 * This file defines the the class to partition a graph.
 */

#include "paddle/fluid/inference/analysis/ir_passes/subgraph_util.h"
#include <algorithm>
#include <string>

namespace paddle {
namespace inference {
namespace analysis {
using framework::ir::Node;

std::vector<std::string> ExtractParameters(
    const std::unordered_set<Node *> &nodes) {
  // We can judge whether a variable is a parameter by
  // its presistable property, but sometimes the presistable
  // of the feed op output is true, so we have to identify it.
  std::vector<std::string> feed_outputs;
  for (const auto &node : nodes) {
    if (!node->IsOp()) continue;
    std::string op_type = node->Op()->Type();
    if (op_type == "feed" || op_type == "fetch") {
      std::vector<std::string> output_names = node->Op()->OutputArgumentNames();
      std::copy(output_names.begin(), output_names.end(),
                std::back_inserter(feed_outputs));
    }
  }

  std::vector<std::string> parameters;
  for (const auto &node : nodes) {
    if (!node->IsVar()) continue;
    if (node->Var()->Persistable() &&
        std::find(feed_outputs.begin(), feed_outputs.end(), node->Name()) ==
            feed_outputs.end()) {
      parameters.push_back(node->Name());
    }
  }
  return parameters;
}

void RenameAndGetOutputs(
    const std::vector<framework::ir::Node *> &subgraph_nodes,
    framework::BlockDesc *block_desc,
    const std::set<std::string> &input_names_with_id,
    std::set<std::string> *output_names_with_id,
    std::set<std::string> *output_names,
    std::unordered_map<std::string, std::string> *output_name_map,
    const std::unordered_map<std::string, framework::ir::Node *> &graph_var_map,
    bool is_trt) {
  //// In the normal case, the paddle-trt exists bug when runing the googlenet.
  // When there are more than two convolutions of 1 * 1 with the same input, the
  // paddle-tensorrt will do the merging optimization, which fuse those conv
  // into one conv, and then trigger bug. So,  We should use strategy to avoid
  // this optimization for the time being. This bug will be fixed in the future.
  std::unordered_map<std::string /*name*/, int /*ITensor_quote_num*/>
      same_hierarchy_conv2d_num_map;

  auto set_var_shape = [&](const std::string &arg_value) {
    auto arg_var_node = graph_var_map.find(arg_value);
    PADDLE_ENFORCE(arg_var_node != graph_var_map.end());
    auto *var_t = block_desc->Var(arg_value);
    var_t->SetShape(arg_var_node->second->Var()->GetShape());
  };

  for (size_t index = 0; index < block_desc->OpSize(); ++index) {
    framework::proto::OpDesc *op = block_desc->Op(index)->Proto();
    framework::OpDesc op_desc(*op, nullptr);
    auto correspond_node = subgraph_nodes[index];
    PADDLE_ENFORCE_EQ(correspond_node->Name(), op->type());

    std::unordered_map<std::string, size_t> var2id;
    std::unordered_map<std::string, framework::ir::Node *> in_vars;
    for (auto *in_var : correspond_node->inputs) {
      var2id[in_var->Name()] = in_var->id();
      in_vars[in_var->Name()] = in_var;
    }
    // rename for the input variables of op inside subgraph
    for (int i = 0; i < op->inputs_size(); i++) {
      // one input
      auto *in_var = op->mutable_inputs(i);
      std::vector<std::string> replaced_names;
      for (int k = 0; k < in_var->arguments_size(); k++) {  // all the arguments
        const std::string arg_value = in_var->arguments(k);
        const std::string arg_value_with_id =
            arg_value + std::to_string(var2id[arg_value]);

        bool is_var_in_graph = graph_var_map.count(arg_value);

        if (input_names_with_id.count(arg_value_with_id)) {
          replaced_names.push_back(arg_value);
        } else {
          replaced_names.push_back(arg_value_with_id);
        }
        if (is_var_in_graph) {
          set_var_shape(arg_value);
        }
      }
      in_var->clear_arguments();
      for (size_t k = 0; k < replaced_names.size(); k++) {
        in_var->add_arguments(replaced_names[k]);
      }
    }
    var2id.clear();
    for (auto out_var : correspond_node->outputs) {
      var2id[out_var->Name()] = out_var->id();
    }
    if (op_desc.Type() == "conv2d" && is_trt) {
      auto input_var_name = op_desc.Input("Input").front();
      auto filter_var_name = op_desc.Input("Filter").front();
      auto out_var_name = op_desc.Output("Output").front();
      auto filter_shape = in_vars[filter_var_name]->Var()->GetShape();
      const std::vector<int> strides =
          boost::get<std::vector<int>>(op_desc.GetAttr("strides"));
      const std::vector<int> paddings =
          boost::get<std::vector<int>>(op_desc.GetAttr("paddings"));
      if (same_hierarchy_conv2d_num_map[input_var_name] > 0) {
        (*output_names_with_id)
            .insert(out_var_name + std::to_string(var2id[out_var_name]));
        (*output_names).insert(out_var_name);
      } else if (filter_shape[2] == 1 && filter_shape[3] == 1 &&
                 strides[0] == 1 && strides[1] == 1 && paddings[0] == 0 &&
                 paddings[1] == 0) {
        same_hierarchy_conv2d_num_map[input_var_name] += 1;
      }
    }
    // rename for the output variables of op inside subgraph
    for (int i = 0; i < op->outputs_size(); i++) {
      framework::proto::OpDesc_Var *out_var = op->mutable_outputs(i);
      std::vector<std::string> replaced_names;
      for (int k = 0; k < out_var->arguments_size(); k++) {
        const std::string arg_value = out_var->arguments(k);
        const std::string arg_value_with_id =
            arg_value + std::to_string(var2id[arg_value]);

        bool is_var_in_graph = graph_var_map.count(arg_value);
        if (is_var_in_graph) {
          set_var_shape(arg_value);
        }

        if (output_names_with_id->count(arg_value_with_id)) {
          (*output_name_map)[arg_value] = arg_value_with_id;
        }
        replaced_names.push_back(arg_value_with_id);
      }
      out_var->clear_arguments();
      for (size_t k = 0; k < replaced_names.size(); k++) {
        out_var->add_arguments(replaced_names[k]);
      }
    }
  }
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
