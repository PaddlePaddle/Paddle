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

#include "paddle/cinn/hlir/pass/infershape.h"

#include "paddle/cinn/hlir/framework/node.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/pass/use_pass.h"
#include "paddle/cinn/hlir/pe/schedule.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace hlir {
namespace pass {

using common::Type;
using framework::Graph;
using framework::Node;
using framework::NodeData;
using framework::Operator;

using infershape_t = std::function<std::vector<framework::shape_t>(
    const std::vector<framework::shape_t>&, const framework::AttrMapType&)>;
using inferdtype_t = std::function<std::vector<Type>(
    const std::vector<Type>&, const framework::AttrMapType&)>;
using dtype_dict_t = absl::flat_hash_map<std::string, common::Type>;
using shape_dict_t = absl::flat_hash_map<std::string, framework::shape_t>;

void InferShape(Node* node,
                dtype_dict_t& dtype_dict,    // NOLINT
                shape_dict_t& shape_dict) {  // NOLINT
  VLOG(3) << "Begin InferShape of node " << node->id();
  auto op_infershape = Operator::GetAttrs<infershape_t>("infershape");
  auto op_inferdtype = Operator::GetAttrs<inferdtype_t>("inferdtype");
  CHECK(node) << "The node can not be nullptr.";

  auto product = [](const framework::shape_t& shape) {
    framework::dim_t numel = 1;
    std::for_each(shape.begin(), shape.end(), [&numel](framework::dim_t dim) {
      numel *= dim;
    });
    return numel;
  };

  std::vector<framework::shape_t> inputs_shape;
  std::vector<Type> inputs_dtype;
  for (auto& in_edge : node->inlinks_in_order()) {
    auto* source_node = in_edge->source()->safe_as<NodeData>();
    CHECK(source_node);
    CHECK(shape_dict.count(source_node->id()))
        << "No shape for " << source_node->id();
    CHECK(dtype_dict.count(source_node->id()))
        << "No dtype for " << source_node->id();
    inputs_shape.push_back(shape_dict[source_node->id()]);
    inputs_dtype.push_back(dtype_dict[source_node->id()]);

    CHECK(product(inputs_shape.back()))
        << node->id() << " 's Input Node " << source_node->id() << "["
        << utils::Join(inputs_shape.back(), ",")
        << "]'s size should not zero ! Please check.";
  }

  auto out_shape =
      op_infershape[node->op()](inputs_shape, node->attrs.attr_store);
  auto out_dtype =
      op_inferdtype[node->op()](inputs_dtype, node->attrs.attr_store);

  CHECK_GE(node->outlinks_in_order().size(), out_shape.size())
      << "The output number of node " << node->id() << " is "
      << node->outlinks_in_order().size()
      << " , which is smaller than the output shape size " << out_shape.size()
      << " . And the op type is " << node->op()->name;
  CHECK_GE(node->outlinks_in_order().size(), out_dtype.size())
      << "The output number of node " << node->id() << " is "
      << node->outlinks_in_order().size()
      << " , which is smaller than the output dtype size " << out_dtype.size()
      << " . And the op type is " << node->op()->name;

  int counter = 0;
  for (auto& out_edge : node->outlinks_in_order()) {
    auto* sink_node = out_edge->sink()->safe_as<NodeData>();
    CHECK(sink_node);

    VLOG(3) << "Infershape: " << sink_node->id() << " "
            << utils::Join(out_shape[counter], ",");
    shape_dict[sink_node->id()] = out_shape[counter];
    dtype_dict[sink_node->id()] = out_dtype[counter];

    CHECK(product(out_shape[counter]))
        << node->id() << " 's Output Node " << sink_node->id() << "["
        << utils::Join(out_shape[counter], ",")
        << "]'s size should not zero ! Please check.";

    counter++;
  }
}

void InferShapePass(Graph* graph) {
  VLOG(3) << "Begin InferShapePass";
  auto& shape_dict = graph->GetMutableAttrs<
      absl::flat_hash_map<std::string, framework::shape_t>>("infershape");
  auto& dtype_dict =
      graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>(
          "inferdtype");
  auto store_nodes = std::get<0>(graph->topological_order());

  auto product = [](const framework::shape_t& shape) {
    framework::dim_t numel = 1;
    std::for_each(shape.begin(), shape.end(), [&numel](framework::dim_t dim) {
      numel *= dim;
    });
    return numel;
  };

  for (auto& n : store_nodes) {
    auto node = n->safe_as<Node>();
    if (node) {
      InferShape(node, dtype_dict, shape_dict);
    }
  }
}

}  // namespace pass
}  // namespace hlir
}  // namespace cinn
CINN_REGISTER_HELPER(InferShape) {
  CINN_REGISTER_PASS(InferShape)
      .describe(
          "This pass infer the shape and data type of tensor and save to "
          "g.attrs[\"infershape\"] and "
          "g.attrs[\"inferdtype\"].")
      .set_change_structure(false)
      .provide_graph_attr("infershape")
      .provide_graph_attr("inferdtype")
      .set_body(cinn::hlir::pass::InferShapePass);
  return true;
}
