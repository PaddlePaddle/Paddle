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

#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/hlir/framework/node.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/pass.h"
#include "paddle/cinn/hlir/pass/use_pass.h"
#include "paddle/cinn/hlir/pe/schedule.h"
#include "paddle/cinn/ir/layout.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace hlir {
namespace pass {

using common::GraphNode;
using common::Type;
using framework::Graph;
using framework::Node;
using framework::NodeData;
using framework::Operator;
using framework::OpValueType;

using InferShapeFunc = std::function<std::vector<framework::shape_t>(
    const std::vector<framework::shape_t>&, const framework::AttrMapType&)>;
using InferTypeFunc = std::function<std::vector<Type>(
    const std::vector<Type>&, const framework::AttrMapType&)>;
using InferLayoutFunc = std::function<std::vector<std::vector<std::string>>(
    const std::vector<framework::shape_t>&,
    const std::vector<std::string>&,
    const framework::NodeAttr&,
    const Target&)>;
// insert layout_transform after the input var
std::tuple<Node*, NodeData*> InsertLayoutTransformNodeAfter(
    Graph* graph,
    NodeData* input_data,
    Node* dst_node,
    int pos,
    const std::string& src_layout,
    const std::string& dst_layout,
    const std::string& name) {
  CHECK(graph);
  CHECK(input_data);
  std::string op_type = "layout_transform";
  auto trans_node = new Node(Operator::Get(op_type), op_type, name);
  trans_node->attrs.attr_store["src_layout"] = src_layout;
  trans_node->attrs.attr_store["dst_layout"] = dst_layout;
  auto output_data =
      InsertGraphOpNodeAfter(graph, trans_node, input_data, dst_node, pos);
  trans_node->attrs.attr_store["input_layouts"] = {src_layout};
  trans_node->attrs.attr_store["out_layouts"] = {dst_layout};
  return std::make_tuple(trans_node, output_data);
}

// insert layout_transform before the output var
std::tuple<Node*, NodeData*> InsertLayoutTransformNodeBefore(
    Graph* graph,
    Node* input_node,
    NodeData* dst_data,
    int pos,
    const std::string& src_layout,
    const std::string& dst_layout,
    const std::string& name) {
  CHECK(graph);
  CHECK(input_node);
  CHECK(dst_data);
  std::string op_type = "layout_transform";
  auto trans_node = new Node(Operator::Get(op_type), op_type, name);
  trans_node->attrs.attr_store["src_layout"] = src_layout;
  trans_node->attrs.attr_store["dst_layout"] = dst_layout;
  auto temp_outdata =
      InsertGraphOpNodeBefore(graph, trans_node, input_node, dst_data, pos);
  trans_node->attrs.attr_store["input_layouts"] = {src_layout};
  trans_node->attrs.attr_store["out_layouts"] = {dst_layout};
  return std::make_tuple(trans_node, temp_outdata);
}

std::vector<framework::shape_t> UpdateInferInfos(
    Node* node,
    const std::vector<framework::shape_t>& input_shapes,
    const std::vector<Type>& input_types,
    const std::vector<std::string>& input_layouts,
    const common::Target& target,
    const OpValueType<InferShapeFunc>& op_infershape,
    const OpValueType<InferTypeFunc>& op_infertype,
    const OpValueType<InferLayoutFunc>& op_inferlayout,
    absl::flat_hash_map<std::string, framework::shape_t>* shape_dict,
    absl::flat_hash_map<std::string, Type>* type_dict,
    absl::flat_hash_map<std::string, std::string>* layout_dict) {
  CHECK(shape_dict);
  CHECK(type_dict);
  CHECK(layout_dict);
  CHECK(op_infershape[node->op()])
      << "find no InferShape function for op " << node->op()->name;
  CHECK(op_infertype[node->op()])
      << "find no InferDtype function for op " << node->op()->name;
  CHECK(op_inferlayout[node->op()])
      << "find no InferLayout function for op " << node->op()->name;
  auto infershapes =
      op_infershape[node->op()](input_shapes, node->attrs.attr_store);
  auto infertypes =
      op_infertype[node->op()](input_types, node->attrs.attr_store);
  auto inferlayouts = op_inferlayout[node->op()](
      input_shapes, input_layouts, node->attrs, target);

  CHECK(!infershapes.empty()) << node->op()->name << " finds no infershape";
  CHECK(!infertypes.empty()) << node->op()->name << " finds no infertype";
  CHECK(!inferlayouts.empty()) << node->op()->name << " finds no inferlayout";
  auto outlinks = node->outlinks_in_order();
  CHECK_EQ(infershapes.size(), infertypes.size());
  CHECK_EQ(inferlayouts.size(), 2U);
  CHECK_EQ(infertypes.size(), inferlayouts[0].size());
  CHECK_EQ(outlinks.size(), infershapes.size());

  for (int i = 0; i < outlinks.size(); i++) {
    auto* sink = outlinks[i]->sink();
    (*shape_dict)[sink->id()] = infershapes[i];
    (*type_dict)[sink->id()] = infertypes[i];
    (*layout_dict)[sink->id()] = inferlayouts[0][i];
    VLOG(3) << "Infershape: " << node->op()->name << "'s " << i
            << "-th outlink " << sink->id() << ": "
            << utils::Join(infershapes[i], ", ");
  }
  node->attrs.attr_store["out_layouts"] = inferlayouts[0];
  node->attrs.attr_store["input_layouts"] = inferlayouts[1];
  return infershapes;
}

void AlterLayoutPass(Graph* graph) {
  // alterlayout only in X86 for it's specific layout requirements
  if (graph->target_.arch == Target::Arch::X86) {
    auto store_nodes = std::get<0>(graph->topological_order());
    auto& shape_dict = graph->GetMutableAttrs<
        absl::flat_hash_map<std::string, framework::shape_t>>("infershape");
    auto& type_dict =
        graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>(
            "inferdtype");
    auto& op_infershape = Operator::GetAttrs<InferShapeFunc>("infershape");
    auto& op_inferdtype = Operator::GetAttrs<InferTypeFunc>("inferdtype");
    auto& op_inferlayout = Operator::GetAttrs<InferLayoutFunc>("inferlayout");
    absl::flat_hash_map<std::string, std::string> layout_dict;
    std::string model_name = "";
    if (graph->HasAttr("model_name")) {
      model_name = graph->GetMutableAttrs<std::string>("model_name");
      VLOG(3) << "model_name: " << model_name;
    }
    // collect all convs' original input config before altering layout for
    // loading tune params afterwards
    int index = 0;
    for (int i = 0; i < store_nodes.size(); i++) {
      auto node = store_nodes[i]->safe_as<Node>();
      if (node && node->op()->name == "conv2d") {
        std::vector<int> padding({0, 0});
        std::vector<int> stride({1, 1});
        std::vector<int> dilation({1, 1});
        if (node->attrs.attr_store.find("padding") !=
            node->attrs.attr_store.end()) {
          padding =
              absl::get<std::vector<int>>(node->attrs.attr_store.at("padding"));
        }
        if (node->attrs.attr_store.find("stride") !=
            node->attrs.attr_store.end()) {
          stride =
              absl::get<std::vector<int>>(node->attrs.attr_store.at("stride"));
        }
        if (node->attrs.attr_store.find("dilation") !=
            node->attrs.attr_store.end()) {
          dilation = absl::get<std::vector<int>>(
              node->attrs.attr_store.at("dilation"));
        }
        const auto& conv_inlinks = node->inlinks_in_order();
        CHECK_EQ(conv_inlinks.size(), 2U) << "conv2d should have 2 inputs";
        std::vector<std::vector<int>> inputs_shape;
        for (auto& link : conv_inlinks) {
          auto* source = link->source();
          CHECK(shape_dict.count(source->id()))
              << source->id() << " finds no infershape";
          inputs_shape.push_back(shape_dict.at(source->id()));
        }
        std::string key = pe::GenerateX86ConvKey(inputs_shape[0],
                                                 inputs_shape[1],
                                                 stride,
                                                 padding,
                                                 dilation,
                                                 index++,
                                                 model_name);
        VLOG(3) << "key: " << key;
        node->attrs.attr_store["key"] = key;
      }
    }

    bool has_altered = false;
    for (int i = 0; i < store_nodes.size(); i++) {
      auto node = store_nodes[i]->safe_as<Node>();
      if (node) {
        if (node->op()->name == "conv2d") {
          CHECK(node->attrs.attr_store.count("data_format"))
              << node->op()->name << " op has no data_format attr";
          std::string data_format =
              absl::get<std::string>(node->attrs.attr_store.at("data_format"));
          if (data_format != "NCHW") {
            // not NCHW such as NHWC or has already been altered layout
            continue;
          }
          has_altered = true;
          std::string new_op_type = node->op()->name + "_NCHWc";
          // alter conv2d op to conv2d_NCHWc
          Node* new_node = new Node(Operator::Get(new_op_type),
                                    new_op_type,
                                    common::UniqName(new_op_type));
          new_node->attrs.attr_store = node->attrs.attr_store;
          std::string new_data_format = "NCHWc";
          new_node->attrs.attr_store["data_format"] = new_data_format;

          const auto& conv_inlinks = node->inlinks_in_order();
          std::vector<common::GraphNode*> input_nodes;
          for (auto& link : conv_inlinks) {
            auto* source = link->source();
            input_nodes.push_back(source);
          }
          // get new layout: ic_bn, oc_bn
          CHECK_EQ(input_nodes.size(), 2U)
              << "conv2d should have 2 input nodes";
          auto* input_node = input_nodes[0];
          auto* weight_node = input_nodes[1];
          CHECK(shape_dict.count(input_node->id()))
              << input_node->id() << " has no infershape";
          CHECK(shape_dict.count(weight_node->id()))
              << weight_node->id() << " has no infershape";
          CHECK(type_dict.count(input_node->id()))
              << input_node->id() << " has no infertype";
          CHECK(type_dict.count(weight_node->id()))
              << weight_node->id() << " has no infertype";
          auto input_shape = shape_dict.at(input_node->id());
          auto weight_shape = shape_dict.at(weight_node->id());
          auto input_type = type_dict.at(input_node->id());
          auto weight_type = type_dict.at(weight_node->id());
          Node* weight_trans_node;
          Node* input_trans_node;
          std::vector<framework::shape_t> conv2d_NCHWc_inputshapes;
          std::vector<Type> conv2d_NCHWc_inputtypes;
          std::vector<std::string> conv2d_NCHWc_inputlayouts;
          CHECK(weight_shape.size() == 4)
              << "old conv2d's weight shape should be 4";
          absl::flat_hash_map<std::string, int> conv2d_factors;
          int oc, fc, ic = 1;
          if (input_shape.size() == 4) {
            ic = input_shape[1];
          } else if (input_shape.size() == 5) {
            ic = input_shape[1] * input_shape[4];
          } else {
            LOG(FATAL)
                << "conv2d's input shape should be 4D/5D. Wrong input shape: "
                << utils::Join(input_shape, ", ");
          }

          if (weight_shape.size() == 4) {
            oc = weight_shape[0];
            fc = weight_shape[1];
          } else if (weight_shape.size() == 6) {
            oc = weight_shape[0] * weight_shape[5];
            fc = weight_shape[1] * weight_shape[4];
          } else {
            LOG(FATAL)
                << "conv2d's weight shape should be 4D/6D. Wrong weight shape: "
                << utils::Join(weight_shape, ", ");
          }
          VLOG(3) << "oc: " << oc;
          VLOG(3) << "ic: " << ic;
          VLOG(3) << "fc: " << fc;

          // get the original conv config stored in the key attr
          CHECK(new_node->attrs.attr_store.count("key"))
              << "conv2d finds no key attr";
          std::string key =
              absl::get<std::string>(new_node->attrs.attr_store.at("key"));
          VLOG(3) << "key: " << key;
          pe::GetConv2dFactors(&conv2d_factors,
                               oc,
                               ic,
                               fc,
                               -1,
                               -1,
                               input_type,
                               graph->target_,
                               key);
          CHECK(conv2d_factors.count("oc_bn"));
          CHECK(conv2d_factors.count("ic_bn"));
          CHECK(conv2d_factors.count("fc_bn"));
          int oc_bn = conv2d_factors["oc_bn"];
          int ic_bn = conv2d_factors["ic_bn"];
          int fc_bn = conv2d_factors["fc_bn"];
          VLOG(3) << "oc_bn: " << oc_bn;
          VLOG(3) << "ic_bn: " << ic_bn;
          VLOG(3) << "fc_bn: " << fc_bn;

          if (input_shape.size() == 4) {
            std::string src_input_layout = "NCHW";
            std::string dst_input_layout = "NCHW" + std::to_string(ic_bn) + "c";
            VLOG(3) << "dst_input_layout: " << dst_input_layout;
            // insert input layout_transform
            auto input_data = input_node->safe_as<NodeData>();
            CHECK(input_data);
            NodeData* output_data;
            std::tie(input_trans_node, output_data) =
                InsertLayoutTransformNodeAfter(
                    graph,
                    input_data,
                    node,
                    0,
                    src_input_layout,
                    dst_input_layout,
                    common::UniqName(node->op()->name +
                                     "_input_layout_tranform"));
            UpdateInferInfos(input_trans_node,
                             {input_shape},
                             {input_type},
                             {src_input_layout},
                             graph->target_,
                             op_infershape,
                             op_inferdtype,
                             op_inferlayout,
                             &shape_dict,
                             &type_dict,
                             &layout_dict);
            CHECK(shape_dict.count(output_data->id()))
                << output_data->id() << " finds no infershape in shape_dict.";
            CHECK(type_dict.count(output_data->id()))
                << output_data->id() << " finds no infertype in shape_dict.";
            auto trans_out_shapes = shape_dict[output_data->id()];
            auto trans_out_dtypes = type_dict[output_data->id()];
            conv2d_NCHWc_inputshapes.push_back(trans_out_shapes);
            conv2d_NCHWc_inputtypes.push_back(trans_out_dtypes);
            conv2d_NCHWc_inputlayouts.push_back(dst_input_layout);
          } else {
            CHECK_EQ(input_shape.size(), 5U)
                << "conv2d_NCHWc op's input shape dim should be 5";
            conv2d_NCHWc_inputshapes.push_back(input_shape);
            conv2d_NCHWc_inputtypes.push_back(input_type);
            CHECK(layout_dict.count(input_node->id()))
                << input_node->id() << " should have out_layout attr";
            conv2d_NCHWc_inputlayouts.push_back(layout_dict[input_node->id()]);
          }
          if (weight_shape.size() == 4) {
            std::string src_kernel_layout = "OIHW";
            std::string dst_kernel_layout = "OIHW" + std::to_string(fc_bn) +
                                            "i" + std::to_string(oc_bn) + "o";
            VLOG(3) << "dst_kernel_layout: " << dst_kernel_layout;
            // insert weight layout_transform
            auto weight_data = weight_node->safe_as<NodeData>();
            CHECK(weight_data);
            NodeData* output_data;
            std::tie(weight_trans_node, output_data) =
                InsertLayoutTransformNodeAfter(
                    graph,
                    weight_data,
                    node,
                    1,
                    src_kernel_layout,
                    dst_kernel_layout,
                    common::UniqName(node->op()->name +
                                     "_weight_layout_tranform"));
            UpdateInferInfos(weight_trans_node,
                             {weight_shape},
                             {weight_type},
                             {src_kernel_layout},
                             graph->target_,
                             op_infershape,
                             op_inferdtype,
                             op_inferlayout,
                             &shape_dict,
                             &type_dict,
                             &layout_dict);
            CHECK(shape_dict.count(output_data->id()))
                << output_data->id() << " finds no infershape in shape_dict.";
            CHECK(type_dict.count(output_data->id()))
                << output_data->id() << " finds no infertype in shape_dict.";
            auto trans_out_shapes = shape_dict[output_data->id()];
            auto trans_out_dtypes = type_dict[output_data->id()];
            conv2d_NCHWc_inputshapes.push_back(trans_out_shapes);
            conv2d_NCHWc_inputtypes.push_back(trans_out_dtypes);
            conv2d_NCHWc_inputlayouts.push_back(dst_kernel_layout);
          } else {
            CHECK_EQ(weight_shape.size(), 6U)
                << weight_node->id() << " shape dim should be 6";
            conv2d_NCHWc_inputshapes.push_back(weight_shape);
            conv2d_NCHWc_inputtypes.push_back(weight_type);
            CHECK(layout_dict.count(weight_node->id()))
                << weight_node->id() << " should have out_layout attr";
            conv2d_NCHWc_inputlayouts.push_back(layout_dict[weight_node->id()]);
          }
          // replace conv2d to conv2d_NCHWc
          auto infershapes = op_infershape[new_node->op()](
              conv2d_NCHWc_inputshapes, new_node->attrs.attr_store);
          const auto& old_inlinks = node->inlinks_in_order();
          const auto& old_outlinks = node->outlinks_in_order();
          for (auto& link : old_inlinks) {
            auto source = link->source();
            source->UnLinkSingleTo(node);
            source->LinkTo(new_node);
          }
          std::vector<Node*> next_ops;
          int count = 0;
          Shared<Node> node_ptr(new_node);
          for (auto& link : old_outlinks) {
            auto sink = link->sink();
            node->UnLinkSingleTo(sink);
            if (!count) {
              // keep the first out var and its outlinks
              auto out_var = sink->safe_as<NodeData>();
              CHECK(out_var);
              out_var->source_node = node_ptr;
              new_node->LinkTo(out_var);
            }
            count++;
          }
          for (int i = 1; i < infershapes.size(); i++) {
            auto* new_out = new NodeData(
                node_ptr,
                i,
                0,
                common::UniqName(new_node->id() + "_out_" + std::to_string(i)));
            graph->RegisterNode(new_out->id(), new_out);
            new_node->as<common::GraphNode>()->LinkTo(new_out);
          }
          graph->RegisterNode(new_node->id(), new_node);
          // update conv2d_NCHWc's infershape, infertype, inferlayout and set
          // attrs
          UpdateInferInfos(new_node,
                           conv2d_NCHWc_inputshapes,
                           conv2d_NCHWc_inputtypes,
                           conv2d_NCHWc_inputlayouts,
                           graph->target_,
                           op_infershape,
                           op_inferdtype,
                           op_inferlayout,
                           &shape_dict,
                           &type_dict,
                           &layout_dict);
        } else if (has_altered) {
          // not alterlayout like conv2d, just inferlayout
          std::vector<framework::shape_t> input_shapes;
          std::vector<Type> input_types;
          std::vector<std::string> input_layouts;
          for (auto& link : node->inlinks_in_order()) {
            auto* source = link->source();
            CHECK(shape_dict.count(source->id()))
                << source->id() << " finds no infershape";
            CHECK(type_dict.count(source->id()))
                << source->id() << " finds no infertype";
            input_shapes.push_back(shape_dict[source->id()]);
            input_types.push_back(type_dict[source->id()]);
            if (layout_dict.count(source->id())) {
              input_layouts.push_back(layout_dict[source->id()]);
            } else {
              input_layouts.push_back("");
            }
          }
          CHECK(op_inferlayout[node->op()])
              << "find no InferLayout function for op " << node->op()->name;
          auto inferlayouts = op_inferlayout[node->op()](
              input_shapes, input_layouts, node->attrs, graph->target_);
          // if input inferred layouts is different from original's, expand dims
          // or do transformation.
          CHECK_EQ(inferlayouts.size(), 2U);
          auto new_input_layouts = inferlayouts[1];
          auto inlinks = node->inlinks_in_order();
          CHECK_EQ(input_layouts.size(), inlinks.size());
          CHECK_EQ(input_layouts.size(), new_input_layouts.size());
          CHECK_EQ(input_layouts.size(), input_shapes.size());
          bool reset_axis = false;
          for (int i = 0; i < inlinks.size(); i++) {
            if (input_layouts[i] != new_input_layouts[i]) {
              // expand dims or do transformation
              int input_shape_size = input_shapes[i].size();
              if (input_shape_size == 1 && new_input_layouts[i].size() > 4) {
                // C -> NCHWxc: 1. C -> NCHW 2. layout transform from NCHW to
                // NCHWxc
                int axis = -1;
                CHECK(node->attrs.attr_store.count("axis"))
                    << node->id() << " find no axis attr";
                axis = absl::get<int>(node->attrs.attr_store["axis"]);
                CHECK(new_input_layouts[i].substr(0, 4) == "NCHW")
                    << "only support NCHWxc";
                if (axis == -1) {
                  axis += 4;
                }
                std::vector<int> new_shapes;
                for (int j = 0; j < 4; j++) {
                  if (axis == j) {
                    new_shapes.push_back(input_shapes[i][0]);
                  } else {
                    new_shapes.push_back(1);
                  }
                }
                // C -> NCHW, insert layout tranfrom
                auto source = inlinks[i]->source();
                std::string src_layout = "C";
                layout_dict[source->id()] = src_layout;
                auto input_data = source->safe_as<NodeData>();
                CHECK(input_data);
                VLOG(3) << source->id() << " do layout_tranform from C to NCHW";
                std::string op_type = "broadcast_to";
                auto trans_node =
                    new Node(Operator::Get(op_type),
                             op_type,
                             common::UniqName(source->id() + "_broadcastto"));
                trans_node->attrs.attr_store["out_shape"] = new_shapes;
                std::vector<int> broadcast_axes = {1};
                trans_node->attrs.attr_store["broadcast_axes"] = broadcast_axes;
                auto output_data = InsertGraphOpNodeAfter(
                    graph, trans_node, input_data, node, i);
                UpdateInferInfos(trans_node,
                                 {input_shapes[i]},
                                 {input_types[i]},
                                 {src_layout},
                                 graph->target_,
                                 op_infershape,
                                 op_inferdtype,
                                 op_inferlayout,
                                 &shape_dict,
                                 &type_dict,
                                 &layout_dict);

                std::string new_src_layout = "NCHW";
                reset_axis = true;
                // insert layout tranfrom
                auto new_input_data = output_data->safe_as<NodeData>();
                CHECK(new_input_data);
                NodeData* new_output_data;
                Node* new_trans_node;
                VLOG(3) << new_input_data->id()
                        << " do layout_tranform from NCHW to NCHWxc";
                std::tie(new_trans_node, new_output_data) =
                    InsertLayoutTransformNodeAfter(
                        graph,
                        new_input_data,
                        node,
                        i,
                        new_src_layout,
                        new_input_layouts[i],
                        common::UniqName(new_input_data->id() +
                                         "_layout_tranform"));
                UpdateInferInfos(new_trans_node,
                                 {shape_dict[new_input_data->id()]},
                                 {input_types[i]},
                                 {new_src_layout},
                                 graph->target_,
                                 op_infershape,
                                 op_inferdtype,
                                 op_inferlayout,
                                 &shape_dict,
                                 &type_dict,
                                 &layout_dict);
              } else if (input_shape_size == 4 &&
                         new_input_layouts[i].size() > 4) {
                // NCHW -> NCHWxc
                // insert layout tranfrom
                auto source = inlinks[i]->source();
                auto src_layout = "NCHW";
                layout_dict[source->id()] = src_layout;
                auto input_data = source->safe_as<NodeData>();
                CHECK(input_data);
                NodeData* output_data;
                Node* trans_node;
                VLOG(3) << source->id()
                        << " do layout_tranform from NCHW to NCHWxc";
                std::tie(trans_node, output_data) =
                    InsertLayoutTransformNodeAfter(
                        graph,
                        input_data,
                        node,
                        i,
                        src_layout,
                        new_input_layouts[i],
                        common::UniqName(source->id() + "_layout_tranform"));
                UpdateInferInfos(trans_node,
                                 {input_shapes[i]},
                                 {input_types[i]},
                                 {src_layout},
                                 graph->target_,
                                 op_infershape,
                                 op_inferdtype,
                                 op_inferlayout,
                                 &shape_dict,
                                 &type_dict,
                                 &layout_dict);
              } else if (input_shape_size == 5 &&
                         new_input_layouts[i].size() == 4) {
                // NCHWxc -> NCHW
                // insert layout tranfrom
                auto source = inlinks[i]->source();
                auto src_layout = input_layouts[i];
                layout_dict[source->id()] = src_layout;
                auto input_data = source->safe_as<NodeData>();
                CHECK(input_data);
                NodeData* output_data;
                Node* trans_node;
                VLOG(3) << source->id()
                        << " do layout_tranform from NCHWxc to NCHW";
                std::tie(trans_node, output_data) =
                    InsertLayoutTransformNodeAfter(
                        graph,
                        input_data,
                        node,
                        i,
                        src_layout,
                        new_input_layouts[i],
                        common::UniqName(source->id() + "_layout_tranform"));
                UpdateInferInfos(trans_node,
                                 {input_shapes[i]},
                                 {input_types[i]},
                                 {src_layout},
                                 graph->target_,
                                 op_infershape,
                                 op_inferdtype,
                                 op_inferlayout,
                                 &shape_dict,
                                 &type_dict,
                                 &layout_dict);
              }
            }
          }
          if (reset_axis) {
            node->attrs.attr_store["axis"] = -1;
          }
          input_shapes.clear();
          input_types.clear();
          input_layouts.clear();
          for (auto& link : node->inlinks_in_order()) {
            auto* source = link->source();
            CHECK(shape_dict.count(source->id()))
                << source->id() << " finds no infershape";
            CHECK(type_dict.count(source->id()))
                << source->id() << " finds no infertype";
            input_shapes.push_back(shape_dict[source->id()]);
            input_types.push_back(type_dict[source->id()]);
            if (layout_dict.count(source->id())) {
              input_layouts.push_back(layout_dict[source->id()]);
            } else {
              input_layouts.push_back("");
            }
          }
          UpdateInferInfos(node,
                           input_shapes,
                           input_types,
                           input_layouts,
                           graph->target_,
                           op_infershape,
                           op_inferdtype,
                           op_inferlayout,
                           &shape_dict,
                           &type_dict,
                           &layout_dict);
        }
      }
    }
    if (has_altered) {
      // final layout transform
      store_nodes = std::get<0>(graph->topological_order());
      for (int i = store_nodes.size() - 1; i >= 0; i--) {
        auto* node = store_nodes[i]->safe_as<Node>();
        if (node) {
          CHECK(node->attrs.attr_store.count("out_layouts"))
              << node->id() << " finds no out_layouts attr";
          auto out_layouts = absl::get<std::vector<std::string>>(
              node->attrs.attr_store.at("out_layouts"));
          CHECK(!out_layouts.empty());
          if (out_layouts[0].size() > 4) {
            // recover the layout finally, NCHWxc->NCHW, only first output
            auto outlinks = node->outlinks_in_order();
            CHECK(!outlinks.empty());
            auto* out_node = outlinks[0]->sink();
            std::string dst_layout = "NCHW";
            CHECK(layout_dict.count(out_node->id()))
                << out_node->id() << " finds no out_layout";
            std::string src_layout = layout_dict[out_node->id()];
            // insert layout_transform
            NodeData* temp_out;
            Node* trans_node;
            CHECK(shape_dict.count(out_node->id()))
                << out_node->id() << " finds no infershape";
            CHECK(type_dict.count(out_node->id()))
                << out_node->id() << " finds no infertype";
            auto shape = shape_dict[out_node->id()];
            auto type = type_dict[out_node->id()];
            // insert layout transform before the output var to keep the final
            // original output var
            std::tie(trans_node, temp_out) = InsertLayoutTransformNodeBefore(
                graph,
                node,
                out_node->safe_as<NodeData>(),
                0,
                src_layout,
                dst_layout,
                common::UniqName(node->op()->name + "_final_layout_tranform"));
            shape_dict[temp_out->id()] = shape;
            type_dict[temp_out->id()] = type;
            layout_dict[temp_out->id()] = src_layout;
            UpdateInferInfos(trans_node,
                             {shape},
                             {type},
                             {src_layout},
                             graph->target_,
                             op_infershape,
                             op_inferdtype,
                             op_inferlayout,
                             &shape_dict,
                             &type_dict,
                             &layout_dict);
          }
          break;
        }
      }
      graph->ClearUnlinkedNodes(&shape_dict, &type_dict, &layout_dict);
      graph->attrs["infershape"] = std::make_shared<absl::any>(shape_dict);
      graph->attrs["inferdtype"] = std::make_shared<absl::any>(type_dict);
      graph->attrs["inferlayout"] = std::make_shared<absl::any>(layout_dict);
    }
  }
}

}  // namespace pass
}  // namespace hlir
}  // namespace cinn
CINN_REGISTER_HELPER(AlterLayout) {
  CINN_REGISTER_PASS(AlterLayout)
      .describe(
          "This pass alters ops' data layouts in the graph(e.g. NCHW -> "
          "NCHWxc, OIHW -> OIHWxoxi) and saves to "
          "g.attrs[\"inferlayout\"]")
      .set_change_structure(true)
      .provide_graph_attr("infershape")
      .provide_graph_attr("inferdtype")
      .set_body(cinn::hlir::pass::AlterLayoutPass);
  return true;
}
