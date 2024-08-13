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

#include "paddle/fluid/platform/device/ipu/popart_canonicalization/canonicalization_utils.h"
#include "paddle/fluid/platform/device/ipu/popart_canonicalization/op_builder.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {
namespace ipu {
namespace {

Node *fill_constant_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto op_inputs = op->Inputs();
  if (op_inputs.find("ShapeTensor") != op_inputs.end() &&
      !op->Input("ShapeTensor").empty()) {
    PADDLE_THROW(
        common::errors::Unimplemented("op fill_constant with ShapeTensor"));
  }
  auto dtype_ = PADDLE_GET_CONST(int, op->GetAttr("dtype"));
  auto dtype = VarType2OnnxDType(static_cast<VarType::Type>(dtype_));
  auto dims = PADDLE_GET_CONST(std::vector<int64_t>, op->GetAttr("shape"));
  auto value_ = PADDLE_GET_CONST(float, op->GetAttr("value"));
  int size = 1;
  for (auto &dim : dims) {
    size *= dim;
  }
  PADDLE_ENFORCE_GT(size,
                    0,
                    errors::InvalidArgument(
                        "IPU doesn't support non-positive dimensions. Please "
                        "check tensor shape setting."));
  Attribute value;
  switch (dtype_) {
    case VarType::FP16:
    case VarType::FP32:
      value = std::vector<float>(size, value_);
      break;
    case VarType::FP64:
      value = std::vector<double>(size, value_);
      break;
    case VarType::INT32:
      value = std::vector<int>(size, value_);
      break;
    case VarType::INT64:
      value = std::vector<int64_t>(size, value_);
      break;
    case VarType::BOOL:
      value = std::vector<bool>(size, value_);
      break;
    default:
      PADDLE_THROW(
          common::errors::Unimplemented("fill_constant dtype: %d", dtype_));
  }
  return CreateConst(graph,
                     node,
                     node->inputs,
                     node->outputs,
                     AttributeMap{
                         {"value", value},
                         {"dims", dims},
                         {"dtype", dtype},
                     });
}

Node *gaussian_random_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto shape = PADDLE_GET_CONST(std::vector<int64_t>, op->GetAttr("shape"));
  auto dtype_ = PADDLE_GET_CONST(int, op->GetAttr("dtype"));
  auto dtype = VarType2OnnxDType(static_cast<VarType::Type>(dtype_));
  auto mean = PADDLE_GET_CONST(float, op->GetAttr("mean"));
  auto scale = PADDLE_GET_CONST(float, op->GetAttr("std"));
  // seed not work
  auto seed_ = PADDLE_GET_CONST(int, op->GetAttr("seed"));
  auto seed = static_cast<float>(seed_);
  return CreateBaseOp(graph,
                      node,
                      "popart_randomnormal",
                      node->inputs,
                      node->outputs,
                      {
                          {"shape", shape},
                          {"dtype", dtype},
                          {"mean", mean},
                          {"scale", scale},
                          {"seed", seed},
                      });
}

Node *uniform_random_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto shape = PADDLE_GET_CONST(std::vector<int64_t>, op->GetAttr("shape"));
  auto dtype_ = PADDLE_GET_CONST(int, op->GetAttr("dtype"));
  auto dtype = VarType2OnnxDType(static_cast<VarType::Type>(dtype_));
  auto high = PADDLE_GET_CONST(float, op->GetAttr("max"));
  auto low = PADDLE_GET_CONST(float, op->GetAttr("min"));
  // seed not work
  auto seed_ = PADDLE_GET_CONST(int, op->GetAttr("seed"));
  auto seed = static_cast<float>(seed_);
  return CreateBaseOp(graph,
                      node,
                      "popart_randomuniform",
                      node->inputs,
                      node->outputs,
                      {
                          {"shape", shape},
                          {"dtype", dtype},
                          {"high", high},
                          {"low", low},
                          {"seed", seed},
                      });
}

Node *transpose_handler(Graph *graph, Node *node) {
  auto *op = node->Op();

  auto axis_ = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("axis"));
  std::vector<int64_t> perm(axis_.begin(), axis_.end());
  auto attrs = AttributeMap{{"perm", perm}};

  auto new_node_transpose = CreateBaseOp(graph,
                                         node,
                                         "popart_transpose",
                                         node->inputs,
                                         {GetOutputVarNode("Out", node)},
                                         attrs);
  return new_node_transpose;
}

Node *reshape_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto shape_ = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("shape"));
  std::vector<int64_t> shape(shape_.begin(), shape_.end());
  auto attrs = AttributeMap{
      {"value", shape},
      {"dims", std::vector<int64_t>{static_cast<int64_t>(shape.size())}},
      {"dtype", ONNXDataType::INT64}};
  auto new_node_const =
      CreateBaseOp(graph, node, "popart_constant", {}, {}, attrs);

  auto new_node_reshape =
      CreateBaseOp(graph,
                   node,
                   "popart_reshape",
                   {GetInputVarNode("X", node), new_node_const->outputs[0]},
                   {GetOutputVarNode("Out", node)},
                   {});
  return new_node_reshape;
}

Node *flatten2_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto axis = PADDLE_GET_CONST(int, op->GetAttr("axis"));
  return CreateBaseOp(graph,
                      node,
                      "popart_flatten",
                      {GetInputVarNode("X", node)},
                      {GetOutputVarNode("Out", node)},
                      {{"axis", int64_t(axis)}});
}

Node *gather_handler(Graph *graph, Node *node) {
  auto new_node_gather =
      CreateBaseOp(graph,
                   node,
                   "popart_gather",
                   {GetInputVarNode("X", node), GetInputVarNode("Index", node)},
                   {GetOutputVarNode("Out", node)},
                   {});
  return new_node_gather;
}

Node *squeeze_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto axes_ = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("axes"));
  auto input_shape_ = GetInputVarNode("X", node)->Var()->GetShape();

  std::vector<int64_t> axes{axes_.begin(), axes_.end()};
  if (axes_.empty()) {
    for (int i = 0; i < input_shape_.size(); i++) {
      if (input_shape_[i] == 1) {
        axes.push_back(i);
      }
    }
  }
  auto new_node_squeeze = CreateBaseOp(graph,
                                       node,
                                       "popart_squeeze",
                                       {GetInputVarNode("X", node)},
                                       {GetOutputVarNode("Out", node)},
                                       {{"axes", axes}});

  return new_node_squeeze;
}

Node *cast_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto otype = PADDLE_GET_CONST(int, op->GetAttr("out_dtype"));
  auto new_node = CreateCast(graph,
                             node,
                             node->inputs,
                             node->outputs,
                             static_cast<VarType::Type>(otype));
  // Cast op created in mixed-precision has no pipeline attrs
  auto &prev_nodes = node->inputs.front()->inputs;
  if (!prev_nodes.empty()) {
    auto *prev_op = prev_nodes.front()->Op();
    if (!new_node->Op()->HasAttr(sIpuIndexAttr) &&
        prev_op->HasAttr(sIpuIndexAttr)) {
      CopyOpAttr(sIpuIndexAttr, prev_op, new_node->Op());
    }
    if (!new_node->Op()->HasAttr(sIpuStageAttr) &&
        prev_op->HasAttr(sIpuStageAttr)) {
      CopyOpAttr(sIpuStageAttr, prev_op, new_node->Op());
    }
  }
  return new_node;
}

Node *lookup_table_op_handler(Graph *graph,
                              Node *node,
                              const std::string &type) {
  auto *op = node->Op();
  auto padding_idx_ = PADDLE_GET_CONST(int64_t, op->GetAttr("padding_idx"));
  auto w_shape_ = GetInputVarNode("W", node)->Var()->GetShape();
  auto table_size_ = w_shape_[0];
  auto emb_size_ = w_shape_[1];

  Node *w_node;
  if (padding_idx_ >= 0 && padding_idx_ < table_size_) {
    std::vector<float> const_value_(emb_size_, 0);
    std::vector<int64_t> const_shape_{1, emb_size_};
    auto concat_const = CreateConst(graph,
                                    node,
                                    {},
                                    {},
                                    {{"value", const_value_},
                                     {"dims", const_shape_},
                                     {"dtype", GetOutputVarDType(node)}});
    if (padding_idx_ == 0) {
      auto right_slice =
          CreateSlice(graph,
                      node,
                      {GetInputVarNode("W", node)},
                      {},
                      std::vector<int>{static_cast<int>(padding_idx_) + 1},
                      std::vector<int>{static_cast<int>(table_size_)},
                      std::vector<int>{0},
                      std::vector<int>{1});
      w_node = CreateBaseOp(graph,
                            node,
                            "popart_concat",
                            {concat_const->outputs[0], right_slice->outputs[0]},
                            {},
                            {{"axis", int64_t(0)}});
    } else if (padding_idx_ == table_size_ - 1) {
      auto left_slice =
          CreateSlice(graph,
                      node,
                      {GetInputVarNode("W", node)},
                      {},
                      std::vector<int>{0},
                      std::vector<int>{static_cast<int>(padding_idx_)},
                      std::vector<int>{0},
                      std::vector<int>{1});
      w_node = CreateBaseOp(graph,
                            node,
                            "popart_concat",
                            {left_slice->outputs[0], concat_const->outputs[0]},
                            {},
                            {{"axis", int64_t{0}}});
    } else {
      auto left_slice =
          CreateSlice(graph,
                      node,
                      {GetInputVarNode("W", node)},
                      {},
                      std::vector<int>{0},
                      std::vector<int>{static_cast<int>(padding_idx_)},
                      std::vector<int>{0},
                      std::vector<int>{1});
      auto right_slice =
          CreateSlice(graph,
                      node,
                      {GetInputVarNode("W", node)},
                      {},
                      std::vector<int>{static_cast<int>(padding_idx_) + 1},
                      std::vector<int>{static_cast<int>(table_size_)},
                      std::vector<int>{0},
                      std::vector<int>{1});
      w_node = CreateBaseOp(graph,
                            node,
                            "popart_concat",
                            {left_slice->outputs[0],
                             concat_const->outputs[0],
                             right_slice->outputs[0]},
                            {},
                            {{"axis", int64_t{0}}});
    }
    w_node = w_node->outputs[0];
  } else {
    w_node = GetInputVarNode("W", node);
  }

  // lookup_table and lookup_table_v2
  auto ids = GetInputVarNode("Ids", node);
  if (type == "v1") {
    ids = CreateBaseOp(graph,
                       node,
                       "popart_squeeze",
                       {GetInputVarNode("Ids", node)},
                       {},
                       {{"axes", std::vector<int64_t>{-1}}});
    ids = ids->outputs[0];
  }

  auto gather = CreateBaseOp(graph,
                             node,
                             "popart_gather",
                             {w_node, ids},
                             {GetOutputVarNode("Out", node)},
                             {});
  return gather;
}

Node *lookup_table_handler(Graph *graph, Node *node) {
  return lookup_table_op_handler(graph, node, "v1");
}

Node *lookup_table_v2_handler(Graph *graph, Node *node) {
  return lookup_table_op_handler(graph, node, "v2");
}

Node *unsqueeze_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto axes_ = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("axes"));
  std::vector<int64_t> axes{axes_.begin(), axes_.end()};
  auto new_node_unsqueeze = CreateBaseOp(graph,
                                         node,
                                         "popart_unsqueeze",
                                         {GetInputVarNode("X", node)},
                                         {GetOutputVarNode("Out", node)},
                                         {{"axes", axes}});

  return new_node_unsqueeze;
}

Node *concat_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  int64_t axis_{PADDLE_GET_CONST(int, op->GetAttr("axis"))};

  auto new_node_concat = CreateBaseOp(graph,
                                      node,
                                      "popart_concat",
                                      node->inputs,
                                      node->outputs,
                                      {{"axis", axis_}});
  return new_node_concat;
}

Node *stack_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  int64_t axis_{PADDLE_GET_CONST(int, op->GetAttr("axis"))};
  std::vector<int64_t> axes_{axis_};

  std::vector<Node *> unsqueeze_outputs_{};
  for (auto input : node->inputs) {
    auto new_unsqueeze_node = CreateBaseOp(
        graph, node, "popart_unsqueeze", {input}, {}, {{"axes", axes_}});
    unsqueeze_outputs_.push_back(new_unsqueeze_node->outputs[0]);
    for (size_t i = 0; i < input->outputs.size(); ++i) {
      if (input->outputs[i] == node) {
        input->outputs[i] = new_unsqueeze_node;
        break;
      }
    }
  }
  auto new_node_concat = CreateBaseOp(graph,
                                      node,
                                      "popart_concat",
                                      unsqueeze_outputs_,
                                      {GetOutputVarNode("Y", node)},
                                      {{"axis", axis_}});
  return new_node_concat;
}

Node *shape_handler(Graph *graph, Node *node) {
  auto new_node =
      CreateBaseOp(graph, node, "popart_shape", node->inputs, node->outputs);
  return new_node;
}

Node *slice_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto inputs = op->Inputs();

  auto axes_value = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("axes"));

  std::vector<std::vector<int>> slice_values(3);
  std::vector<std::string> tensor_names{"Starts", "Ends", "Strides"};
  std::vector<std::string> attr_names{"starts", "ends", "strides"};
  for (int i = 0; i < 3; i++) {
    // Starts and Ends are default keys in inputs, but Strides.
    bool is_tensor =
        (inputs.find(tensor_names[i] + "TensorList") != inputs.end() &&
         !inputs.at(tensor_names[i] + "TensorList").empty()) ||
        (inputs.find(tensor_names[i] + "Tensor") != inputs.end() &&
         !inputs.at(tensor_names[i] + "Tensor").empty());
    if (is_tensor) {
      PADDLE_THROW(common::errors::Unimplemented(
          "Do not support starts, ends and strides as tensors."));
    } else {
      if (i == 2 && !op->HasAttr("strides")) {
        slice_values[i] = std::vector<int>(axes_value.size(), 1);
      } else {
        slice_values[i] =
            PADDLE_GET_CONST(std::vector<int>, op->GetAttr(attr_names[i]));
      }
    }
  }

  auto decrease_axis_ =
      PADDLE_GET_CONST(std::vector<int>, op->GetAttr("decrease_axis"));
  if (decrease_axis_.size() == 0) {
    return CreateSlice(graph,
                       node,
                       {GetInputVarNode("Input", node)},
                       {GetOutputVarNode("Out", node)},
                       slice_values[0],
                       slice_values[1],
                       axes_value,
                       slice_values[2]);
  } else {
    auto *slice = CreateSlice(graph,
                              node,
                              {GetInputVarNode("Input", node)},
                              {},
                              slice_values[0],
                              slice_values[1],
                              axes_value,
                              slice_values[2])
                      ->outputs[0];
    return CreateBaseOp(
        graph,
        node,
        "popart_squeeze",
        {slice},
        {GetOutputVarNode("Out", node)},
        {{"axes",
          std::vector<int64_t>{decrease_axis_.begin(), decrease_axis_.end()}}});
  }
}

Node *strided_slice_handler(Graph *graph, Node *node) {
  return slice_handler(graph, node);
}

Node *expand_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  if (!op->Input("expand_times_tensor").empty()) {
    PADDLE_THROW(
        common::errors::Unimplemented("Expand op with expand_times_tensor"));
  }

  Node *expand_times = nullptr;
  if (!op->Input("ExpandTimes").empty()) {
    // cast to int64
    expand_times = CreateCast(graph,
                              node,
                              {GetInputVarNode("ExpandTimes", node)},
                              {},
                              VarType::INT64);
  } else {
    auto expand_times_i32 =
        PADDLE_GET_CONST(std::vector<int>, op->GetAttr("expand_times"));
    auto expand_times_ =
        std::vector<int64_t>{expand_times_i32.begin(), expand_times_i32.end()};
    auto dim = int64_t(expand_times_.size());
    expand_times = CreateConst(graph,
                               node,
                               std::vector<int64_t>{expand_times_},
                               {dim},
                               ONNXDataType::INT64);
  }
  auto new_node =
      CreateBaseOp(graph,
                   node,
                   "popart_tile",
                   {GetInputVarNode("X", node), expand_times->outputs[0]},
                   node->outputs);
  return new_node;
}

Node *assign_handler(Graph *graph, Node *node) {
  return CreateBaseOp(graph,
                      node,
                      "popart_identity",
                      {GetInputVarNode("X", node)},
                      {GetOutputVarNode("Out", node)},
                      {});
}

Node *share_data_handler(Graph *graph, Node *node) {
  return assign_handler(graph, node);
}

Node *assign_value_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto dtype_ = PADDLE_GET_CONST(int, op->GetAttr("dtype"));
  auto dtype = VarType2OnnxDType(static_cast<VarType::Type>(dtype_));
  auto dims_ = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("shape"));
  std::vector<int64_t> dims(dims_.begin(), dims_.end());
  Attribute values;
  std::string value_name;
  switch (dtype_) {
    case VarType::BOOL: {
      value_name = "bool_values";
      auto vec_int =
          PADDLE_GET_CONST(std::vector<int>, op->GetAttr(value_name));
      std::vector<bool> vec_bool(vec_int.begin(), vec_int.end());
      values = vec_bool;
    } break;
    case VarType::INT32:
      value_name = "int32_values";
      values = PADDLE_GET_CONST(std::vector<int>, op->GetAttr(value_name));
      break;
    case VarType::FP16:
    case VarType::FP32:
      value_name = "fp32_values";
      values = PADDLE_GET_CONST(std::vector<float>, op->GetAttr(value_name));
      break;
    case VarType::INT64:
      value_name = "int64_values";
      values = PADDLE_GET_CONST(std::vector<int64_t>, op->GetAttr(value_name));
      break;
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Unsupported data type(code %d) for AssignValue operator, only "
          "supports bool, int32, float32 and int64.",
          dtype));
  }
  return CreateConst(graph,
                     node,
                     node->inputs,
                     node->outputs,
                     AttributeMap{
                         {"value", values},
                         {"dims", dims},
                         {"dtype", dtype},
                     });
}

Node *fill_any_like_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto value = PADDLE_GET_CONST(float, op->GetAttr("value"));
  auto x_shape = GetInputVarNode("X", node)->Var()->GetShape();
  auto dtype_ = PADDLE_GET_CONST(int, op->GetAttr("dtype"));
  auto dtype = static_cast<VarType::Type>(dtype_);
  int size = 1;
  for (auto &dim : x_shape) {
    size *= dim;
  }
  PADDLE_ENFORCE_GT(size,
                    0,
                    errors::InvalidArgument(
                        "IPU doesn't support non-positive dimensions. Please "
                        "check tensor shape setting."));

  Attribute out_value;
  switch (dtype) {
    case VarType::FP16:
    case VarType::FP32:
      out_value = std::vector<float>(size, value);
      break;
    case VarType::FP64:
      out_value = std::vector<double>(size, value);
      break;
    case VarType::INT32:
      out_value = std::vector<int>(size, value);
      break;
    case VarType::INT64:
      out_value = std::vector<int64_t>(size, value);
      break;
    case VarType::BOOL:
      out_value = std::vector<int64_t>(size, value);
      break;
    default:
      PADDLE_THROW(
          common::errors::Unimplemented("fill_any_like dtype: %d", dtype));
  }
  return CreateConst(graph,
                     node,
                     node->inputs,
                     node->outputs,
                     AttributeMap{
                         {"value", out_value},
                         {"dims", x_shape},
                         {"dtype", VarType2OnnxDType(dtype)},
                     });
}

Node *one_hot_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto depth = PADDLE_GET_CONST(int, op->GetAttr("depth"));
  auto allow_out_of_range =
      PADDLE_GET_CONST(bool, op->GetAttr("allow_out_of_range"));
  if (allow_out_of_range) {
    PADDLE_THROW(common::errors::Unimplemented(
        "Do not support allow_out_of_range=True"));
  } else {
    auto depth_tensor = CreateConst(graph,
                                    node,
                                    {},
                                    {},
                                    {{"value", std::vector<int64_t>{depth}},
                                     {"dims", std::vector<int64_t>{1}},
                                     {"dtype", ONNXDataType::INT64}});
    auto value_tensor = CreateConst(graph,
                                    node,
                                    {},
                                    {},
                                    {{"value", std::vector<float>{0, 1}},
                                     {"dims", std::vector<int64_t>{2}},
                                     {"dtype", ONNXDataType::FLOAT}});
    return CreateBaseOp(graph,
                        node,
                        "popart_onehot",
                        {GetInputVarNode("X", node),
                         depth_tensor->outputs[0],
                         value_tensor->outputs[0]},
                        {GetOutputVarNode("Out", node)},
                        {{"axis", int64_t{-1}}});
  }
}

Node *one_hot_v2_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto depth = PADDLE_GET_CONST(int, op->GetAttr("depth"));
  auto allow_out_of_range =
      PADDLE_GET_CONST(bool, op->GetAttr("allow_out_of_range"));
  if (allow_out_of_range) {
    PADDLE_THROW(common::errors::Unimplemented(
        "Do not support allow_out_of_range=True"));
  } else {
    auto depth_tensor = CreateConst(graph,
                                    node,
                                    {},
                                    {},
                                    {{"value", std::vector<int>{depth}},
                                     {"dims", std::vector<int64_t>{1}},
                                     {"dtype", ONNXDataType::INT32}});
    Node *value_tensor = nullptr;
    if (GetOutputVarNode("Out", node)->Var()->GetDataType() == VarType::FP16) {
      value_tensor = CreateConst(graph,
                                 node,
                                 {},
                                 {},
                                 {{"value", std::vector<float>{0, 1}},
                                  {"dims", std::vector<int64_t>{2}},
                                  {"dtype", ONNXDataType::FLOAT16}});
    } else {
      value_tensor = CreateConst(graph,
                                 node,
                                 {},
                                 {},
                                 {{"value", std::vector<float>{0, 1}},
                                  {"dims", std::vector<int64_t>{2}},
                                  {"dtype", ONNXDataType::FLOAT}});
    }

    return CreateBaseOp(graph,
                        node,
                        "popart_onehot",
                        {GetInputVarNode("X", node),
                         depth_tensor->outputs[0],
                         value_tensor->outputs[0]},
                        {GetOutputVarNode("Out", node)},
                        {{"axis", int64_t{-1}}});
  }
}

Node *split_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto axis = PADDLE_GET_CONST(int, op->GetAttr("axis"));
  auto sections = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("sections"));
  return CreateSplit(graph,
                     node,
                     {GetInputVarNode("X", node)},
                     node->outputs,
                     std::vector<int64_t>{sections.begin(), sections.end()},
                     axis);
}

Node *dot_handler(Graph *graph, Node *node) {
  auto x = GetInputVarNode("X", node);
  auto mul_node =
      CreateBaseOp(
          graph, node, "popart_mul", {x, GetInputVarNode("Y", node)}, {})
          ->outputs.front();
  int64_t axes = x->Var()->GetShape().size() - 1;
  return CreateBaseOp(graph,
                      node,
                      "popart_reducesum",
                      {mul_node},
                      {GetOutputVarNode("Out", node)},
                      {
                          {"axes", std::vector<int64_t>{axes}},
                      });
}

Node *clip_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  // if (min_value == -FLT_MAX) then means no min_value
  // if (max_value == FLT_MAX) then means no max_value
  auto min_value = PADDLE_GET_CONST(float, op->GetAttr("min"));
  auto max_value = PADDLE_GET_CONST(float, op->GetAttr("max"));

  bool has_min_tensor = false;
  bool has_max_tensor = false;
  if (node->Op()->Input("Min").size()) {
    has_min_tensor = true;
  }
  if (node->Op()->Input("Max").size()) {
    has_max_tensor = true;
  }

  bool transfer_input_dtype = false;
  Node *input_data = GetInputVarNode("X", node);
  if (input_data->Var()->GetDataType() != VarType::FP32 &&
      input_data->Var()->GetDataType() != VarType::FP16) {
    input_data =
        CreateCast(graph, node, {input_data}, {}, VarType::FP32)->outputs[0];
    transfer_input_dtype = true;
  }

  Node *min_tensor = nullptr;
  if (has_min_tensor) {
    if (GetInputVarNode("Min", node)->Var()->GetDataType() != VarType::FP32) {
      min_tensor =
          CreateCast(
              graph, node, {GetInputVarNode("Min", node)}, {}, VarType::FP32)
              ->outputs[0];
    } else {
      min_tensor = GetInputVarNode("Min", node);
    }
  } else {
    min_tensor = CreateConst(graph,
                             node,
                             {},
                             {},
                             {{"value", std::vector<float>{min_value}},
                              {"dims", std::vector<int64_t>{1}},
                              {"dtype", ONNXDataType::FLOAT}})
                     ->outputs[0];
  }

  Node *max_tensor = nullptr;
  if (has_max_tensor) {
    if (GetInputVarNode("Max", node)->Var()->GetDataType() != VarType::FP32) {
      max_tensor =
          CreateCast(
              graph, node, {GetInputVarNode("Max", node)}, {}, VarType::FP32)
              ->outputs[0];
    } else {
      max_tensor = GetInputVarNode("Max", node);
    }
  } else {
    max_tensor = CreateConst(graph,
                             node,
                             {},
                             {},
                             {{"value", std::vector<float>{max_value}},
                              {"dims", std::vector<int64_t>{1}},
                              {"dtype", ONNXDataType::FLOAT}})
                     ->outputs[0];
  }

  if (transfer_input_dtype) {
    auto clip_res = CreateBaseOp(
        graph, node, "popart_clip", {input_data, min_tensor, max_tensor}, {});
    return CreateCast(graph,
                      node,
                      clip_res->outputs,
                      {GetOutputVarNode("Out", node)},
                      GetInputVarNode("X", node)->Var()->GetDataType());
  } else {
    return CreateBaseOp(graph,
                        node,
                        "popart_clip",
                        {input_data, min_tensor, max_tensor},
                        {GetOutputVarNode("Out", node)});
  }
}

Node *dist_handler(Graph *graph, Node *node) {
  // Minimum negative float
  union neg_infinity {
    int neg_int_inf;
    float neg_float_int;
  };
  neg_infinity neg_inf;
  neg_inf.neg_int_inf = 0xFF800000;
  float g_NegFloatInfinity = neg_inf.neg_float_int;

  auto *op = node->Op();
  auto *sub_node =
      CreateBaseOp(graph,
                   node,
                   "popart_sub",
                   {GetInputVarNode("X", node), GetInputVarNode("Y", node)},
                   {})
          ->outputs[0];
  auto *abs_node =
      CreateBaseOp(graph, node, "popart_abs", {sub_node}, {})->outputs[0];

  auto p = PADDLE_GET_CONST(float, op->GetAttr("p"));

  // Reshape to 1-D output
  auto target_shape = AttributeMap{{"value", std::vector<int64_t>{-1}},
                                   {"dims", std::vector<int64_t>{1}},
                                   {"dtype", ONNXDataType::INT64}};
  auto *target_shape_node =
      CreateBaseOp(graph, node, "popart_constant", {}, {}, target_shape)
          ->outputs[0];

  if (fabs(p) < 1e-6) {
    auto *sign_node =
        CreateBaseOp(graph, node, "popart_sign", {abs_node}, {})->outputs[0];
    auto *sum_node = CreateBaseOp(graph,
                                  node,
                                  "popart_reducesum",
                                  {sign_node},
                                  {},
                                  {{"keepdims", int64_t{0}}})
                         ->outputs[0];
    return CreateBaseOp(graph,
                        node,
                        "popart_reshape",
                        {sum_node, target_shape_node},
                        {GetOutputVarNode("Out", node)});
  } else if (p == std::numeric_limits<float>::infinity()) {
    auto *max_node = CreateBaseOp(graph,
                                  node,
                                  "popart_reducemax",
                                  {abs_node},
                                  {},
                                  {{"keepdims", int64_t{0}}})
                         ->outputs[0];
    return CreateBaseOp(graph,
                        node,
                        "popart_reshape",
                        {max_node, target_shape_node},
                        {GetOutputVarNode("Out", node)});
  } else if (p == g_NegFloatInfinity) {
    auto *min_node = CreateBaseOp(graph,
                                  node,
                                  "popart_reducemin",
                                  {abs_node},
                                  {},
                                  {{"keepdims", int64_t{0}}})
                         ->outputs[0];
    return CreateBaseOp(graph,
                        node,
                        "popart_reshape",
                        {min_node, target_shape_node},
                        {GetOutputVarNode("Out", node)});
  } else {
    auto target_dtype = ONNXDataType::FLOAT;
    if (GetInputVarNode("X", node)->Var()->GetDataType() == VarType::FP16) {
      target_dtype = ONNXDataType::FLOAT16;
    }

    auto pow_factor = AttributeMap{{"value", std::vector<float>{p}},
                                   {"dims", std::vector<int64_t>{1}},
                                   {"dtype", target_dtype}};
    auto *pow_factor_node =
        CreateBaseOp(graph, node, "popart_constant", {}, {}, pow_factor)
            ->outputs[0];
    auto *pow_node =
        CreateBaseOp(graph, node, "popart_pow", {abs_node, pow_factor_node}, {})
            ->outputs[0];
    auto *sum_node = CreateBaseOp(graph,
                                  node,
                                  "popart_reducesum",
                                  {pow_node},
                                  {},
                                  {{"keepdims", int64_t{0}}})
                         ->outputs[0];
    auto *s_node =
        CreateBaseOp(
            graph, node, "popart_reshape", {sum_node, target_shape_node}, {})
            ->outputs[0];
    auto *p_1 =
        CreateBaseOp(graph, node, "popart_reciprocal", {pow_factor_node}, {})
            ->outputs[0];
    return CreateBaseOp(graph,
                        node,
                        "popart_pow",
                        {s_node, p_1},
                        {GetOutputVarNode("Out", node)});
  }
}

Node *expand_as_v2_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  Node *shape = nullptr;
  auto op_inputs = op->Inputs();
  // PopART Expand Op only support the constant tensor as the input `shape`.
  if (op_inputs.find("target_tensor") != op_inputs.end()) {
    PADDLE_THROW(common::errors::Unimplemented(
        "Do not support input tensor `target_tensor`. Please use the attribute "
        "`target_shape`."));
  }
  auto input_shape = GetInputVarNode("X", node)->Var()->GetShape();
  auto shape_value =
      PADDLE_GET_CONST(std::vector<int>, op->GetAttr("target_shape"));
  // Check the dimensions
  int input_shape_index = input_shape.size() - 1;
  int target_shape_index = shape_value.size() - 1;
  while (input_shape_index >= 0) {
    if (input_shape[input_shape_index] !=
            int64_t(shape_value[target_shape_index]) &&
        input_shape[input_shape_index] != int64_t(1)) {
      PADDLE_THROW(common::errors::Unimplemented(
          "For input and `shape`, corresponding dimensions must have the same "
          "value or input dim = 1."));
    }
    target_shape_index--;
    input_shape_index--;
  }
  shape = CreateConst(
              graph,
              node,
              {},
              {},
              {{"value",
                std::vector<int64_t>{shape_value.begin(), shape_value.end()}},
               {"dims", std::vector<int64_t>{int64_t(shape_value.size())}},
               {"dtype", ONNXDataType::INT64}})
              ->outputs[0];
  return CreateBaseOp(graph,
                      node,
                      "popart_expand",
                      {GetInputVarNode("X", node), shape},
                      {GetOutputVarNode("Out", node)});
}

Node *expand_v2_handler(Graph *graph, Node *node) {
  auto *op = node->Op();

  // PopART Expand Op only support the constant tensor as the input `shape`.
  if (op->Input("Shape").size()) {
    PADDLE_THROW(
        common::errors::Unimplemented("Do not support input tensor `Shape`. "
                                      "Please use the attribute `shape`."));
  }
  if (op->Input("expand_shapes_tensor").size()) {
    PADDLE_THROW(common::errors::Unimplemented(
        "Do not support input tensor `expand_shapes_tensor`. Please use the "
        "attribute `shape`."));
  }
  auto input_shape = GetInputVarNode("X", node)->Var()->GetShape();
  auto shape_value = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("shape"));
  // Check the dimensions
  int input_shape_index = input_shape.size() - 1;
  int target_shape_index = shape_value.size() - 1;
  while (input_shape_index >= 0) {
    if (input_shape[input_shape_index] !=
            int64_t(shape_value[target_shape_index]) &&
        input_shape[input_shape_index] != int64_t(1)) {
      PADDLE_THROW(common::errors::Unimplemented(
          "For input and `shape`, corresponding dimensions must have the same "
          "value or input dim = 1."));
    }
    target_shape_index--;
    input_shape_index--;
  }

  auto *shape =
      CreateConst(
          graph,
          node,
          {},
          {},
          {{"value",
            std::vector<int64_t>{shape_value.begin(), shape_value.end()}},
           {"dims", std::vector<int64_t>{int64_t(shape_value.size())}},
           {"dtype", ONNXDataType::INT64}})
          ->outputs[0];

  return CreateBaseOp(graph,
                      node,
                      "popart_expand",
                      {GetInputVarNode("X", node), shape},
                      {GetOutputVarNode("Out", node)});
}

Node *flatten_contiguous_range_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto start_axis = PADDLE_GET_CONST(int, op->GetAttr("start_axis"));
  auto stop_axis = PADDLE_GET_CONST(int, op->GetAttr("stop_axis"));
  auto input_rank = GetInputVarNode("X", node)->Var()->GetShape().size();

  if (start_axis < 0) {
    start_axis += input_rank;
  }
  if (stop_axis < 0) {
    stop_axis += input_rank;
  }

  std::vector<int64_t> target_shape;
  if (start_axis == 0 && stop_axis == input_rank - 1) {
    target_shape.push_back(-1);
  } else {
    auto input_shape = GetInputVarNode("X", node)->Var()->GetShape();
    if (start_axis == 0) {
      target_shape.assign(input_shape.begin() + stop_axis + 1,
                          input_shape.end());
      target_shape.insert(target_shape.begin(), -1);
    } else if (stop_axis == input_rank - 1) {
      target_shape.assign(input_shape.begin(),
                          input_shape.begin() + start_axis);
      target_shape.push_back(-1);
    } else {
      target_shape.insert(target_shape.begin(),
                          input_shape.begin(),
                          input_shape.begin() + start_axis);
      target_shape.push_back(-1);
      target_shape.insert(target_shape.end(),
                          input_shape.begin() + stop_axis + 1,
                          input_shape.end());
    }
  }
  auto *unknown_dim_node = CreateConst(graph,
                                       node,
                                       target_shape,
                                       {int64_t(target_shape.size())},
                                       ONNXDataType::INT64)
                               ->outputs[0];
  return CreateBaseOp(graph,
                      node,
                      "popart_reshape",
                      {GetInputVarNode("X", node), unknown_dim_node},
                      {GetOutputVarNode("Out", node)},
                      {});
}

Node *flip_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto axes = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("axis"));
  auto input_shape = GetInputVarNode("X", node)->Var()->GetShape();
  for (auto it = axes.begin(); it != axes.end();) {
    if (*it < 0) {
      *it += input_shape.size();
    }
    // Remove input_shape[axis] == 1
    if (input_shape[*it] == 1) {
      it = axes.erase(it);
    } else {
      it++;
    }
  }
  auto *temp_node = GetInputVarNode("X", node);
  for (auto i = 0; i < axes.size(); i++) {
    auto axis = axes[i];
    std::vector<int64_t> split;
    split.resize(input_shape[axis], 1);
    auto splits_outputs =
        CreateSplit(graph, node, {temp_node}, {}, split, axis)->outputs;
    std::reverse(splits_outputs.begin(), splits_outputs.end());
    if (i != axes.size() - 1) {
      temp_node = CreateBaseOp(graph,
                               node,
                               "popart_concat",
                               splits_outputs,
                               {},
                               {{"axis", int64_t(axis)}})
                      ->outputs[0];
    } else {
      temp_node = CreateBaseOp(graph,
                               node,
                               "popart_concat",
                               splits_outputs,
                               {},
                               {{"axis", int64_t(axis)}})
                      ->outputs[0];
    }
  }
  // In case of `axis` is empty. Identity Op will be deleted in passes.
  return CreateBaseOp(graph,
                      node,
                      "popart_identity",
                      {temp_node},
                      {GetOutputVarNode("Out", node)},
                      {});
}

Node *meshgrid_handler(Graph *graph, Node *node) {
  Node *res = nullptr;
  // All inputs are 1-D tensors
  std::vector<int64_t> out_shape;
  for (auto input : node->inputs) {
    auto input_shape = input->Var()->GetShape();
    out_shape.push_back(input_shape[0]);
  }
  // Expand Op only allows a const tensor as `shape`
  auto *out_shape_node = CreateConst(graph,
                                     node,
                                     out_shape,
                                     {int64_t(out_shape.size())},
                                     ONNXDataType::INT64)
                             ->outputs[0];

  for (int i = 0; i < node->inputs.size(); i++) {
    // Reshape each input tensor to [node->inputs.size()] by filling with 1
    std::vector<int64_t> target_shape(node->inputs.size(), 1);
    target_shape[i] = node->inputs[i]->Var()->GetShape()[0];
    auto *target_shape_node = CreateConst(graph,
                                          node,
                                          target_shape,
                                          {int64_t(target_shape.size())},
                                          ONNXDataType::INT64)
                                  ->outputs[0];
    auto *t_reshaped = CreateBaseOp(graph,
                                    node,
                                    "popart_reshape",
                                    {node->inputs[i], target_shape_node},
                                    {},
                                    {})
                           ->outputs[0];
    res = CreateBaseOp(graph,
                       node,
                       "popart_expand",
                       {t_reshaped, out_shape_node},
                       {node->outputs[i]});
  }
  return res;
}

Node *p_norm_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto keepdim = PADDLE_GET_CONST(bool, op->GetAttr("keepdim"));
  auto axis = PADDLE_GET_CONST(int, op->GetAttr("axis"));
  auto porder = PADDLE_GET_CONST(float, op->GetAttr("porder"));

  auto target_dtype = ONNXDataType::FLOAT;
  if (GetInputVarNode("X", node)->Var()->GetDataType() == VarType::FP16) {
    target_dtype = ONNXDataType::FLOAT16;
  }

  auto *pnode = CreateConst(graph,
                            node,
                            std::vector<float>{porder},
                            std::vector<int64_t>{1},
                            target_dtype)
                    ->outputs[0];
  auto *abs_node =
      CreateBaseOp(graph, node, "popart_abs", {GetInputVarNode("X", node)}, {})
          ->outputs[0];
  auto *pow_node =
      CreateBaseOp(graph, node, "popart_pow", {abs_node, pnode}, {})
          ->outputs[0];
  auto *reducesum_node = CreateBaseOp(graph,
                                      node,
                                      "popart_reducesum",
                                      {pow_node},
                                      {},
                                      {{"axes", std::vector<int64_t>{axis}},
                                       {"keepdims", int64_t(keepdim)}})
                             ->outputs[0];
  auto *pnode1 =
      CreateConst(graph,
                  node,
                  std::vector<float>{static_cast<float>(1.0 / porder)},
                  std::vector<int64_t>{1},
                  target_dtype)
          ->outputs[0];
  return CreateBaseOp(graph,
                      node,
                      "popart_pow",
                      {reducesum_node, pnode1},
                      {GetOutputVarNode("Out", node)});
}

Node *tile_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto inputs = op->Inputs();
  bool is_repeat_tensors =
      (inputs.find("RepeatTimes") != inputs.end() &&
       !inputs.at("RepeatTimes").empty()) ||
      (inputs.find("repeat_times_tensor") != inputs.end() &&
       !inputs.at("repeat_times_tensor").empty());
  if (is_repeat_tensors) {
    PADDLE_THROW(
        common::errors::Unimplemented("Do not support repeats as tensors."));
  }
  auto repeat_times =
      PADDLE_GET_CONST(std::vector<int>, op->GetAttr("repeat_times"));
  int nums = repeat_times.size();
  std::vector<int> ones(
      GetInputVarNode("X", node)->Var()->GetShape().size() - nums, 1);
  repeat_times.insert(repeat_times.begin(), ones.begin(), ones.end());
  auto *repeat_node = CreateConst(graph,
                                  node,
                                  std::vector<int64_t>{repeat_times.begin(),
                                                       repeat_times.end()},
                                  {int64_t(repeat_times.size())},
                                  ONNXDataType::INT64)
                          ->outputs[0];
  return CreateBaseOp(graph,
                      node,
                      "popart_tile",
                      {GetInputVarNode("X", node), repeat_node},
                      {GetOutputVarNode("Out", node)});
}

Node *unstack_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto axis = PADDLE_GET_CONST(int, op->GetAttr("axis"));
  if (axis < 0) {
    axis += GetInputVarNode("X", node)->Var()->GetShape().size();
  }
  std::vector<int64_t> split(node->outputs.size(), 1);
  auto split_output_nodes =
      CreateSplit(graph, node, {GetInputVarNode("X", node)}, {}, split, axis)
          ->outputs;
  Node *output = nullptr;
  for (int i = 0; i < split.size(); i++) {
    output = CreateBaseOp(graph,
                          node,
                          "popart_squeeze",
                          {split_output_nodes[i]},
                          {node->outputs[i]},
                          {{"axes", std::vector<int64_t>{axis}}});
  }
  return output;
}

Node *where_handler(Graph *graph, Node *node) {
  return CreateBaseOp(graph,
                      node,
                      "popart_where",
                      {GetInputVarNode("Condition", node),
                       GetInputVarNode("X", node),
                       GetInputVarNode("Y", node)},
                      {GetOutputVarNode("Out", node)});
}

}  // namespace
}  // namespace ipu
}  // namespace platform
}  // namespace paddle

REGISTER_HANDLER(fill_constant, fill_constant_handler);
REGISTER_HANDLER(gaussian_random, gaussian_random_handler);
REGISTER_HANDLER(uniform_random, uniform_random_handler);
REGISTER_HANDLER(transpose2, transpose_handler);
REGISTER_HANDLER(reshape2, reshape_handler);
REGISTER_HANDLER(flatten2, flatten2_handler);
REGISTER_HANDLER(flatten_contiguous_range, flatten_contiguous_range_handler);
REGISTER_HANDLER(gather, gather_handler);
REGISTER_HANDLER(squeeze2, squeeze_handler);
REGISTER_HANDLER(cast, cast_handler);
REGISTER_HANDLER(lookup_table, lookup_table_handler);
REGISTER_HANDLER(unsqueeze2, unsqueeze_handler);
REGISTER_HANDLER(concat, concat_handler);
REGISTER_HANDLER(stack, stack_handler);
REGISTER_HANDLER(shape, shape_handler);
REGISTER_HANDLER(slice, slice_handler);
REGISTER_HANDLER(strided_slice, strided_slice_handler);
REGISTER_HANDLER(expand, expand_handler);
REGISTER_HANDLER(expand_v2, expand_v2_handler);
REGISTER_HANDLER(expand_as_v2, expand_as_v2_handler);
REGISTER_HANDLER(assign, assign_handler);
REGISTER_HANDLER(assign_value, assign_value_handler);
REGISTER_HANDLER(fill_any_like, fill_any_like_handler);
REGISTER_HANDLER(lookup_table_v2, lookup_table_v2_handler);
REGISTER_HANDLER(split, split_handler);
REGISTER_HANDLER(one_hot, one_hot_handler);
REGISTER_HANDLER(one_hot_v2, one_hot_v2_handler);
REGISTER_HANDLER(dot, dot_handler);
REGISTER_HANDLER(clip, clip_handler);
REGISTER_HANDLER(dist, dist_handler);
REGISTER_HANDLER(flip, flip_handler);
REGISTER_HANDLER(meshgrid, meshgrid_handler);
REGISTER_HANDLER(p_norm, p_norm_handler);
REGISTER_HANDLER(share_data, share_data_handler);
REGISTER_HANDLER(tile, tile_handler);
REGISTER_HANDLER(unstack, unstack_handler);
REGISTER_HANDLER(where, where_handler);
