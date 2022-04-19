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
  if (!op->Input("ShapeTensor").empty()) {
    PADDLE_THROW(
        platform::errors::Unimplemented("op fill_constant with ShapeTensor"));
  }
  auto dtype_ = BOOST_GET_CONST(int, op->GetAttr("dtype"));
  auto dtype = VarType2OnnxDtype(dtype_);
  auto dims = BOOST_GET_CONST(std::vector<int64_t>, op->GetAttr("shape"));
  auto value_ = BOOST_GET_CONST(float, op->GetAttr("value"));
  size_t size = 1;
  for (auto &dim : dims) {
    size *= dim;
  }
  Attribute value;
  switch (dtype_) {
    case framework::proto::VarType::FP32:
      value = std::vector<float>(size, value_);
      break;
    case framework::proto::VarType::FP64:
      value = std::vector<double>(size, value_);
      break;
    case framework::proto::VarType::INT32:
      value = std::vector<int>(size, value_);
      break;
    case framework::proto::VarType::INT64:
      value = std::vector<int64_t>(size, value_);
      break;
    case framework::proto::VarType::BOOL:
      value = std::vector<bool>(size, value_);
      break;
    default:
      PADDLE_THROW(
          platform::errors::Unimplemented("fill_constant dtype: %d", dtype_));
  }
  return CreateConst(graph, node, node->inputs, node->outputs,
                     AttributeMap{
                         {"value", value}, {"dims", dims}, {"dtype", dtype},
                     });
}

Node *gaussian_random_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto shape = BOOST_GET_CONST(std::vector<int64_t>, op->GetAttr("shape"));
  auto dtype_ = BOOST_GET_CONST(int, op->GetAttr("dtype"));
  auto dtype = VarType2OnnxDtype(dtype_);
  auto mean = BOOST_GET_CONST(float, op->GetAttr("mean"));
  auto scale = BOOST_GET_CONST(float, op->GetAttr("std"));
  // seed not work
  auto seed_ = BOOST_GET_CONST(int, op->GetAttr("seed"));
  auto seed = static_cast<float>(seed_);
  return CreateBaseOp(graph, node, "popart_randomnormal", node->inputs,
                      node->outputs, {
                                         {"shape", shape},
                                         {"dtype", dtype},
                                         {"mean", mean},
                                         {"scale", scale},
                                         {"seed", seed},
                                     });
}

Node *uniform_random_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto shape = BOOST_GET_CONST(std::vector<int64_t>, op->GetAttr("shape"));
  auto dtype_ = BOOST_GET_CONST(int, op->GetAttr("dtype"));
  auto dtype = VarType2OnnxDtype(dtype_);
  auto high = BOOST_GET_CONST(float, op->GetAttr("max"));
  auto low = BOOST_GET_CONST(float, op->GetAttr("min"));
  // seed not work
  auto seed_ = BOOST_GET_CONST(int, op->GetAttr("seed"));
  auto seed = static_cast<float>(seed_);
  return CreateBaseOp(graph, node, "popart_randomuniform", node->inputs,
                      node->outputs, {
                                         {"shape", shape},
                                         {"dtype", dtype},
                                         {"high", high},
                                         {"low", low},
                                         {"seed", seed},
                                     });
}

Node *transpose_handler(Graph *graph, Node *node) {
  auto *op = node->Op();

  auto axis_ = BOOST_GET_CONST(std::vector<int>, op->GetAttr("axis"));
  std::vector<int64_t> perm(axis_.begin(), axis_.end());
  auto attrs = AttributeMap{{"perm", perm}};

  auto new_node_transpose =
      CreateBaseOp(graph, node, "popart_transpose", node->inputs,
                   {GetOutputVarNode("Out", node)}, attrs);
  return new_node_transpose;
}

Node *reshape_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto shape_ = BOOST_GET_CONST(std::vector<int>, op->GetAttr("shape"));
  std::vector<int64_t> shape(shape_.begin(), shape_.end());
  auto attrs = AttributeMap{
      {"value", shape},
      {"dims", std::vector<int64_t>{static_cast<int64_t>(shape.size())}},
      {"dtype", ONNXDataType::INT64}};
  auto new_node_const =
      CreateBaseOp(graph, node, "popart_constant", {}, {}, attrs);

  auto new_node_reshape =
      CreateBaseOp(graph, node, "popart_reshape",
                   {GetInputVarNode("X", node), new_node_const->outputs[0]},
                   {GetOutputVarNode("Out", node)}, {});
  return new_node_reshape;
}

Node *flatten2_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto axis = BOOST_GET_CONST(int, op->GetAttr("axis"));
  return CreateBaseOp(
      graph, node, "popart_flatten", {GetInputVarNode("X", node)},
      {GetOutputVarNode("Out", node)}, {{"axis", int64_t(axis)}});
}

Node *gather_handler(Graph *graph, Node *node) {
  auto new_node_gather =
      CreateBaseOp(graph, node, "popart_gather",
                   {GetInputVarNode("X", node), GetInputVarNode("Index", node)},
                   {GetOutputVarNode("Out", node)}, {});
  return new_node_gather;
}

Node *squeeze_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto axes_ = BOOST_GET_CONST(std::vector<int>, op->GetAttr("axes"));
  auto input_shape_ = GetInputVarNode("X", node)->Var()->GetShape();

  std::vector<int64_t> axes{axes_.begin(), axes_.end()};
  if (axes_.empty()) {
    for (int i = 0; i < input_shape_.size(); i++) {
      if (input_shape_[i] == 1) {
        axes.push_back(i);
      }
    }
  }
  auto new_node_squeeze =
      CreateBaseOp(graph, node, "popart_squeeze", {GetInputVarNode("X", node)},
                   {GetOutputVarNode("Out", node)}, {{"axes", axes}});

  return new_node_squeeze;
}

Node *cast_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto otype = BOOST_GET_CONST(int, op->GetAttr("out_dtype"));
  auto new_node_cast =
      CreateCast(graph, node, node->inputs, node->outputs, otype);
  return new_node_cast;
}

Node *lookup_table_op_handler(Graph *graph, Node *node,
                              const std::string &type) {
  auto *op = node->Op();
  auto padding_idx_ = BOOST_GET_CONST(int64_t, op->GetAttr("padding_idx"));
  auto w_shape_ = GetInputVarNode("W", node)->Var()->GetShape();
  auto table_size_ = w_shape_[0];
  auto emb_size_ = w_shape_[1];

  Node *w_node;
  if (padding_idx_ >= 0 && padding_idx_ < table_size_) {
    std::vector<float> const_value_(emb_size_, 0);
    std::vector<int64_t> const_shape_{1, emb_size_};
    auto concat_const =
        CreateConst(graph, node, {}, {}, {{"value", const_value_},
                                          {"dims", const_shape_},
                                          {"dtype", GetOutputVarDtype(node)}});
    auto axes =
        CreateConst(graph, node, {}, {}, {{"value", std::vector<int64_t>{0}},
                                          {"dims", std::vector<int64_t>{1}},
                                          {"dtype", ONNXDataType::INT64}});
    auto step =
        CreateConst(graph, node, {}, {}, {{"value", std::vector<int64_t>{1}},
                                          {"dims", std::vector<int64_t>{1}},
                                          {"dtype", ONNXDataType::INT64}});

    auto left_start =
        CreateConst(graph, node, {}, {}, {{"value", std::vector<int64_t>{0}},
                                          {"dims", std::vector<int64_t>{1}},
                                          {"dtype", ONNXDataType::INT64}});
    auto left_end = CreateConst(graph, node, {}, {},
                                {{"value", std::vector<int64_t>{padding_idx_}},
                                 {"dims", std::vector<int64_t>{1}},
                                 {"dtype", ONNXDataType::INT64}});

    auto right_start = CreateConst(
        graph, node, {}, {}, {{"value", std::vector<int64_t>{padding_idx_ + 1}},
                              {"dims", std::vector<int64_t>{1}},
                              {"dtype", ONNXDataType::INT64}});
    auto right_end = CreateConst(graph, node, {}, {},
                                 {{"value", std::vector<int64_t>{table_size_}},
                                  {"dims", std::vector<int64_t>{1}},
                                  {"dtype", ONNXDataType::INT64}});

    auto left_slice =
        CreateBaseOp(graph, node, "popart_slice",
                     {GetInputVarNode("W", node), left_start->outputs[0],
                      left_end->outputs[0], axes->outputs[0], step->outputs[0]},
                     {}, {});
    auto right_slice = CreateBaseOp(
        graph, node, "popart_slice",
        {GetInputVarNode("W", node), right_start->outputs[0],
         right_end->outputs[0], axes->outputs[0], step->outputs[0]},
        {}, {});

    if (padding_idx_ == 0) {
      w_node = CreateBaseOp(graph, node, "popart_concat",
                            {concat_const->outputs[0], right_slice->outputs[0]},
                            {}, {{"axis", int64_t(0)}});
      ClearNode(left_start);
      ClearNode(left_end);
      ClearNode(left_slice);
    } else if (padding_idx_ == table_size_ - 1) {
      w_node = CreateBaseOp(graph, node, "popart_concat",
                            {left_slice->outputs[0], concat_const->outputs[0]},
                            {}, {{"axis", int64_t{0}}});
      ClearNode(right_start);
      ClearNode(right_end);
      ClearNode(right_slice);
    } else {
      w_node = CreateBaseOp(graph, node, "popart_concat",
                            {left_slice->outputs[0], concat_const->outputs[0],
                             right_slice->outputs[0]},
                            {}, {{"axis", int64_t{0}}});
    }
    w_node = w_node->outputs[0];
  } else {
    w_node = GetInputVarNode("W", node);
  }

  // lookup_table and lookup_table_v2
  auto ids = GetInputVarNode("Ids", node);
  if (type == "v1") {
    ids = CreateBaseOp(graph, node, "popart_squeeze",
                       {GetInputVarNode("Ids", node)}, {},
                       {{"axes", std::vector<int64_t>{-1}}});
    ids = ids->outputs[0];
  }

  auto gather = CreateBaseOp(graph, node, "popart_gather", {w_node, ids},
                             {GetOutputVarNode("Out", node)}, {});
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
  auto axes_ = BOOST_GET_CONST(std::vector<int>, op->GetAttr("axes"));
  std::vector<int64_t> axes{axes_.begin(), axes_.end()};
  auto new_node_unsqueeze = CreateBaseOp(
      graph, node, "popart_unsqueeze", {GetInputVarNode("X", node)},
      {GetOutputVarNode("Out", node)}, {{"axes", axes}});

  return new_node_unsqueeze;
}

Node *concat_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  int64_t axis_{BOOST_GET_CONST(int, op->GetAttr("axis"))};

  auto new_node_concat =
      CreateBaseOp(graph, node, "popart_concat", node->inputs, node->outputs,
                   {{"axis", axis_}});
  return new_node_concat;
}

Node *stack_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  int64_t axis_{BOOST_GET_CONST(int, op->GetAttr("axis"))};
  std::vector<int64_t> axes_{axis_};

  std::vector<Node *> unsqueeze_outputs_{};
  for (auto input : node->inputs) {
    auto new_unsqueeze_node = CreateBaseOp(graph, node, "popart_unsqueeze",
                                           {input}, {}, {{"axes", axes_}});
    unsqueeze_outputs_.push_back(new_unsqueeze_node->outputs[0]);
    for (size_t i = 0; i < input->outputs.size(); ++i) {
      if (input->outputs[i] == node) {
        input->outputs[i] = new_unsqueeze_node;
        break;
      }
    }
  }
  auto new_node_concat =
      CreateBaseOp(graph, node, "popart_concat", unsqueeze_outputs_,
                   {GetOutputVarNode("Y", node)}, {{"axis", axis_}});
  return new_node_concat;
}

Node *shape_handler(Graph *graph, Node *node) {
  auto new_node =
      CreateBaseOp(graph, node, "popart_shape", node->inputs, node->outputs);
  return new_node;
}

Node *slice_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  Node *starts = nullptr;
  if (!op->HasAttr("starts")) {
    starts = GetInputVarNode("StartsTensor", node);
  } else {
    auto starts_ = BOOST_GET_CONST(std::vector<int>, op->GetAttr("starts"));
    auto dim = int64_t(starts_.size());
    auto attr = MakeConstAttrMap<int>(starts_, {dim}, ONNXDataType::INT32);
    starts = CreateConst(graph, node, {}, {}, attr);
    starts = starts->outputs[0];
  }
  Node *ends = nullptr;
  if (!op->HasAttr("ends")) {
    ends = GetInputVarNode("EndsTensor", node);
  } else {
    auto ends_ = BOOST_GET_CONST(std::vector<int>, op->GetAttr("ends"));
    auto dim = int64_t(ends_.size());
    auto attr = MakeConstAttrMap<int>(ends_, {dim}, ONNXDataType::INT32);
    ends = CreateConst(graph, node, {}, {}, attr);
    ends = ends->outputs[0];
  }
  Node *axes = nullptr;
  {
    auto axes_ = BOOST_GET_CONST(std::vector<int>, op->GetAttr("axes"));
    auto dim = int64_t(axes_.size());
    auto attr = MakeConstAttrMap<int>(axes_, {dim}, ONNXDataType::INT32);
    axes = CreateConst(graph, node, {}, {}, attr);
  }

  auto decrease_axis_ =
      BOOST_GET_CONST(std::vector<int>, op->GetAttr("decrease_axis"));
  auto input_shape_ = GetInputVarNode("Input", node)->Var()->GetShape();
  auto output_shape_ = GetOutputVarNode("Out", node)->Var()->GetShape();
  if (decrease_axis_.size() == 0) {
    return CreateBaseOp(
        graph, node, "popart_slice",
        {GetInputVarNode("Input", node), starts, ends, axes->outputs[0]},
        node->outputs);
  } else if (output_shape_ == std::vector<int64_t>{0} ||
             input_shape_.size() > output_shape_.size()) {
    auto slice = CreateBaseOp(
        graph, node, "popart_slice",
        {GetInputVarNode("Input", node), starts, ends, axes->outputs[0]}, {},
        {});
    return CreateBaseOp(graph, node, "popart_squeeze", {slice->outputs[0]},
                        {GetOutputVarNode("Out", node)},
                        {{"axes", std::vector<int64_t>{decrease_axis_.begin(),
                                                       decrease_axis_.end()}}});
  } else {
    return CreateBaseOp(
        graph, node, "popart_slice",
        {GetInputVarNode("Input", node), starts, ends, axes->outputs[0]},
        node->outputs);
  }
}

Node *expand_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  if (!op->Input("expand_times_tensor").empty()) {
    PADDLE_THROW(
        platform::errors::Unimplemented("Expand op with expand_times_tensor"));
  }

  Node *expand_times = nullptr;
  if (!op->Input("ExpandTimes").empty()) {
    // cast to int64
    expand_times =
        CreateCast(graph, node, {GetInputVarNode("ExpandTimes", node)}, {},
                   framework::proto::VarType::INT64);
  } else {
    auto expand_times_i32 =
        BOOST_GET_CONST(std::vector<int>, op->GetAttr("expand_times"));
    auto expand_times_ =
        std::vector<int64_t>{expand_times_i32.begin(), expand_times_i32.end()};
    auto dim = int64_t(expand_times_.size());
    auto attr =
        MakeConstAttrMap<int64_t>(expand_times_, {dim}, ONNXDataType::INT64);
    expand_times = CreateConst(graph, node, {}, {}, attr);
  }
  auto new_node = CreateBaseOp(
      graph, node, "popart_tile",
      {GetInputVarNode("X", node), expand_times->outputs[0]}, node->outputs);
  return new_node;
}

Node *assign_handler(Graph *graph, Node *node) {
  return CreateBaseOp(graph, node, "popart_identity",
                      {GetInputVarNode("X", node)},
                      {GetOutputVarNode("Out", node)}, {});
}

Node *assign_value_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto dtype_ = BOOST_GET_CONST(int, op->GetAttr("dtype"));
  auto dtype = VarType2OnnxDtype(dtype_);
  auto dims_ = BOOST_GET_CONST(std::vector<int>, op->GetAttr("shape"));
  std::vector<int64_t> dims(dims_.begin(), dims_.end());
  Attribute values;
  std::string value_name;
  switch (dtype_) {
    case framework::proto::VarType::BOOL: {
      value_name = "bool_values";
      auto vec_int = BOOST_GET_CONST(std::vector<int>, op->GetAttr(value_name));
      std::vector<bool> vec_bool(vec_int.begin(), vec_int.end());
      values = vec_bool;
    } break;
    case framework::proto::VarType::INT32:
      value_name = "int32_values";
      values = BOOST_GET_CONST(std::vector<int>, op->GetAttr(value_name));
      break;
    case framework::proto::VarType::FP32:
      value_name = "fp32_values";
      values = BOOST_GET_CONST(std::vector<float>, op->GetAttr(value_name));
      break;
    case framework::proto::VarType::INT64:
      value_name = "int64_values";
      values = BOOST_GET_CONST(std::vector<int64_t>, op->GetAttr(value_name));
      break;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unsupported data type(code %d) for AssignValue operator, only "
          "supports bool, int32, float32 and int64.",
          dtype));
  }
  return CreateConst(graph, node, node->inputs, node->outputs,
                     AttributeMap{
                         {"value", values}, {"dims", dims}, {"dtype", dtype},
                     });
}

Node *fill_any_like_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto value = BOOST_GET_CONST(float, op->GetAttr("value"));
  auto x_shape = GetInputVarNode("X", node)->Var()->GetShape();
  auto dtype = BOOST_GET_CONST(int, op->GetAttr("dtype"));
  auto x_dtype = static_cast<framework::proto::VarType::Type>(dtype);
  size_t size = 1;
  for (auto &dim : x_shape) {
    size *= dim;
  }

  Attribute out_value;
  switch (x_dtype) {
    case framework::proto::VarType::FP32:
      out_value = std::vector<float>(size, value);
      break;
    case framework::proto::VarType::FP64:
      out_value = std::vector<double>(size, value);
      break;
    case framework::proto::VarType::INT32:
      out_value = std::vector<int>(size, value);
      break;
    case framework::proto::VarType::INT64:
      out_value = std::vector<int64_t>(size, value);
      break;
    case framework::proto::VarType::BOOL:
      out_value = std::vector<int64_t>(size, value);
      break;
    default:
      PADDLE_THROW(
          platform::errors::Unimplemented("fill_any_like dtype: %d", x_dtype));
  }
  return CreateConst(graph, node, node->inputs, node->outputs,
                     AttributeMap{
                         {"value", out_value},
                         {"dims", x_shape},
                         {"dtype", VarType2OnnxDtype(dtype)},
                     });
}

Node *one_hot_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto depth = BOOST_GET_CONST(int, op->GetAttr("depth"));
  auto allow_out_of_range =
      BOOST_GET_CONST(bool, op->GetAttr("allow_out_of_range"));
  if (allow_out_of_range) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Do not support allow_out_of_range=True"));
  } else {
    auto depth_tensor = CreateConst(graph, node, {}, {},
                                    {{"value", std::vector<int64_t>{depth}},
                                     {"dims", std::vector<int64_t>{1}},
                                     {"dtype", ONNXDataType::INT64}});
    auto value_tensor =
        CreateConst(graph, node, {}, {}, {{"value", std::vector<float>{0, 1}},
                                          {"dims", std::vector<int64_t>{2}},
                                          {"dtype", ONNXDataType::FLOAT}});
    return CreateBaseOp(graph, node, "popart_onehot",
                        {GetInputVarNode("X", node), depth_tensor->outputs[0],
                         value_tensor->outputs[0]},
                        {GetOutputVarNode("Out", node)},
                        {{"axis", int64_t{-1}}});
  }
}

Node *one_hot_v2_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto depth = BOOST_GET_CONST(int, op->GetAttr("depth"));
  auto allow_out_of_range =
      BOOST_GET_CONST(bool, op->GetAttr("allow_out_of_range"));
  if (allow_out_of_range) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Do not support allow_out_of_range=True"));
  } else {
    auto depth_tensor =
        CreateConst(graph, node, {}, {}, {{"value", std::vector<int>{depth}},
                                          {"dims", std::vector<int64_t>{1}},
                                          {"dtype", ONNXDataType::INT32}});
    Node *value_tensor = nullptr;
    if (GetOutputVarNode("Out", node)->Var()->GetDataType() ==
        framework::proto::VarType::FP16) {
      value_tensor =
          CreateConst(graph, node, {}, {}, {{"value", std::vector<float>{0, 1}},
                                            {"dims", std::vector<int64_t>{2}},
                                            {"dtype", ONNXDataType::FLOAT16}});
    } else {
      value_tensor =
          CreateConst(graph, node, {}, {}, {{"value", std::vector<float>{0, 1}},
                                            {"dims", std::vector<int64_t>{2}},
                                            {"dtype", ONNXDataType::FLOAT}});
    }

    return CreateBaseOp(graph, node, "popart_onehot",
                        {GetInputVarNode("X", node), depth_tensor->outputs[0],
                         value_tensor->outputs[0]},
                        {GetOutputVarNode("Out", node)},
                        {{"axis", int64_t{-1}}});
  }
}

Node *split_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto axis = BOOST_GET_CONST(int, op->GetAttr("axis"));
  auto sections = BOOST_GET_CONST(std::vector<int>, op->GetAttr("sections"));
  return CreateBaseOp(
      graph, node, "popart_split", {GetInputVarNode("X", node)}, node->outputs,
      {{"num_outputs", int64_t(sections.size())},
       {"axis", int64_t(axis)},
       {"split", std::vector<int64_t>{sections.begin(), sections.end()}}});
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
REGISTER_HANDLER(gather, gather_handler);
REGISTER_HANDLER(squeeze2, squeeze_handler);
REGISTER_HANDLER(cast, cast_handler);
REGISTER_HANDLER(lookup_table, lookup_table_handler);
REGISTER_HANDLER(unsqueeze2, unsqueeze_handler);
REGISTER_HANDLER(concat, concat_handler);
REGISTER_HANDLER(stack, stack_handler);
REGISTER_HANDLER(shape, shape_handler);
REGISTER_HANDLER(slice, slice_handler);
REGISTER_HANDLER(expand, expand_handler);
REGISTER_HANDLER(assign, assign_handler);
REGISTER_HANDLER(assign_value, assign_value_handler);
REGISTER_HANDLER(fill_any_like, fill_any_like_handler);
REGISTER_HANDLER(lookup_table_v2, lookup_table_v2_handler);
REGISTER_HANDLER(split, split_handler);
REGISTER_HANDLER(one_hot, one_hot_handler);
REGISTER_HANDLER(one_hot_v2, one_hot_v2_handler);
