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

#include "paddle/fluid/platform/device/ipu/popart_canonicalization/op_builder.h"

namespace paddle {
namespace platform {
namespace ipu {

// singleton
static int var_count = 0;
static int op_count = 0;

const std::string GenerateVarName() {
  return std::string("_gen_var_") + std::to_string(var_count++);
}

const std::string GenerateOpName() {
  return std::string("_gen_op_") + std::to_string(op_count++);
}

const std::string CreateOpIdentifyId(Node *node) {
  // format:
  //   op_type/_gen_*
  // this name will be used as op name when exporting onnx model from popart
  auto op_type = node->Name();
  return {op_type + "/" + GenerateOpName()};
}

Node *MakeVarNode(Graph *graph, Node *node) {
  auto var_name = GenerateVarName();
  auto var_desc = std::make_unique<framework::VarDesc>(var_name);

  auto var = graph->CreateVarNode(var_desc.get());
  return var;
}

Node *MakeOpNode(Graph *graph, Node *node, const std::string &type,
                 const std::vector<Node *> &inputs,
                 const std::vector<Node *> &outputs) {
  auto op_desc = std::make_unique<framework::OpDesc>();
  op_desc->SetType(type);
  auto op = graph->CreateOpNode(op_desc.get());

  for (auto *in : inputs) {
    ConnectNodes(in, op);
  }
  if (outputs.empty()) {
    auto var = MakeVarNode(graph, node);
    ConnectNodes(op, var);
  } else {
    for (auto *out : outputs) {
      ConnectNodes(op, out);
    }
  }

  // i/o
  std::vector<std::string> input_names;
  for (auto node : op->inputs) {
    input_names.push_back(node->Name());
  }
  op->Op()->SetInput("__inputs__", input_names);
  std::vector<std::string> output_names;
  for (auto node : op->outputs) {
    output_names.push_back(node->Name());
  }
  op->Op()->SetOutput("__outputs__", output_names);
  op->Op()->Flush();

  return op;
}

Node *CreateBaseOp(Graph *graph, Node *node, const std::string &type,
                   const std::vector<Node *> &inputs,
                   const std::vector<Node *> &outputs,
                   const AttributeMap &attrs) {
  auto new_node = MakeOpNode(graph, node, type, inputs, outputs);
  if (!attrs.empty()) {
    new_node->Op()->SetAttrMap(attrs);
  }
  // deal special attr
  if (!new_node->Op()->HasAttr(sIpuIndexAttr)) {
    CopyOpAttr(sIpuIndexAttr, node->Op(), new_node->Op());
  }
  if (!new_node->Op()->HasAttr(sIpuStageAttr)) {
    CopyOpAttr(sIpuStageAttr, node->Op(), new_node->Op());
  }
  if (node->Op()->HasAttr(sMatmulSerializeFactor)) {
    CopyOpAttr(sMatmulSerializeFactor, node->Op(), new_node->Op());
  }
  if (node->Op()->HasAttr(sMatmulSerializeMode)) {
    CopyOpAttr(sMatmulSerializeMode, node->Op(), new_node->Op());
  }
  if (node->Op()->HasAttr(sAvailMemAttribute)) {
    CopyOpAttr(sAvailMemAttribute, node->Op(), new_node->Op());
  }
  if (node->Op()->HasAttr(sOpNamescope)) {
    CopyOpAttr(sOpNamescope, node->Op(), new_node->Op());
  }
  {
    new_node->Op()->SetAttr(sOpIdentifyIdAttr, CreateOpIdentifyId(node));
    new_node->Op()->Flush();
  }

  return new_node;
}

Node *CreateConst(Graph *graph, Node *node, const std::vector<Node *> &inputs,
                  const std::vector<Node *> &outputs,
                  const AttributeMap &attrs) {
  return CreateBaseOp(graph, node, "popart_constant", inputs, outputs, attrs);
}

Node *CreateCast(Graph *graph, Node *node, const std::vector<Node *> &inputs,
                 const std::vector<Node *> &outputs, const int otype) {
  auto to = VarType2PopStr(otype);
  return CreateBaseOp(graph, node, "popart_cast", inputs, outputs,
                      {{"to", to}});
}

Node *CreateGemm(Graph *graph, Node *node, const std::vector<Node *> &inputs,
                 const std::vector<Node *> &outputs, int64_t transA,
                 int64_t transB, float alpha, float beta) {
  return CreateBaseOp(graph, node, "popart_gemm", inputs, outputs,
                      {
                          {"alpha", alpha},
                          {"beta", beta},
                          {"transA", transA},
                          {"transB", transB},
                      });
}

Node *CreateReshape(Graph *graph, Node *node, const std::vector<Node *> &inputs,
                    const std::vector<Node *> &outputs,
                    const std::vector<int64_t> &oshape) {
  auto attr = AttributeMap{
      {"value", oshape},
      {"dims", std::vector<int64_t>{static_cast<int64_t>(oshape.size())}},
      {"dtype", ONNXDataType::INT64}};
  auto new_node_const =
      CreateBaseOp(graph, node, "popart_constant", {}, {}, attr);
  auto new_node_reshape =
      CreateBaseOp(graph, node, "popart_reshape",
                   {inputs[0], new_node_const->outputs[0]}, outputs);
  return new_node_reshape;
}

Node *CreateConv(Graph *graph, Node *node, const std::vector<Node *> &inputs,
                 const std::vector<Node *> &outputs,
                 const std::vector<int64_t> &dilations, int64_t group,
                 const std::vector<int64_t> &kernel_shape,
                 const std::vector<int64_t> &pads,
                 const std::vector<int64_t> &strides) {
  auto attrs = AttributeMap{
      {"dilations", dilations},       {"group", group},
      {"kernel_shape", kernel_shape}, {"pads", pads},
      {"strides", strides},
  };
  return CreateBaseOp(graph, node, "popart_conv", inputs, outputs, attrs);
}

Node *CreateSoftmaxOpset11(Graph *graph, Node *node,
                           const std::vector<Node *> &inputs,
                           const std::vector<Node *> &outputs, int64_t axis) {
  PADDLE_ENFORCE_EQ(inputs.size(), 1, platform::errors::InvalidArgument(
                                          "Softmax op only support one input"));
  auto x_shape = inputs[0]->Var()->GetShape();
  int x_rank = x_shape.size();
  if (axis < 0) {
    axis = axis + x_rank;
  }
  if (axis == x_rank - 1) {
    return CreateBaseOp(graph, node, "popart_softmax", inputs, outputs,
                        {{"axis", int64_t{-1}}});
  } else {
    auto perm = std::vector<int64_t>(x_rank);
    std::iota(perm.begin(), perm.end(), 0);
    perm[x_rank - 1] = axis;
    perm[axis] = x_rank - 1;
    auto new_transpose_pre = CreateBaseOp(graph, node, "popart_transpose",
                                          inputs, {}, {{"perm", perm}});
    auto new_softmax =
        CreateBaseOp(graph, node, "popart_softmax", new_transpose_pre->outputs,
                     {}, {{"axis", int64_t{-1}}});
    return CreateBaseOp(graph, node, "popart_transpose", new_softmax->outputs,
                        outputs, {{"perm", perm}});
  }
}

}  // namespace ipu
}  // namespace platform
}  // namespace paddle
