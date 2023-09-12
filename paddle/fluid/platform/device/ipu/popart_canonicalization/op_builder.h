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

#pragma once

#include "paddle/fluid/platform/device/ipu/ipu_names.h"
#include "paddle/fluid/platform/device/ipu/popart_canonicalization/canonicalization_utils.h"

using AttributeMap = paddle::framework::AttributeMap;
using Attribute = paddle::framework::Attribute;

namespace paddle {
namespace platform {
namespace ipu {

const std::string GenerateVarName();
const std::string CreateOpIdentifyId(Node *node);

Node *MakeVarNode(Graph *graph, Node *node);
Node *MakeOpNode(Graph *graph,
                 Node *node,
                 const std::string &type,
                 const std::vector<Node *> &inputs,
                 const std::vector<Node *> &outputs);

Node *CreateBaseOp(Graph *graph,
                   Node *node,
                   const std::string &type,
                   const std::vector<Node *> &inputs,
                   const std::vector<Node *> &outputs,
                   const AttributeMap &attrs = {});

Node *CreateConst(Graph *graph,
                  Node *node,
                  const std::vector<Node *> &inputs,
                  const std::vector<Node *> &outputs,
                  const AttributeMap &attrs);

template <typename T>
Node *CreateConst(Graph *graph,
                  Node *node,
                  const std::vector<T> &value,
                  const std::vector<int64_t> &dims,
                  ONNXDataType dtype) {
  return CreateConst(
      graph,
      node,
      {},
      {},
      AttributeMap{{"value", value}, {"dims", dims}, {"dtype", dtype}});
}

Node *CreateCast(Graph *graph,
                 Node *node,
                 const std::vector<Node *> &inputs,
                 const std::vector<Node *> &outputs,
                 const VarType::Type otype);

Node *CreateIdentityLossOp(Graph *graph,
                           Node *node,
                           const std::vector<Node *> &inputs,
                           const std::vector<Node *> &outputs,
                           int reduction);

Node *CreateGemm(Graph *graph,
                 Node *node,
                 const std::vector<Node *> &inputs,
                 const std::vector<Node *> &outputs,
                 int64_t transA = 0,
                 int64_t transB = 0,
                 float alpha = 1.0f,
                 float beta = 1.0f);

Node *CreateReshape(Graph *graph,
                    Node *node,
                    const std::vector<Node *> &inputs,
                    const std::vector<Node *> &outputs,
                    const std::vector<int64_t> &oshape);

Node *CreateConv(Graph *graph,
                 Node *node,
                 const std::vector<Node *> &inputs,
                 const std::vector<Node *> &outputs,
                 const std::vector<int64_t> &dilations = {1, 1},
                 int64_t group = 1,
                 const std::vector<int64_t> &kernel_shape = {},
                 const std::vector<int64_t> &pads = {0, 0, 0, 0},
                 const std::vector<int64_t> &strides = {1, 1});

Node *CreateSoftmaxOpset11(Graph *graph,
                           Node *node,
                           const std::vector<Node *> &inputs,
                           const std::vector<Node *> &outputs,
                           int64_t axis);

Node *CreateSlice(Graph *graph,
                  Node *node,
                  const std::vector<Node *> &inputs,
                  const std::vector<Node *> &outputs,
                  const std::vector<int> &starts,
                  const std::vector<int> &ends,
                  const std::vector<int> &axes,
                  const std::vector<int> &strides);

Node *CreateSplit(Graph *graph,
                  Node *node,
                  const std::vector<Node *> &inputs,
                  const std::vector<Node *> &outputs,
                  const std::vector<int64_t> &split,
                  const int64_t axis);

}  // namespace ipu
}  // namespace platform
}  // namespace paddle
