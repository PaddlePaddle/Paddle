// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>

#include "paddle/fluid/framework/ir/graph_helper.h"

namespace paddle {
namespace framework {
namespace ir {

using StringPairMap =
    std::unordered_map<std::string, std::pair<bool, phi::DenseTensor>>;

static void SaveInfoInTheTmpOp(
    ir::Graph* graph,
    const std::string& flag,
    const std::string& key_suffix,
    const std::unordered_map<std::string, std::vector<float>>& info_map) {
  VLOG(3) << "save variables in the first op's attr";

  const std::string suffix = "_" + key_suffix + "_" + flag;
  OpDesc op_desc;
  op_desc.SetType("save");
  auto* op_node = graph->CreateOpNode(&op_desc);

  op_node->Op()->SetAttr(flag, true);
  for (auto iter = info_map.begin(); iter != info_map.end(); ++iter) {
    op_node->Op()->SetAttr(iter->first + suffix, iter->second);
  }
}

static void SaveInfoInTheTmpOp(ir::Graph* graph,
                               const std::string& flag,
                               const std::string& key_suffix,
                               const StringPairMap& info_map) {
  VLOG(3) << "save variables in the first op's attr";

  const std::string suffix = "_" + key_suffix + "_" + flag;

  OpDesc op_desc;
  op_desc.SetType("save");
  auto* op_node = graph->CreateOpNode(&op_desc);

  op_node->Op()->SetAttr(flag, true);
  for (auto iter = info_map.begin(); iter != info_map.end(); ++iter) {
    auto* data = iter->second.second.data<float>();
    std::vector<float> data_v(data, data + iter->second.second.numel());
    op_node->Op()->SetAttr(iter->first + suffix + "_unsigned",
                           iter->second.first);
    op_node->Op()->SetAttr(iter->first + suffix, data_v);
  }
}

static void GetInfoFromTheTmpOp(
    ir::Graph* graph,
    const std::string& flag,
    const std::string& key_suffix,
    std::unordered_map<std::string, std::vector<float>>* info_map) {
  VLOG(3) << "get variables from the first op's attr";

  const std::string suffix = "_" + key_suffix + "_" + flag;
  for (auto* op_node :
       ir::TopologyVariantSort(*graph, static_cast<ir::SortKind>(0))) {
    if (!op_node->IsOp() || op_node->Op()->Type() != "save") continue;

    auto* op_desc = op_node->Op();
    if (op_desc->GetAttrIfExists<bool>(flag)) {
      op_desc->RemoveAttr(flag);
      std::vector<std::string> attr_names = op_desc->AttrNames();
      for (auto fake_name : attr_names) {
        size_t pos = fake_name.find(suffix);
        if (pos != std::string::npos) {
          std::string name = fake_name.substr(0, pos);
          auto scales_vector =
              PADDLE_GET_CONST(std::vector<float>, op_desc->GetAttr(fake_name));
          info_map->insert(std::make_pair(name, scales_vector));
          op_desc->RemoveAttr(fake_name);
        }
      }
      graph->RemoveNode(op_node);
      break;
    }
  }
}

static void GetInfoFromTheTmpOp(ir::Graph* graph,
                                const std::string& flag,
                                const std::string& key_suffix,
                                StringPairMap* info_map) {
  VLOG(3) << "get variables from the first op's attr";
  const std::string unsigned_flag = "_unsigned";
  const std::string suffix = "_" + key_suffix + "_" + flag;
  const std::string suffix_is_unsigned = suffix + unsigned_flag;
  for (auto* op_node :
       ir::TopologyVariantSort(*graph, static_cast<ir::SortKind>(0))) {
    if (!op_node->IsOp() || op_node->Op()->Type() != "save") continue;

    auto* op_desc = op_node->Op();
    if (op_desc->GetAttrIfExists<bool>(flag)) {
      op_desc->RemoveAttr(flag);
      std::vector<std::string> attr_names = op_desc->AttrNames();
      for (auto fake_name : attr_names) {
        auto is_unsigned = false;
        size_t pos = fake_name.find(suffix_is_unsigned);

        if (pos != std::string::npos) {
          std::string unsigned_var_name = fake_name;
          is_unsigned =
              PADDLE_GET_CONST(bool, op_desc->GetAttr(unsigned_var_name));

          std::string var_name = fake_name.substr(0, pos);
          size_t unsigned_pos = fake_name.find(unsigned_flag);
          std::string vector_name =
              fake_name.erase(unsigned_pos, unsigned_flag.length());
          auto scales_vector = PADDLE_GET_CONST(std::vector<float>,
                                                op_desc->GetAttr(vector_name));
          phi::DenseTensor tensor;
          const int size = static_cast<int>(scales_vector.size());
          auto data = tensor.mutable_data<double>({size}, phi::CPUPlace());
          std::copy(scales_vector.begin(), scales_vector.end(), data);
          auto pair = std::make_pair(is_unsigned, tensor);
          info_map->insert(std::make_pair(var_name, pair));
          op_desc->RemoveAttr(unsigned_var_name);
          op_desc->RemoveAttr(vector_name);
        }
      }
      graph->RemoveNode(op_node);
      break;
    }
  }
}

inline void ConvertToFusedOp(OpDesc* op) {
  const std::map<std::string, std::string> fused_ops = {
      {"conv2d", "fused_conv2d"},
      {"conv2d_transpose", "conv2d_transpose_bias"},
      {"depthwise_conv2d", "fused_conv2d"},
      {"elementwise_add", "fused_elementwise_add"},
      {"elementwise_sub", "fused_elementwise_sub"},
      {"elementwise_mul", "fused_elementwise_mul"},
      {"elementwise_div", "fused_elementwise_div"},
      {"matmul", "fused_matmul"},
      {"matmul_v2", "fused_matmul"},
      {"softplus", "fused_softplus"},
      {"transpose2", "fused_transpose"}};

  if (op->Type() == "matmul") {
    op->SetAttr("trans_x", op->GetAttr("transpose_X"));
    op->SetAttr("trans_y", op->GetAttr("transpose_Y"));
    op->SetAttr("matmul_alpha", op->GetAttr("alpha"));
  }

  auto it = fused_ops.find(op->Type());
  if (it != fused_ops.end()) {
    op->SetType(it->second);
    VLOG(3) << "Converted " << it->first << " to " << it->second;
  } else {
    VLOG(3) << "Fused op for " << op->Type() << " is not implemented yet.";
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
