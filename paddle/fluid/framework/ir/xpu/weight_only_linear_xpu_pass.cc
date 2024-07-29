// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "glog/logging.h"

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/xpu/pass_utils.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

namespace patterns {

struct PermuteINT8WeightOnlyPattern : public PatternBase {
  PermuteINT8WeightOnlyPattern(PDPattern* pattern,
                               const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(weight_only_linear);

  // declare variable node's name
  PATTERN_DECL_NODE(input);
  PATTERN_DECL_NODE(weight);
  PATTERN_DECL_NODE(weight_scale);
  PATTERN_DECL_NODE(out);
};

PermuteINT8WeightOnlyPattern::PermuteINT8WeightOnlyPattern(
    PDPattern* pattern, const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* input = pattern->NewNode(input_repr())
                    ->assert_is_op_input("weight_only_linear_xpu", "x")
                    ->AsInput();
  auto* weight = pattern->NewNode(weight_repr())
                     ->assert_is_op_input("weight_only_linear_xpu", "weight")
                     ->AsInput();
  auto* weight_scale =
      pattern->NewNode(weight_scale_repr())
          ->assert_is_op_input("weight_only_linear_xpu", "weight_scale")
          ->AsInput();
  auto* out = pattern->NewNode(out_repr())
                  ->assert_is_op_output("weight_only_linear_xpu", "out")
                  ->AsOutput();
  auto* weight_only_linear = pattern->NewNode(weight_only_linear_repr())
                                 ->assert_is_op("weight_only_linear_xpu");

  std::vector<PDNode*> input_vars{input, weight, weight_scale};
  std::vector<PDNode*> output_vars{out};
  weight_only_linear->LinksFrom(input_vars).LinksTo(output_vars);
}

}  // namespace patterns

class PermuteINT8WeightOnlyPass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  void ApplyPermuteINT8WeightOnly(ir::Graph* graph) const;
  const std::string name_scope_{"weight_only_linear_xpu_pass"};
};

void PermuteINT8WeightOnlyPass::ApplyPermuteINT8WeightOnly(
    ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::PermuteINT8WeightOnlyPattern pattern(gpd.mutable_pattern(),
                                                 name_scope_);
  int found_subgraph_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle PermuteINT8WeightOnlyPass";

    // declare operator node's name
    GET_IR_NODE(weight_only_linear);

    // declare variable node's name
    GET_IR_NODE(input);
    GET_IR_NODE(weight);
    GET_IR_NODE(weight_scale);
    GET_IR_NODE(out);

    auto* block = weight_only_linear->Op()->Block();
    auto* scope = param_scope();
    auto* cpu_ctx = static_cast<phi::CPUContext*>(
        phi::DeviceContextPool::Instance().Get(phi::CPUPlace()));

    auto permute_weight = [&](const std::string& input_name,
                              const std::string& scale_name,
                              std::vector<Node*>* weight_node,
                              std::vector<std::string>* weight_name,
                              std::vector<Node*>* old_weight_node,
                              std::vector<std::string>* old_weight_name) {
      auto GetPrefixWithoutHash = [](const std::string& name) -> std::string {
        std::size_t found = name.find("_#");
        return found == std::string::npos ? name : name.substr(0, found);
      };

      auto input_names = weight_only_linear->Op()->Input(input_name);
      auto scale_names = weight_only_linear->Op()->Input(scale_name);
      int id = 0;
      for (auto name : input_names) {
        phi::DenseTensor* scale_tensor =
            scope->Var(scale_names[id])->GetMutable<phi::DenseTensor>();
        PADDLE_ENFORCE_NOT_NULL(
            scale_tensor,
            common::errors::Fatal(
                "weight_scale tensor node should not be nullptr"));

        size_t dst_hash = HashTensor<phi::dtype::float16>(*scale_tensor);
        std::string pre_name = GetPrefixWithoutHash(name);
        std::string dst_name = pre_name + "_#" + std::to_string(dst_hash);
        auto* dst_node = FindNodeWithName(graph, dst_name);
        if (dst_node == nullptr) {
          phi::DenseTensor* curr_tensor =
              scope->Var(name)->GetMutable<phi::DenseTensor>();
          PADDLE_ENFORCE_NOT_NULL(
              curr_tensor,
              common::errors::Fatal("tensor node should not be nullptr"));
          // Create dst node
          // Update dst var_desc in block
          VarDesc dst_desc(dst_name);
          dst_desc.SetPersistable(true);
          dst_desc.SetShape(vectorize(curr_tensor->dims()));
          dst_desc.SetDataType(
              framework::TransToProtoVarType(curr_tensor->dtype()));
          Node* dst = graph->CreateVarNode(&dst_desc);
          auto* block_dst_desc = block->Var(dst_name);
          block_dst_desc->SetPersistable(dst_desc.Persistable());
          block_dst_desc->SetShape(dst_desc.GetShape());
          block_dst_desc->SetDataType(dst_desc.GetDataType());
          weight_node->push_back(dst);
          weight_name->push_back(dst_name);

          auto* src_node = FindNodeWithName(graph, name);
          old_weight_node->push_back(src_node);
          old_weight_name->push_back(name);
          auto* dst_var = scope->FindVar(dst_name);
          if (dst_var == nullptr) {
            phi::DenseTensor tmp_tensor;
            tmp_tensor.set_type(phi::DataType::INT8);
            tmp_tensor.Resize(curr_tensor->dims());
            cpu_ctx->Alloc<int8_t>(&tmp_tensor);

            int k = curr_tensor->dims()[1];
            for (int i = 0; i < curr_tensor->numel(); i += 8 * 16) {
              int read_j_len = ((curr_tensor->numel() - i + 15) / 16) > 8
                                   ? 8
                                   : ((curr_tensor->numel() - i + 15) / 16);
              for (int j = 0; j < read_j_len; ++j) {
                int permuted_w_offset = i / (2 * k) * 2 * k +
                                        ((j > 3) ? k : 0) + (i % (2 * k)) / 2 +
                                        16 * (j % 4);
                int w_offset = i + j * 16;
                int read_l_len = (curr_tensor->numel() - i - j * 16) > 16
                                     ? 16
                                     : (curr_tensor->numel() - i - j * 16);
                for (int l = 0; l < read_l_len; l++) {
                  // {0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15}
                  int offset_l = (l / 8) % 2 ? (l - 7 + (l % 8)) : l + (l % 8);
                  tmp_tensor.data<int8_t>()[permuted_w_offset + l] =
                      static_cast<int8_t>(
                          static_cast<int32_t>(reinterpret_cast<uint8_t*>(
                              curr_tensor
                                  ->data<int8_t>())[w_offset + offset_l]) -
                          128);
                }
              }
            }
            Assign(tmp_tensor,
                   scope->Var(dst_name)->GetMutable<phi::DenseTensor>());
          }
        }
        id++;
      }
    };
    std::vector<Node*> weight_node;
    std::vector<std::string> weight_name;
    std::vector<Node*> old_weight_node;
    std::vector<std::string> old_weight_name;
    permute_weight("weight",
                   "weight_scale",
                   &weight_node,
                   &weight_name,
                   &old_weight_node,
                   &old_weight_name);

    framework::OpDesc* weight_only_linear_desc = weight_only_linear->Op();
    weight_only_linear_desc->SetInput("weight", weight_name);

    for (auto node : old_weight_node) {
      IR_NODE_UNLINK(node, weight_only_linear);
    }

    for (auto node : weight_node) {
      IR_NODE_LINK_TO(node, weight_only_linear);
    }

    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

void PermuteINT8WeightOnlyPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  ApplyPermuteINT8WeightOnly(graph);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(weight_only_linear_xpu_pass,
              paddle::framework::ir::PermuteINT8WeightOnlyPass);

REGISTER_PASS_CAPABILITY(weight_only_linear_xpu_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "weight_only_linear_xpu", 0));
