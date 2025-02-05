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

#include <string>

#include "glog/logging.h"

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/ir/xpu/pass_utils.h"
#include "paddle/fluid/framework/ir/xpu/quant_utils.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {

class BlockMultiHeadAttentionXPUPass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  void InplaceBlockMultiHeadAttentionXPU(ir::Graph* graph) const;

  const std::string name_scope_{"block_multihead_attention_xpu_pass"};
};

void BlockMultiHeadAttentionXPUPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  InplaceBlockMultiHeadAttentionXPU(graph);
}

void BlockMultiHeadAttentionXPUPass::InplaceBlockMultiHeadAttentionXPU(
    ir::Graph* graph) const {
  const int64_t max_batch_size = 10;
  auto* scope = param_scope();
  for (auto* node : graph->Nodes()) {
    if (node->IsOp() && node->Op()->Type() == "block_multihead_attention") {
      auto* op_desc = node->Op();
      op_desc->SetType("block_multihead_attention_xpu");
      phi::DenseTensor cache_k_per_batch_maxs;
      auto base_name = op_desc->Input("qkv")[0];
      int max_ptr_size = phi::backends::xpu::get_xpu_max_ptr_size(-1);
      std::string cache_k_per_batch_maxs_name = base_name + "_max_cache_k";
      VarDesc cache_k_per_batch_maxs_desc(cache_k_per_batch_maxs_name);
      cache_k_per_batch_maxs_desc.SetPersistable(true);
      cache_k_per_batch_maxs_desc.SetShape(
          {max_batch_size, static_cast<int64_t>(max_ptr_size)});
      cache_k_per_batch_maxs_desc.SetDataType(
          proto::VarType::Type::VarType_Type_FP32);
      Node* cache_k_per_batch_maxs_in =
          graph->CreateVarNode(&cache_k_per_batch_maxs_desc);
      phi::DenseTensor cpu_tensor;
      auto* cpu_ctx = static_cast<phi::CPUContext*>(
          phi::DeviceContextPool::Instance().Get(phi::CPUPlace()));
      cpu_tensor.set_type(phi::DataType::FLOAT32);
      cpu_tensor.Resize({max_batch_size, max_ptr_size});
      std::vector<float> tmp(max_batch_size * max_ptr_size, 0);
      memcpy(cpu_ctx->Alloc<float>(&cpu_tensor),
             tmp.data(),
             max_batch_size * max_ptr_size * sizeof(float));
      Assign(cpu_tensor,
             scope->Var(cache_k_per_batch_maxs_name)
                 ->GetMutable<phi::DenseTensor>());
      op_desc->SetInput("cache_k_per_batch_maxs",
                        {cache_k_per_batch_maxs_name});

      std::string cache_v_per_batch_maxs_name = base_name + "_max_cache_v";
      VarDesc cache_v_per_batch_maxs_desc(cache_v_per_batch_maxs_name);
      cache_v_per_batch_maxs_desc.SetPersistable(true);
      cache_v_per_batch_maxs_desc.SetShape(
          {max_batch_size, static_cast<int64_t>(max_ptr_size)});
      cache_v_per_batch_maxs_desc.SetDataType(
          proto::VarType::Type::VarType_Type_FP32);
      Node* cache_v_per_batch_maxs_in =
          graph->CreateVarNode(&cache_v_per_batch_maxs_desc);
      Assign(cpu_tensor,
             scope->Var(cache_v_per_batch_maxs_name)
                 ->GetMutable<phi::DenseTensor>());
      op_desc->SetInput("cache_v_per_batch_maxs",
                        {cache_v_per_batch_maxs_name});

      IR_NODE_LINK_TO(cache_k_per_batch_maxs_in, node);
      IR_NODE_LINK_TO(cache_v_per_batch_maxs_in, node);
    }
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(block_multihead_attention_xpu_pass,
              paddle::framework::ir::BlockMultiHeadAttentionXPUPass);

REGISTER_PASS_CAPABILITY(block_multihead_attention_xpu_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "block_multihead_attention_xpu", 0));
