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

#include "paddle/fluid/framework/ir/trt_flash_multihead_matmul_fuse_pass.h"

#include <string>
#include "math.h"  // NOLINT

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_version_registry.h"
#ifdef PADDLE_WITH_TENSORRT
#include "paddle/fluid/inference/tensorrt/helper.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#endif
namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

//     input
//       |q     k       v
//       |------|-------|
//    matmul  matmul  matmul
//       |      |       |
//    reshape reshape reshape
//       |      |       |
//     trans   trans   trans
//       |(x)   |(y)    |
//        matmul        |
//          |           |
//        scale         |(y)
//          |           |
//        softmax       |
//          |------matmul
//            (x)     |
//                  trans
//                    |
//                  reshape
//                    |
//                   output
//
// -> fused to
//
//     input
//       |
//    flash_multihead_matmul
//       |
//     output

PDNode* TrtFlashMultiHeadMatmulPattern::operator()() {
  std::unordered_set<std::string> mul_ops{"matrix_multiply"};
  std::unordered_set<std::string> matmul_ops{"matrix_multiply"};
  auto* input0 = pattern->NewNode(input0_repr());
  input0->assert_is_ops_input(mul_ops);
  VLOG(5) << "Start match TrtFlashMultiHeadMatmulPattern";

  // First path
  auto* mul0 = pattern->NewNode(mul0_repr())->assert_is_ops(mul_ops);
  auto* mul0_w_var = pattern->NewNode(mul0_w_repr())
                         ->AsInput()
                         ->assert_is_ops_input(mul_ops, "Y");
  auto* mul0_out_var =
      pattern->NewNode(mul0_out_repr())->assert_is_ops_output(mul_ops);

  mul0_out_var->AsIntermediate()->assert_is_op_input("reshape2");

  auto* reshape2_0 =
      pattern->NewNode(reshape2_0_repr())->assert_is_op("reshape2");

  auto* reshape2_0_out_var =
      pattern->NewNode(reshape2_0_out_repr())->assert_is_op_output("reshape2");
  reshape2_0_out_var->AsIntermediate()->assert_is_op_input("transpose2");

  auto* transpose2_0 =
      pattern->NewNode(transpose2_0_repr())->assert_is_op("transpose2");
  auto* transpose2_0_out_var = pattern->NewNode(transpose2_0_out_repr())
                                   ->assert_is_op_output("transpose2");
  transpose2_0_out_var->AsIntermediate()->assert_is_ops_input(matmul_ops, "X");

  auto* matmul_qk =
      pattern->NewNode(matmul_qk_repr())->assert_is_ops(matmul_ops);
  auto* matmul_qk_out_var =
      pattern->NewNode(matmul_qk_out_repr())->assert_is_ops_output(matmul_ops);
  matmul_qk_out_var->AsIntermediate()->assert_is_op_input("scale");

  auto* scale = pattern->NewNode(scale_repr())->assert_is_op("scale");
  auto* scale_out_var =
      pattern->NewNode(scale_out_repr())->assert_is_op_output("scale");
  scale_out_var->AsIntermediate()->assert_is_op_input("softmax");

  auto* softmax_qk =
      pattern->NewNode(softmax_qk_repr())->assert_is_op("softmax");
  auto* softmax_qk_out_var =
      pattern->NewNode(softmax_qk_out_repr())->assert_is_op_output("softmax");
  softmax_qk_out_var->AsIntermediate()->assert_is_ops_input(matmul_ops);

  auto* matmul_qkv =
      pattern->NewNode(matmul_qkv_repr())->assert_is_ops(matmul_ops);
  auto* matmul_qkv_out_var =
      pattern->NewNode(matmul_qkv_out_repr())->assert_is_ops_output(matmul_ops);
  matmul_qkv_out_var->AsIntermediate()->assert_is_op_input("transpose2");

  auto* transpose2_qkv =
      pattern->NewNode(transpose2_qkv_repr())->assert_is_op("transpose2");
  auto* transpose2_qkv_out_var = pattern->NewNode(transpose2_qkv_out_repr())
                                     ->assert_is_op_output("transpose2");
  transpose2_qkv_out_var->AsIntermediate()->assert_is_op_input("reshape2");

  auto* reshape2_qkv =
      pattern->NewNode(reshape2_qkv_repr())->assert_is_op("reshape2");
  auto* reshape2_qkv_out_var = pattern->NewNode(reshape2_qkv_out_repr())
                                   ->assert_is_op_output("reshape2");

  // Second path to matmul
  auto* mul1 = pattern->NewNode(mul1_repr())->assert_is_ops(mul_ops);
  auto* mul1_w_var = pattern->NewNode(mul1_w_repr())
                         ->AsInput()
                         ->assert_is_ops_input(mul_ops, "Y");
  auto* mul1_out_var =
      pattern->NewNode(mul1_out_repr())->assert_is_ops_output(mul_ops);

  mul1_out_var->AsIntermediate()->assert_is_op_input("reshape2");

  auto* reshape2_1 =
      pattern->NewNode(reshape2_1_repr())->assert_is_op("reshape2");

  auto* reshape2_1_out_var =
      pattern->NewNode(reshape2_1_out_repr())->assert_is_op_output("reshape2");
  reshape2_1_out_var->AsIntermediate()->assert_is_op_input("transpose2");

  auto* transpose2_1 =
      pattern->NewNode(transpose2_1_repr())->assert_is_op("transpose2");
  auto* transpose2_1_out_var = pattern->NewNode(transpose2_1_out_repr())
                                   ->assert_is_op_output("transpose2");
  transpose2_1_out_var->AsIntermediate()->assert_is_ops_input(
      matmul_ops, "Y");  // link to matmul qk

  // Third path to matmul
  auto* mul2 = pattern->NewNode(mul2_repr())->assert_is_ops(mul_ops);
  auto* mul2_w_var = pattern->NewNode(mul2_w_repr())
                         ->AsInput()
                         ->assert_is_ops_input(mul_ops, "Y");
  auto* mul2_out_var =
      pattern->NewNode(mul2_out_repr())->assert_is_ops_output(mul_ops);

  mul2_out_var->AsIntermediate()->assert_is_op_input("reshape2");

  auto* reshape2_2 =
      pattern->NewNode(reshape2_2_repr())->assert_is_op("reshape2");

  auto* reshape2_2_out_var =
      pattern->NewNode(reshape2_2_out_repr())->assert_is_op_output("reshape2");
  reshape2_2_out_var->AsIntermediate()->assert_is_op_input("transpose2");

  auto* transpose2_2 =
      pattern->NewNode(transpose2_2_repr())->assert_is_op("transpose2");
  auto* transpose2_2_out_var = pattern->NewNode(transpose2_2_out_repr())
                                   ->assert_is_op_output("transpose2");
  transpose2_2_out_var->AsIntermediate()->assert_is_ops_input(
      matmul_ops);  // link to matmul qkv

  // Q path
  mul0->LinksFrom({input0, mul0_w_var}).LinksTo({mul0_out_var});
  reshape2_0->LinksFrom({mul0_out_var}).LinksTo({reshape2_0_out_var});
  transpose2_0->LinksFrom({reshape2_0_out_var}).LinksTo({transpose2_0_out_var});
  // K path
  mul1->LinksFrom({input0, mul1_w_var}).LinksTo({mul1_out_var});

  reshape2_1->LinksFrom({mul1_out_var}).LinksTo({reshape2_1_out_var});
  transpose2_1->LinksFrom({reshape2_1_out_var}).LinksTo({transpose2_1_out_var});
  // compute q*k
  matmul_qk->LinksFrom({transpose2_0_out_var, transpose2_1_out_var})
      .LinksTo({matmul_qk_out_var});
  scale->LinksFrom({matmul_qk_out_var}).LinksTo({scale_out_var});
  softmax_qk->LinksFrom({scale_out_var}).LinksTo({softmax_qk_out_var});
  // V  path
  mul2->LinksFrom({input0, mul2_w_var}).LinksTo({mul2_out_var});

  reshape2_2->LinksFrom({mul2_out_var}).LinksTo({reshape2_2_out_var});
  transpose2_2->LinksFrom({reshape2_2_out_var}).LinksTo({transpose2_2_out_var});
  // compute q*k*v
  matmul_qkv->LinksFrom({softmax_qk_out_var, transpose2_2_out_var})
      .LinksTo({matmul_qkv_out_var});
  transpose2_qkv->LinksFrom({matmul_qkv_out_var})
      .LinksTo({transpose2_qkv_out_var});
  reshape2_qkv->LinksFrom({transpose2_qkv_out_var})
      .LinksTo({reshape2_qkv_out_var});

  return reshape2_qkv_out_var;
}

}  // namespace patterns

TrtFlashMultiHeadMatmulFusePass::TrtFlashMultiHeadMatmulFusePass() {
  AddOpCompat(OpCompat("reshape2"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Shape")
      .IsTensor()
      .IsOptional()
      .End()
      .AddInput("ShapeTensor")
      .IsTensor()
      .IsOptional()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddOutput("XShape")
      .IsOptional()
      .IsTensor()
      .End()
      .AddAttr("shape")  // -->(B, S, H, N)  <--(B, S, N*H)
      .IsType<std::vector<int>>()
      .End();

  // -->: (B, S, H, N) -> (B, H, S, N)
  // <--: (B, H, S, N) -> (B, S, H, N)
  AddOpCompat(OpCompat("transpose2"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddOutput("XShape")
      .IsOptional()
      .IsTensor()
      .End()
      .AddAttr("axis")  // {0, 2, 1, 3}
      .IsType<std::vector<int>>()
      .End();

  // QK (B, H, S, N)*(B, H, S, N) -> (B, H, S, S)
  // QKV (B, H, S, S)*(B, H, S, N) -> (B, H, S, N)
  AddOpCompat(OpCompat("softmax"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .IsIntIn({-1, 3})  // shape is (B, H, S, S), so axis is -1 or 3
      .End();
}

int TrtFlashMultiHeadMatmulFusePass::BuildFlashFusion(
    Graph* graph, const std::string& name_scope, Scope* scope) const {
  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();
  bool use_trt_fma = false;

#ifdef PADDLE_WITH_TENSORRT
  int sm = platform::GetGPUComputeCapability(platform::GetCurrentDeviceId());
  use_trt_fma = sm >= 80 ? true : false;
#endif
  // Lora's attention weight cannot be manipulated during pass processing
  bool weight_is_constant = false;

  // Create pattern.
  patterns::TrtFlashMultiHeadMatmulPattern multihead_pattern(pattern,
                                                             name_scope);

  multihead_pattern();
  auto fuse_creater = [&](Node* input0,
                          Node* mul0,
                          Node* mul1,
                          Node* mul2,
                          Node* mul0_out,
                          Node* mul1_out,
                          Node* mul2_out,
                          Node* mul0_w,
                          Node* mul1_w,
                          Node* mul2_w,
                          Node* reshape2,
                          Node* reshape2_qkv_out,
                          Node* scale,
                          Node* scale_out) {
    // get Device context
    auto* dev_ctx = static_cast<phi::CPUContext*>(
        platform::DeviceContextPool::Instance().Get(platform::CPUPlace()));

    auto scale_attr = PADDLE_GET_CONST(float, scale->Op()->GetAttr("scale"));

    // create multihead
    OpDesc multihead_op_desc(mul0->Op()->Block());
    auto reshape_desc = reshape2->Op();
    int head_number =
        PADDLE_GET_CONST(std::vector<int>, reshape_desc->GetAttr("shape"))
            .at(2);
    int head_size =
        PADDLE_GET_CONST(std::vector<int>, reshape_desc->GetAttr("shape"))
            .at(3);
    multihead_op_desc.SetType("flash_multihead_matmul");
    multihead_op_desc.SetInput("Input", {input0->Name()});
    if (mul0_w->Var()->Persistable() && mul1_w->Var()->Persistable() &&
        mul2_w->Var()->Persistable()) {
      weight_is_constant = true;
    }
    // check the scale
    int hidden_out = head_number * head_size;
    if (abs(scale_attr - 1.0f / sqrt(static_cast<float>(head_size))) > 1e-5) {
      VLOG(3) << "scale of multi-head matmul do not fit the requirement of "
                 "flash attention plugin, Stop fusing.";
      return;
    }
    if (use_trt_fma && weight_is_constant) {
      auto* wq_tensor =
          scope->FindVar(mul0_w->Name())->GetMutable<phi::DenseTensor>();
      auto* wk_tensor =
          scope->FindVar(mul1_w->Name())->GetMutable<phi::DenseTensor>();
      auto* wv_tensor =
          scope->FindVar(mul2_w->Name())->GetMutable<phi::DenseTensor>();
      float* wq_data = wq_tensor->data<float>();
      float* wk_data = wk_tensor->data<float>();
      float* wv_data = wv_tensor->data<float>();
      // auto dims = wq_tensor->dims();
      // combined_w_dims = [in,3,out]
      auto combined_w_dims =
          common::make_ddim({wq_tensor->dims()[0], 3, wq_tensor->dims()[1]});
      auto* combined_w_desc = mul0_w->Var();
      combined_w_desc->SetShape(
          {wq_tensor->dims()[0], 3, wq_tensor->dims()[1]});
      combined_w_desc->SetPersistable(true);
      phi::DenseTensor tmp_combined_w_tensor;
      tmp_combined_w_tensor.Resize(combined_w_dims);
      float* tmp_combined_w_data =
          dev_ctx->template HostAlloc<float>(&tmp_combined_w_tensor);
      std::vector<const float*> w_vec = {wq_data, wk_data, wv_data};
      int dims_h = combined_w_dims[0], dims_w = combined_w_dims[2];
      // dims_h=in_feature, dims_w=out_feature
      // Combine the three fc weights together.
      // weight [Hidden_in * 3 * N * H]
      for (int i = 0; i < dims_h; i++) {
        for (int j = 0; j < 3; j++) {
          for (int k = 0; k < dims_w; k++) {
            int out_index = i * (3 * dims_w) + j * dims_w + k;
            int in_index = i * dims_w + k;
            tmp_combined_w_data[out_index] = w_vec[j][in_index];
          }
        }
      }
      // clear weight for reuse
      wq_tensor->clear();
      wq_tensor->Resize(combined_w_dims);
      float* new_combined_w_data = dev_ctx->template HostAlloc<float>(
          wq_tensor, sizeof(float) * wq_tensor->numel());
      memcpy(new_combined_w_data,
             tmp_combined_w_data,
             sizeof(float) * wq_tensor->numel());

      scope->EraseVars({mul1_w->Name(), mul2_w->Name()});
      multihead_op_desc.SetInput("W", {mul0_w->Name()});

    } else {
      multihead_op_desc.SetInput("weight_query", {mul0_w->Name()});
      multihead_op_desc.SetInput("weight_key", {mul1_w->Name()});
      multihead_op_desc.SetInput("weight_value", {mul2_w->Name()});
    }
    multihead_op_desc.SetAttr("scale", scale_attr);
    multihead_op_desc.SetAttr("hidden_out", hidden_out);
    multihead_op_desc.SetAttr("head_number", head_number);
    multihead_op_desc.SetOutput("Out", {reshape2_qkv_out->Name()});
    multihead_op_desc.SetAttr("use_trt_fma", use_trt_fma);
    multihead_op_desc.SetAttr("weight_is_constant", weight_is_constant);
    auto* multihead = graph->CreateOpNode(&multihead_op_desc);
    IR_NODE_LINK_TO(input0, multihead);
    IR_NODE_LINK_TO(mul0_w, multihead);
    if (!use_trt_fma || !weight_is_constant) {
      IR_NODE_LINK_TO(mul1_w, multihead);
      IR_NODE_LINK_TO(mul2_w, multihead);
    }
    IR_NODE_LINK_TO(multihead, reshape2_qkv_out);
  };
  int fusion_count{0};
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(input0, input0, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(mul0, mul0, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul0_out, mul0_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul0_w, mul0_w, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_0, reshape2_0, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape2_0_out, reshape2_0_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_0, transpose2_0, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_0_out, transpose2_0_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale, scale, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale_out, scale_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(mul1, mul1, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul1_out, mul1_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul1_w, mul1_w, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_1, reshape2_1, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape2_1_out, reshape2_1_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_1, transpose2_1, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_1_out, transpose2_1_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(mul2, mul2, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul2_out, mul2_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul2_w, mul2_w, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_2, reshape2_2, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape2_2_out, reshape2_2_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_2, transpose2_2, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_2_out, transpose2_2_out, multihead_pattern);

    // nodes need be removed
    GET_IR_NODE_FROM_SUBGRAPH(matmul_qk, matmul_qk, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_qk_out, matmul_qk_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(softmax_qk, softmax_qk, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        softmax_qk_out, softmax_qk_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(matmul_qkv, matmul_qkv, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_qkv_out, matmul_qkv_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(reshape2_qkv, reshape2_qkv, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape2_qkv_out, reshape2_qkv_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_qkv, transpose2_qkv, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_qkv_out, transpose2_qkv_out, multihead_pattern);

    fuse_creater(input0,
                 mul0,
                 mul1,
                 mul2,
                 mul0_out,
                 mul1_out,
                 mul2_out,
                 mul0_w,
                 mul1_w,
                 mul2_w,
                 reshape2_0,
                 reshape2_qkv_out,
                 scale,
                 scale_out);

    std::unordered_set<const Node*> marked_nodes({reshape2_0,
                                                  reshape2_1,
                                                  reshape2_2,
                                                  reshape2_0_out,
                                                  reshape2_1_out,
                                                  reshape2_2_out,
                                                  transpose2_0,
                                                  transpose2_1,
                                                  transpose2_2,
                                                  transpose2_0_out,
                                                  transpose2_1_out,
                                                  transpose2_2_out,
                                                  matmul_qk,
                                                  matmul_qk_out,
                                                  softmax_qk,
                                                  softmax_qk_out,
                                                  transpose2_qkv,
                                                  transpose2_qkv_out,
                                                  matmul_qkv,
                                                  matmul_qkv_out,
                                                  mul0,
                                                  mul1,
                                                  mul2,
                                                  mul0_out,
                                                  mul1_out,
                                                  mul2_out,
                                                  mul1_w,
                                                  mul2_w,
                                                  reshape2_qkv,
                                                  scale});
    // Remove unneeded nodes.
    if (!use_trt_fma || !weight_is_constant) {
      marked_nodes.erase(mul1_w);
      marked_nodes.erase(mul2_w);
    }
    GraphSafeRemoveNodes(graph, marked_nodes);
    ++fusion_count;
  };
  gpd(graph, handler);
  return fusion_count;
}

void TrtFlashMultiHeadMatmulFusePass::ApplyImpl(Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);
  auto* scope = param_scope();

#ifdef PADDLE_WITH_TENSORRT
  auto trt_version = paddle::inference::tensorrt::GetTrtRuntimeVersion();
  if (std::get<0>(trt_version) * 1000 + std::get<1>(trt_version) * 100 +
          std::get<2>(trt_version) * 10 <
      8520) {
    VLOG(3) << "Flash attention oss plugin only available for trt version >= "
               "8.5.2.2. Stop this pass";
    return;
  }
#else
  // if no tensorrt, early stop
  return;
#endif

  bool with_dynamic_shape = Get<bool>("with_dynamic_shape");
  if (!with_dynamic_shape) {
    VLOG(3) << "Flash attention oss plugin need trt "
               "with_dynamic_shape. Stop this pass";
    return;
  }

  int fusion_count = BuildFlashFusion(graph, name_scope_, scope);
  AddStatis(fusion_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(trt_flash_multihead_matmul_fuse_pass,
              paddle::framework::ir::TrtFlashMultiHeadMatmulFusePass);
REGISTER_PASS_CAPABILITY(trt_flash_multihead_matmul_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("elementwise_add", 1)
            .EQ("reshape2", 0)
            .EQ("transpose2", 0)
            .EQ("scale", 0)
            .EQ("softmax", 0));
