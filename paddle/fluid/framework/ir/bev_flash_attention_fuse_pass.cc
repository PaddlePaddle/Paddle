// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/bev_flash_attention_fuse_pass.h"

#include <string>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_version_registry.h"
#ifdef PADDLE_WITH_TENSORRT
#include "paddle/fluid/inference/tensorrt/helper.h"
#endif
#include "paddle/phi/kernels/funcs/blas/blas.h"

#define GET_IR_NODE(node__) GET_IR_NODE_FROM_SUBGRAPH(node__, node__, pattern);
#define GET_NODES                  \
  GET_IR_NODE(elementwiseA0_op);   \
  GET_IR_NODE(elementwiseA0_in_y); \
  GET_IR_NODE(elementwiseA0_out);  \
  GET_IR_NODE(matmulA0_op);        \
  GET_IR_NODE(matmulA0_in_y);      \
  GET_IR_NODE(matmulA0_out);       \
  GET_IR_NODE(elementwiseA1_op);   \
  GET_IR_NODE(elementwiseA1_in_y); \
  GET_IR_NODE(elementwiseA1_out);  \
  GET_IR_NODE(reshapeA1_op);       \
  GET_IR_NODE(reshapeA1_out);      \
  GET_IR_NODE(transposeA1_op);     \
  GET_IR_NODE(transposeA1_out);    \
  GET_IR_NODE(scaleA1_op);         \
  GET_IR_NODE(scaleA1_out);        \
  GET_IR_NODE(matmulA1_op);        \
  GET_IR_NODE(matmulA1_out);       \
  GET_IR_NODE(softmaxA1_op);       \
  GET_IR_NODE(softmaxA1_out);      \
  GET_IR_NODE(matmulA2_op);        \
  GET_IR_NODE(matmulA2_out);       \
  GET_IR_NODE(transposeA2_op);     \
  GET_IR_NODE(transposeA2_out);    \
  GET_IR_NODE(reshapeA2_op);       \
  GET_IR_NODE(reshapeA2_out);      \
  GET_IR_NODE(elementwiseB0_op);   \
  GET_IR_NODE(elementwiseB0_out);  \
  GET_IR_NODE(matmulB0_op);        \
  GET_IR_NODE(matmulB0_in_y);      \
  GET_IR_NODE(matmulB0_out);       \
  GET_IR_NODE(elementwiseB1_op);   \
  GET_IR_NODE(elementwiseB1_in_y); \
  GET_IR_NODE(elementwiseB1_out);  \
  GET_IR_NODE(reshapeB1_op);       \
  GET_IR_NODE(reshapeB1_out);      \
  GET_IR_NODE(transposeB1_op);     \
  GET_IR_NODE(transposeB1_out);    \
  GET_IR_NODE(matmulC0_op);        \
  GET_IR_NODE(matmulC0_in_y);      \
  GET_IR_NODE(matmulC0_out);       \
  GET_IR_NODE(elementwiseC1_op);   \
  GET_IR_NODE(elementwiseC1_in_y); \
  GET_IR_NODE(elementwiseC1_out);  \
  GET_IR_NODE(reshapeC1_op);       \
  GET_IR_NODE(reshapeC1_out);      \
  GET_IR_NODE(transposeC1_op);     \
  GET_IR_NODE(transposeC1_out);

// fuse struct
//            input
//              |
//       |      |       |
//     element element  |
//       |      |       |
//       |q     k       v
//       |      |       |
//    matmul  matmul  matmul
//       |      |       |
//     element element element
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

namespace paddle {
namespace framework {
namespace ir {
namespace {

bool IsScale(OpDesc* const op_ptr,
             std::string* name,
             std::string regexp = "Input_scale_") {
  name->clear();
  std::unordered_map<std::string, Attribute> attr_map = op_ptr->GetAttrMap();
  std::unordered_map<std::string, Attribute>::iterator iter;
  int len = regexp.size();
  for (iter = attr_map.begin(); iter != attr_map.end(); iter++) {
    if (regexp == iter->first.substr(0, len)) {
      *name = iter->first;
      return true;
    }
  }
  return false;
}

// Naive gemm.
// A=[M,K], B=[K,N], C=[M, N]
// C = A * B + C
template <typename T>
void NaiveGemm(const T* A, const T* B, T* C, int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < K; k++) {
        C[i * N + j] = (C[i * N + j]) + A[i * K + k] * B[k * N + j];
      }
    }
  }
}

// B need to broadcast.
// A=[M,N], B=[N]
// A = A + B
template <typename T>
void ElementAdd(T* A, T* B, int M, int N) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      int out_index = N * i + j;
      int in_index = j;
      A[out_index] = A[out_index] + B[in_index];
    }
  }
}
}  // namespace

template <typename T>
inline void QKVWeightsProcess(phi::DenseTensor* wq_tensor,
                              phi::DenseTensor* wk_tensor,
                              phi::DenseTensor* wv_tensor,
                              phi::DenseTensor* bq_tensor,
                              phi::DenseTensor* bk_tensor,
                              phi::DenseTensor* bv_tensor,
                              phi::DenseTensor* bqk_tensor_before) {
  auto* wq_data = wq_tensor->data<T>();
  auto* wk_data = wk_tensor->data<T>();
  auto* wv_data = wv_tensor->data<T>();
  auto* bq_data = bq_tensor->data<T>();
  auto* bk_data = bk_tensor->data<T>();
  auto* bv_data = bv_tensor->data<T>();
  auto* bqk_data_before = bqk_tensor_before->data<T>();
  auto* dev_ctx = static_cast<phi::CPUContext*>(
      platform::DeviceContextPool::Instance().Get(platform::CPUPlace()));

  auto combined_w_dims =
      phi::make_ddim({wq_tensor->dims()[0], 3, wq_tensor->dims()[1]});
  auto combined_bias_dims =
      phi::make_ddim({bqk_tensor_before->dims()[1], 3, bq_tensor->dims()[0]});

  /*
  //merge elementwise+matmul+element

       |                    |
  elementwise_add     elementwise_add
       |bias=[900,256]      |
       |                    |
     matmul               matmul
       |                    |
       |                    |
  elementwise          elementwise
       |bias=[256]          |
         \                /
                matmul

                  \/
                  \/

       |                    |
     matmul               matmul
       |                    |
       |                    |
  elementwise          elementwise
       |bias=[900,256]      |
         \                /
                matmul
 */
  auto tmp_bias_dims =
      phi::make_ddim({bqk_tensor_before->dims()[1], bq_tensor->dims()[0]});
  int M = bqk_tensor_before->dims()[1];
  int K = bqk_tensor_before->dims()[2];
  int N = bq_tensor->dims()[0];
  // tmp_q

  phi::DenseTensor tmp_bias_q_tensor;
  tmp_bias_q_tensor.Resize(tmp_bias_dims);
  auto* tmp_bias_q_data = dev_ctx->template HostAlloc<T>(&tmp_bias_q_tensor);
  memset(tmp_bias_q_data, 0, M * N * sizeof(T));
  NaiveGemm(bqk_data_before, wq_data, tmp_bias_q_data, M, N, K);
  ElementAdd(tmp_bias_q_data, bq_data, M, N);
  // tmp_k
  phi::DenseTensor tmp_bias_k_tensor;
  tmp_bias_k_tensor.Resize(tmp_bias_dims);
  auto* tmp_bias_k_data = dev_ctx->template HostAlloc<T>(&tmp_bias_k_tensor);
  memset(tmp_bias_k_data, 0, M * N * sizeof(T));
  NaiveGemm(bqk_data_before, wk_data, tmp_bias_k_data, M, N, K);
  ElementAdd(tmp_bias_k_data, bk_data, M, N);
  // tmp_v
  phi::DenseTensor tmp_bias_v_tensor;
  tmp_bias_v_tensor.Resize(tmp_bias_dims);
  auto* tmp_bias_v_data = dev_ctx->template HostAlloc<T>(&tmp_bias_v_tensor);
  memset(tmp_bias_v_data, 0, M * N * sizeof(T));
  ElementAdd(tmp_bias_v_data, bv_data, M, N);

  // combine bias_qkv
  phi::DenseTensor tmp_combined_bias_tensor;
  tmp_combined_bias_tensor.Resize(combined_bias_dims);
  auto* tmp_combined_bias_data =
      dev_ctx->template HostAlloc<T>(&tmp_combined_bias_tensor);

  std::vector<T*> bias_vec = {
      tmp_bias_q_data, tmp_bias_k_data, tmp_bias_v_data};
  int dims_row = combined_bias_dims[0], dims_col = combined_bias_dims[2];
  // Combine the three bias together.
  for (int i = 0; i < dims_row; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < dims_col; k++) {
        int out_index = i * (3 * dims_col) + j * dims_col + k;
        int in_index = i * dims_col + k;
        tmp_combined_bias_data[out_index] = bias_vec[j][in_index];
      }
    }
  }

  bq_tensor->Resize(combined_bias_dims);
  auto* new_combined_bias_data = dev_ctx->template HostAlloc<T>(bq_tensor);
  memcpy(new_combined_bias_data,
         tmp_combined_bias_data,
         sizeof(T) * bq_tensor->numel());

  phi::DenseTensor tmp_combined_w_tensor;
  tmp_combined_w_tensor.Resize(combined_w_dims);
  auto* tmp_combined_w_data =
      dev_ctx->template HostAlloc<T>(&tmp_combined_w_tensor);

  std::vector<T*> w_vec = {wq_data, wk_data, wv_data};
  int dims_h = combined_w_dims[0], dims_w = combined_w_dims[2];
  // Combine the three fc weights together.
  for (int i = 0; i < dims_h; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < dims_w; k++) {
        int out_index = i * (3 * dims_w) + j * dims_w + k;
        int in_index = i * dims_w + k;
        tmp_combined_w_data[out_index] = w_vec[j][in_index];
      }
    }
  }

  wq_tensor->clear();
  wq_tensor->Resize(combined_w_dims);
  auto* new_combined_w_data = dev_ctx->template HostAlloc<T>(wq_tensor);
  memcpy(
      new_combined_w_data, tmp_combined_w_data, sizeof(T) * wq_tensor->numel());
}

void BevFlashAttentionFusePass::ApplyImpl(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  const std::string pattern_name = "bev_flash_attention_fuse";
  FusePassBase::Init(pattern_name, graph);
  auto* scope = param_scope();

#ifdef PADDLE_WITH_TENSORRT
  auto trt_version = paddle::inference::tensorrt::GetTrtRuntimeVersion();
  if (std::get<0>(trt_version) * 1000 + std::get<1>(trt_version) * 100 +
          std::get<2>(trt_version) * 10 <
      8520) {
    VLOG(3)
        << "BevFlash attention oss plugin only available for trt version >= "
           "8.5.2.2. Stop this pass";
    return;
  }
#else
  return;
#endif

  // pattern
  std::unordered_set<std::string> matmul_ops{"matmul", "matmul_v2"};
  PDNode* x = gpd.mutable_pattern()
                  ->NewNode("x")
                  ->assert_is_ops_input(matmul_ops, "X")
                  ->AsInput();
  patterns::BevFlashAttention pattern(gpd.mutable_pattern(), pattern_name);
  pattern(x);

  int fusion_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_NODES;
    // new desc;
    OpDesc desc(matmulA0_op->Op()->Block());
    desc.SetType("flash_multihead_matmul");
    desc.SetInput("Input", {subgraph.at(x)->Name()});
    // refactor W and Bias
    auto* wq_tensor =
        scope->FindVar(matmulA0_in_y->Name())->GetMutable<phi::DenseTensor>();
    auto* wk_tensor =
        scope->FindVar(matmulB0_in_y->Name())->GetMutable<phi::DenseTensor>();
    auto* wv_tensor =
        scope->FindVar(matmulC0_in_y->Name())->GetMutable<phi::DenseTensor>();

    auto* bq_tensor = scope->FindVar(elementwiseA1_in_y->Name())
                          ->GetMutable<phi::DenseTensor>();
    auto* bk_tensor = scope->FindVar(elementwiseB1_in_y->Name())
                          ->GetMutable<phi::DenseTensor>();
    auto* bv_tensor = scope->FindVar(elementwiseC1_in_y->Name())
                          ->GetMutable<phi::DenseTensor>();
    auto* bqk_tensor_before = scope->FindVar(elementwiseA0_in_y->Name())
                                  ->GetMutable<phi::DenseTensor>();

    if (wq_tensor->dtype() == phi::DataType::FLOAT32) {
      QKVWeightsProcess<float>(wq_tensor,
                               wk_tensor,
                               wv_tensor,
                               bq_tensor,
                               bk_tensor,
                               bv_tensor,
                               bqk_tensor_before);
    } else if (wq_tensor->dtype() == phi::DataType::FLOAT16) {
      QKVWeightsProcess<platform::float16>(wq_tensor,
                                           wk_tensor,
                                           wv_tensor,
                                           bq_tensor,
                                           bk_tensor,
                                           bv_tensor,
                                           bqk_tensor_before);
    } else {
      PADDLE_THROW(platform::errors::Unavailable(
          "multihead_matmul not supported weight dtype. we now only support "
          "fp32 and fp16."));
    }

    // reuse the mul0_w and eltadd_0_b nodes for the combined nodes.
    auto* combined_w_desc = matmulA0_in_y->Var();
    combined_w_desc->SetShape({wq_tensor->dims()[0], 3, wq_tensor->dims()[2]});
    combined_w_desc->SetPersistable(true);

    auto* combined_bias_desc = elementwiseA1_in_y->Var();
    combined_bias_desc->SetShape(
        {bqk_tensor_before->dims()[1], 3, bq_tensor->dims()[1]});
    combined_bias_desc->SetPersistable(true);

    scope->EraseVars({matmulB0_in_y->Name(), matmulC0_in_y->Name()});
    scope->EraseVars({elementwiseB1_in_y->Name(), elementwiseC1_in_y->Name()});
    paddle::memory::Release(platform::CPUPlace());

    desc.SetInput("W", {matmulA0_in_y->Name()});
    desc.SetInput("Bias", {elementwiseA1_in_y->Name()});
    std::vector<int64_t> shape = softmaxA1_out->Var()->GetShape();
    desc.SetOutput("Out", {reshapeA2_out->Name()});
    desc.SetAttr("head_number", static_cast<int>(shape[1]));
    float alpha = PADDLE_GET_CONST(float, scaleA1_op->Op()->GetAttr("scale"));
    desc.SetAttr("alpha", alpha);

    // int8 for fc
    std::string scale_name;
    if (IsScale(matmulA0_op->Op(), &scale_name)) {
      desc.SetAttr("Input_scale", matmulA0_op->Op()->GetAttr(scale_name));
    }
    if (IsScale(elementwiseA0_op->Op(), &scale_name, "Out")) {
      desc.SetAttr("fc_out_threshold",
                   elementwiseA0_op->Op()->GetAttr(scale_name));
    }

    // Create a new node for the fused op.
    auto flash_attention_node = graph->CreateOpNode(&desc);

    // Link inputs and outputs.
    PADDLE_ENFORCE_NE(
        subgraph.count(x),
        0,
        platform::errors::NotFound("Detector did not find input x of conv2d."));

    IR_NODE_LINK_TO(subgraph.at(x), flash_attention_node);  // Input
    IR_NODE_LINK_TO(matmulA0_in_y, flash_attention_node);
    IR_NODE_LINK_TO(elementwiseA1_in_y, flash_attention_node);
    IR_NODE_LINK_TO(flash_attention_node, reshapeA2_out);  // Output

    // Delete the unneeded nodes.
    std::unordered_set<const Node*> marked_nodes(
        {elementwiseA0_op, elementwiseA0_out, matmulA0_op,
         matmulA0_out,     elementwiseA1_op,  elementwiseA1_out,
         reshapeA1_op,     reshapeA1_out,     transposeA1_op,
         transposeA1_out,  scaleA1_op,        scaleA1_out,
         matmulA1_op,      matmulA1_out,      softmaxA1_op,
         softmaxA1_out,    matmulA2_op,       matmulA2_out,
         transposeA2_op,   transposeA2_out,   reshapeA2_op,
         elementwiseB0_op, elementwiseB0_out, matmulB0_op,
         matmulB0_out,     elementwiseB1_op,  elementwiseB1_out,
         reshapeB1_op,     reshapeB1_out,     transposeB1_op,
         transposeB1_out,  matmulC0_op,       matmulC0_out,
         elementwiseC1_op, elementwiseC1_out, reshapeC1_op,
         reshapeC1_out,    transposeC1_op,    transposeC1_out});

    GraphSafeRemoveNodes(graph, marked_nodes);
    ++fusion_count;
  };
  gpd(graph, handler);
  AddStatis(fusion_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(bev_flash_attention_fuse_pass,
              paddle::framework::ir::BevFlashAttentionFusePass);
REGISTER_PASS_CAPABILITY(bev_flash_attention_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("reshape2", 0)
            .EQ("transpose2", 0)
            .EQ("slice", 0)
            .EQ("scale", 0)
            .EQ("softmax", 0)
            .EQ("matmul_v2", 0));
