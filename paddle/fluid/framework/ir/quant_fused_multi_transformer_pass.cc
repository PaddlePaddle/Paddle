/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/quant_fused_multi_transformer_pass.h"
#include "paddle/fluid/framework/ir/fuse_pass_base.h"

#include "glog/logging.h"

namespace paddle {
namespace framework {
namespace ir {

class Graph;

template <typename T>
inline void TransposeWeights(phi::DenseTensor* weight_tensor, int k, int n) {
  auto* dev_ctx = static_cast<phi::CPUContext*>(
      platform::DeviceContextPool::Instance().Get(platform::CPUPlace()));

  phi::DenseTensor tmp_weight_tensor;
  tmp_weight_tensor.Resize({n, k});
  dev_ctx->Alloc<T>(&tmp_weight_tensor);
  auto tmp_weight_data = tmp_weight_tensor.data<T>();
  auto weight_data = weight_tensor->data<T>();
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < n; ++j) {
      int in_idx = i * n + j;
      int out_idx = j * k + i;
      tmp_weight_data[out_idx] = weight_data[in_idx];
    }
  }
  weight_tensor->Resize({n, k});
  dev_ctx->Alloc<T>(weight_tensor);
  auto new_weight_data = weight_tensor->data<T>();
  memcpy(new_weight_data, tmp_weight_data, sizeof(T) * k * n);
}

template <typename T>
inline T roundWithTiesToEven(T x) {
  T xLower = floor(x);
  T xUpper = ceil(x);
  // x is in interval [xl,xu]. Choose closest of two bounds, breaking ties to
  // even.
  T dLower = x - xLower;
  T dUpper = xUpper - x;
  return static_cast<T>(
      (dLower == dUpper ? fmod(xLower, 2.0F) == 0.0F : dLower < dUpper)
          ? xLower
          : xUpper);
}

template <typename T>
inline int8_t quant_helper(const T input,
                           const float scale,
                           const int round_type,
                           const float max_bound,
                           const float min_bound) {
  float quant_value = max_bound * (1.0f / scale) * static_cast<float>(input);

  if (round_type == 0) {
    quant_value = static_cast<float>(roundWithTiesToEven(quant_value));
  } else {
    quant_value = static_cast<float>(round(quant_value));
  }
  quant_value = quant_value > max_bound ? max_bound : quant_value;
  quant_value = quant_value < min_bound ? min_bound : quant_value;
  return static_cast<int8_t>(quant_value);
}

template <typename T>
inline void ChannelWiseScales(
    const T* data, float* scales, int k, int n, bool is_transposed) {
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < n; ++j) {
      int index = i * n + j;
      int scale_idx = is_transposed ? i : j;
      float v = static_cast<float>(data[index]);
      scales[scale_idx] = v > scales[scale_idx] ? v : scales[scale_idx];
    }
  }
}

template <typename T>
void QuantWeight(phi::DenseTensor* weight_tensor,
                 float* scales,
                 int k,
                 int n,
                 bool is_transposed,
                 bool need_transpose) {
  auto dims = weight_tensor->dims();
  if (need_transpose) {
    PADDLE_ENFORCE(
        dims.size() == 2 && dims[0] == k && dims[1] == n,
        platform::errors::InvalidArgument(
            "QuantWeight function need 2D tensor as input, and "
            "dims[0] == k && dims[1] == n when need_transpose is True"));
  }
  phi::DenseTensor tmp;
  tmp.Resize(dims);
  auto* dev_ctx = static_cast<phi::CPUContext*>(
      platform::DeviceContextPool::Instance().Get(platform::CPUPlace()));
  dev_ctx->Alloc<T>(&tmp);

  memcpy(tmp.data<T>(),
         weight_tensor->data<T>(),
         weight_tensor->numel() * sizeof(T));
  if (need_transpose) {
    weight_tensor->Resize({{n, k}});
  }
  dev_ctx->Alloc<int8_t>(weight_tensor);

  // calcu scales
  VLOG(0) << "Begin ChannelWiseScales";
  ChannelWiseScales(tmp.data<T>(), scales, k, n, is_transposed);
  VLOG(0) << "Begin Quant";
  // quant
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < n; ++j) {
      // int index = i * n + j;
      // int out_index = need_transpose ? j * k + i : index;
      // float scale = is_transposed ? scales[i] : scales[j];
      // weight_tensor->data<int8_t>()[out_index] =
      //     quant_helper(tmp.data<T>()[index], scale, 1, 127, -127);
    }
  }
  VLOG(0) << "End Quant";

  // store
  int scales_size = is_transposed ? k : n;
  for (int i = 0; i < scales_size; ++i) {
    float& v = scales[i];
    v = v / 127.0f;
  }
}

inline Node* CreatePersistableVarNode(Graph* graph, const std::string& name) {
  auto var_desc = VarDesc(name);
  var_desc.SetDataType(framework::proto::VarType::FP32);
  var_desc.SetPersistable(true);
  auto node = graph->CreateVarNode(&var_desc);
  return node;
}

void QuantFusedMultiTrasformerPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_EQ(graph->Has(kParamScopeAttr),
                    true,
                    platform::errors::InvalidArgument(
                        "Graph must have kParamScopeAttr attribute."));

  auto& scope = graph->Get<framework::Scope>(kParamScopeAttr);

  for (const Node* n : graph->Nodes()) {
    if (n->IsOp()) {
      auto* op = n->Op();
      if (op->Type() != "fused_multi_transformer") continue;
      auto fuse_mt_node = const_cast<Node*>(n);
      VLOG(0) << "Replace fused_multi_transformer with "
                 "fused_multi_transformer_int8";

      op->SetType("fused_multi_transformer_int8");

      auto* dev_ctx = static_cast<phi::CPUContext*>(
          platform::DeviceContextPool::Instance().Get(platform::CPUPlace()));

      // Get weight tensor and quant
      std::vector<std::string> qkv_out_scale_var_names;
      for (auto weight_var_name : op->Input("QKVW")) {
        VLOG(0) << "process qkvw";
        auto* weight_tensor =
            scope.GetVar(weight_var_name)->GetMutable<phi::DenseTensor>();
        auto qkv_w_dims = weight_tensor->dims();
        int k_size = qkv_w_dims[3],
            n_size = qkv_w_dims[0] * qkv_w_dims[1] * qkv_w_dims[2];

        auto qkv_out_scale_var = scope.Var(weight_var_name + "_out_scale");
        auto* qkv_out_scale_tensor =
            qkv_out_scale_var->GetMutable<phi::DenseTensor>();
        qkv_out_scale_tensor->Resize({n_size});
        dev_ctx->Alloc<float>(qkv_out_scale_tensor);
        float* qkv_out_scale_data = qkv_out_scale_tensor->data<float>();

        switch (weight_tensor->dtype()) {
          case paddle::experimental::DataType::FLOAT16:
            QuantWeight<platform::float16>(
                weight_tensor, qkv_out_scale_data, n_size, k_size, true, false);
            break;
          case paddle::experimental::DataType::FLOAT32:
            QuantWeight<float>(
                weight_tensor, qkv_out_scale_data, n_size, k_size, true, false);
            break;
          case paddle::experimental::DataType::INT8:
            break;
          default:
            PADDLE_THROW(platform::errors::Unavailable(
                "fused_multi_transformer not supported weight dtype. "
                "we now only support fp32/fp16."));
            break;
        }
        qkv_out_scale_var_names.push_back(weight_var_name + "_out_scale");
        auto qkv_out_scale_node =
            CreatePersistableVarNode(graph, weight_var_name + "_out_scale");
        IR_NODE_LINK_TO(qkv_out_scale_node, fuse_mt_node)
      }
      op->SetInput("QKVOutScale", qkv_out_scale_var_names);

      auto weight_names =
          std::vector<std::string>{"OutLinearW", "FFN1Weight", "FFN2Weight"};
      auto out_scale_names = std::vector<std::string>{
          "OutLinearOutScale", "FFN1OutScale", "FFN2OutScale"};

      for (int i = 0; i < 3; ++i) {
        std::vector<std::string> out_scale_var_names;
        auto weights_to_quant = op->Input(weight_names[i]);
        for (auto weight_var_name : weights_to_quant) {
          VLOG(0) << "process " << weight_names[i];
          auto* weight_tensor =
              scope.GetVar(weight_var_name)->GetMutable<phi::DenseTensor>();
          auto w_dims = weight_tensor->dims();
          int k_size = w_dims[0], n_size = w_dims[1];

          auto out_scale_var = scope.Var(weight_var_name + "_out_scale");
          auto* out_scale_tensor =
              out_scale_var->GetMutable<phi::DenseTensor>();
          out_scale_tensor->Resize({n_size});
          dev_ctx->Alloc<float>(out_scale_tensor);
          float* out_scale_data = out_scale_tensor->data<float>();

          switch (weight_tensor->dtype()) {
            case paddle::experimental::DataType::FLOAT16:
              QuantWeight<platform::float16>(
                  weight_tensor, out_scale_data, k_size, n_size, false, true);
              break;
            case paddle::experimental::DataType::FLOAT32:
              QuantWeight<float>(
                  weight_tensor, out_scale_data, k_size, n_size, false, true);
              break;
            case paddle::experimental::DataType::INT8:
              break;
            default:
              PADDLE_THROW(platform::errors::Unavailable(
                  "fused_multi_transformer not supported weight dtype. "
                  "we now only support fp32/fp16."));
              break;
          }
          out_scale_var_names.push_back(weight_var_name + "_out_scale");
          auto out_scale_node =
              CreatePersistableVarNode(graph, weight_var_name + "_out_scale");
          IR_NODE_LINK_TO(out_scale_node, fuse_mt_node)
        }
        op->SetInput(out_scale_names[i], out_scale_var_names);
      }
    }
  }
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(quant_fused_multi_transformer_pass,
              paddle::framework::ir::QuantFusedMultiTrasformerPass);
