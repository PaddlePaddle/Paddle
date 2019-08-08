// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/lite/core/mir/trans_weigths_pass.h"
#include <list>
#include <string>
#include <utility>
#include <vector>
#include "paddle/fluid/lite/core/mir/graph_visualize_pass.h"
#include "paddle/fluid/lite/core/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

void TransWeightPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  // Start from inputs of the graph, those should have place set.
  std::list<Node*> nodes;
  for (auto& node : graph->mutable_nodes()) {
    nodes.push_back(&node);
  }

  for (auto& node : nodes) {
    if (!node->IsStmt()) continue;
    auto& instruct = node->AsStmt();
    if (!instruct.op_info()->HasAttr("enable_int8")) {
      continue;
    }
    std::vector<std::string> output_arg_names =
        instruct.op_info()->output_argnames();

    CHECK(output_arg_names.size() == 1)
        << "Currently, the op that supports int8 supports only one output";
    // After static kernel select pass, there is only one kernel here.
    const Type* out_arg_ty =
        instruct.kernels()[0]->GetOutputDeclType(output_arg_names[0]);
    auto out_precision = out_arg_ty->precision();
    bool out_type_int8 = out_precision == PRECISION(kInt8) ? true : false;
    float in_scale, out_scale;

    in_scale = instruct.op_info()->GetAttr<float>("input_scale");

    // Get next input op's input_scale
    if (out_type_int8) {
      LOG(INFO) << "output_type_int8";
      auto out_node = node->outlinks.front();
      CHECK(out_node->IsArg());
      auto one_adj_op_node = out_node->outlinks.front();
      CHECK(one_adj_op_node->IsStmt());
      auto& one_adj_instruct = one_adj_op_node->AsStmt();
      CHECK(one_adj_instruct.op_info()->HasAttr("enable_int8"));
      CHECK(one_adj_instruct.op_info()->HasAttr("input_scale"));
      out_scale = one_adj_instruct.op_info()->GetAttr<float>("input_scale");
      instruct.mutable_op_info()->SetAttr("output_scale", out_scale);
    } else {
      LOG(INFO) << "output_type_fp32";
    }

    std::string op_type = instruct.op_info()->Type();
    std::vector<float> weight_scale;
    auto* scope = instruct.op()->scope();

    if (op_type == "depthwise_conv2d" || op_type == "conv2d") {
      std::string weight_var_name = instruct.op_info()->Input("Filter").front();
      auto conv_weight_t =
          scope->FindVar(weight_var_name)->GetMutable<lite::Tensor>();
      // till now, all the weight should be float32 type
      float* conv_weight_d = conv_weight_t->mutable_data<float>();
      int64_t axis_size = conv_weight_t->dims()[0];
      int64_t inner_size = conv_weight_t->data_size() / axis_size;
      weight_scale =
          GetWeightScale(conv_weight_d, axis_size, inner_size, 127.0);

      Tensor temp_tensor;
      temp_tensor.Resize(conv_weight_t->dims());
      int8_t* temp_data = temp_tensor.mutable_data<int8_t>();
      FP32ToInt8(conv_weight_d, temp_data, weight_scale.data(), axis_size, 1,
                 inner_size);
      conv_weight_t->CopyDataFrom(temp_tensor);
    } else if (op_type == "fc" || op_type == "mul") {
      std::string weight_arg_name = "W";
      if (op_type == "mul") weight_arg_name = "Y";
      std::string weight_var_name =
          instruct.op_info()->Input(weight_arg_name).front();

      auto fc_weight_t =
          scope->FindVar(weight_var_name)->GetMutable<lite::Tensor>();
      // till now, all the weight should be float32 type
      float* fc_weight_d = fc_weight_t->mutable_data<float>();

      CHECK_EQ(fc_weight_t->dims().size(), 2UL);

      int64_t h = fc_weight_t->dims()[0];
      int64_t w = fc_weight_t->data_size() / h;
      Tensor trans_w_t, int8_temp_t;
      trans_w_t.CopyDataFrom(*fc_weight_t);
      float* trans_w_data = trans_w_t.mutable_data<float>();
      int8_temp_t.Resize(fc_weight_t->dims());
      int8_t* int8_temp_data = int8_temp_t.mutable_data<int8_t>();
      // trans weight for calc the weight scale.
      for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
          trans_w_data[i * w + j] = fc_weight_d[j * h + i];
        }
      }
      weight_scale = GetWeightScale(trans_w_data, w, h, 127.0);

      int8_t* fc_weight_int8_d = fc_weight_t->mutable_data<int8_t>();
      FP32ToInt8(trans_w_data, int8_temp_data, weight_scale.data(), w, 1, h);
      // Retrans back
      for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {
          fc_weight_int8_d[i * h + j] = int8_temp_data[j * w + i];
        }
      }
    }

    // Convert fp32 bias to int8 bias
    std::vector<std::string> input_arg_names =
        instruct.op_info()->InputArgumentNames();
    if (std::find(input_arg_names.begin(), input_arg_names.end(), "Bias") !=
            input_arg_names.end() &&
        instruct.op_info()->Input("Bias").size() > 0) {
      std::string bias_var_name = instruct.op_info()->Input("Bias").front();
      auto bias_weight_t =
          scope->FindVar(bias_var_name)->GetMutable<lite::Tensor>();
      float* bias_weight_d = bias_weight_t->mutable_data<float>();

      Tensor temp_bias;
      temp_bias.Resize(bias_weight_t->dims());
      int* temp_bias_data = temp_bias.mutable_data<int>();
      TransFP32BiasToInt32(bias_weight_d, temp_bias_data, temp_bias.data_size(),
                           in_scale, weight_scale);
      bias_weight_t->CopyDataFrom(temp_bias);
    }

    instruct.mutable_op_info()->SetAttr("weight_scale", weight_scale);

    auto original_selected_kernel = std::move(instruct.kernels().front());
    auto updated_op_info = *instruct.mutable_op_info();
    instruct.ResetOp(updated_op_info, graph->valid_places());
    instruct.kernels().clear();
    instruct.kernels().emplace_back(std::move(original_selected_kernel));
    for (auto& kernel : instruct.kernels()) {
      LOG(INFO) << "kernel info: " << kernel->name();
      instruct.op()->AttachKernel(kernel.get());
    }
  }
}

void TransWeightPass::SetValidPlaces(const std::vector<Place>& valid_places) {
  CHECK(!valid_places.empty());
  valid_places_ = valid_places;
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(trans_weight_pass, paddle::lite::mir::TransWeightPass);
