/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <fstream>
#include <iostream>

#include "paddle/fluid/inference/api/paddle_analysis_config.h"
#include "test/cpp/inference/api/tester_helper.h"

#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/program_desc.h"
#include <unordered_map>
#include <glog/logging.h>

using namespace std;

DEFINE_bool(enable_mkldnn, true, "Enable MKLDNN");

namespace paddle {
namespace inference {
namespace analysis {

static float const SCALE = 2.f;
const std::vector<std::string> PreGraphPasses({
   "simplify_with_basic_ops_pass",
   "quant_dequant_mkldnn_pass",
//    "mkldnn_placement_pass",
//    "constant_folding_pass",
   "squeeze2_transpose2_onednn_fuse_pass",
   "layer_norm_fuse_pass",
   "attention_lstm_fuse_pass",
   "seqconv_eltadd_relu_fuse_pass",
   "fc_lstm_fuse_pass",
   "mul_lstm_fuse_pass",
//    "fc_gru_fuse_pass",
//    "mul_gru_fuse_pass",
//    "multi_gru_fuse_pass",
//    "multi_gru_seq_fuse_pass",
//    "seq_concat_fc_fuse_pass",
   "gpu_cpu_squeeze2_matmul_fuse_pass",
   "gpu_cpu_reshape2_matmul_fuse_pass",
   "gpu_cpu_flatten2_matmul_fuse_pass",
   "matmul_v2_scale_fuse_pass",
   "squared_mat_sub_fuse_pass",
   "is_test_pass",
   "gpu_cpu_map_matmul_v2_to_mul_pass",
   "gpu_cpu_map_matmul_v2_to_matmul_pass",
   "matmul_scale_fuse_pass",
   "gpu_cpu_map_matmul_to_mul_pass",
   "repeated_fc_relu_fuse_pass",
   "depthwise_conv_mkldnn_pass",
   "conv_bn_fuse_pass",
   "conv_eltwiseadd_bn_fuse_pass",
   "conv_affine_channel_mkldnn_fuse_pass",
   "conv_transpose_bn_fuse_pass",
   "conv_transpose_eltwiseadd_bn_fuse_pass",
   "conv_bias_mkldnn_fuse_pass",
   "conv_transpose_bias_mkldnn_fuse_pass",
   "conv_elementwise_add_mkldnn_fuse_pass",
   "conv_activation_mkldnn_fuse_pass",
   "fc_fuse_pass",
   "repeated_fc_relu_fuse_pass",
   "fc_mkldnn_pass",
   "fc_act_mkldnn_fuse_pass",
   "matmul_transpose_reshape_mkldnn_fuse_pass",
   "batch_norm_act_fuse_pass",
   "softplus_activation_onednn_fuse_pass",
//    "compute_propagate_scales_mkldnn_pass",
   "scale_matmul_fuse_pass",
   "reshape_transpose_matmul_mkldnn_fuse_pass",
   "matmul_elementwise_add_mkldnn_fuse_pass",
   "operator_scale_onednn_fuse_pass",
   "operator_unsqueeze2_onednn_fuse_pass",
   "operator_reshape2_onednn_fuse_pass",
   "cpu_quantize_placement_pass",
   "cpu_quantize_pass",
   "cpu_quantize_squash_pass",
   "quant_transpose2_dequant_onednn_fuse_pass"
});
TEST(cpuQuantizePass, ConvReLU6) {
  paddle::framework::ProgramDesc prog;
  auto* block = prog.MutableBlock(0);

  auto* conv2d_op = block->AppendOp();
  conv2d_op->SetType("conv2d");
  conv2d_op->SetInput("Input", {"conv2d-X"});
  conv2d_op->SetInput("Filter", {"conv2d-Y"});
  conv2d_op->SetOutput("Output", {"conv2d-Out"});
  
  const std::vector<int> strides({1, 1});
  const std::vector<int> paddings({1, 1});
  const std::vector<int> dilations({1, 1});
  const int groups = 1;

  conv2d_op->SetAttr("strides", strides);
  conv2d_op->SetAttr("paddings", paddings);
  conv2d_op->SetAttr("dilations", dilations);
  conv2d_op->SetAttr("groups", groups);

  auto* relu6_op = block->AppendOp();
  relu6_op->SetType("relu6");
  relu6_op->SetAttr("threshold", 6.f);
  relu6_op->SetInput("X", {"conv2d-Out"});
  relu6_op->SetOutput("Out", {"relu-Out"}); 

  auto place = phi::CPUPlace();
  auto* scales = new VarQuantScale();
  phi::DenseTensor input_tensor;
  phi::DenseTensor weight_tensor;
  input_tensor.Resize({1});
  weight_tensor.Resize({1});
  auto* ptr_input = input_tensor.mutable_data<double>(place);
  auto* ptr_weight = weight_tensor.mutable_data<double>(place);
  ptr_input[0] = SCALE;
  ptr_weight[0] = SCALE;

  (*scales)["conv2d-X"] = std::make_pair(false, std::move(input_tensor));
  (*scales)["conv2d-Y"] = std::make_pair(false, std::move(weight_tensor));

  paddle::framework::Scope scope;

  std::unique_ptr<paddle::framework::ir::Graph> graph(new paddle::framework::ir::Graph(prog));
  (graph)->SetNotOwned(paddle::framework::ir::kParamScopeAttr, &scope);
   
  int num_nodes_before = graph->Nodes().size();
  VLOG(3) << "Before apply pass, Graph node num:" << num_nodes_before;
  VLOG(3) << DebugString(graph);

  for (const auto &pass : PreGraphPasses) {
    auto pass_ = paddle::framework::ir::PassRegistry::Instance().Get(pass);
    if (pass == "fc_fuse_pass") {
        pass_->Set("use_gpu", new bool(false));
    }else if(pass == "cpu_quantize_pass"){
        pass_->Set("quant_var_scales", scales);
    };
    graph.reset(pass_->Apply(graph.release()));
  }
  int num_nodes_after = graph->Nodes().size();
  VLOG(3) << "-------------------------------------------";
  VLOG(3) << "After apply pass, Graph node num:" << num_nodes_after;
  VLOG(3) << DebugString(graph);
  for (auto* node : graph->Nodes()) {
    if(node->IsOp() && node->Op() && node->Op()->Type() =="fused_conv2d"){
        VLOG(3) << "fused_conv2d OP: ";
        VLOG(3) << "Op Attr(fuse_beta) value: ";
        VLOG(3) << node->Op()->GetAttrIfExists<float>("fuse_beta");
        CHECK_EQ(node->Op()->GetAttrIfExists<float>("fuse_beta"), 6)
            << "Attr fuse_beta must equal to 6.";
    };
  };

}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
