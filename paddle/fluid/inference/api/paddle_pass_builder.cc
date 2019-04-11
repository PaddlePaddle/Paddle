// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/api/paddle_pass_builder.h"
#ifdef PADDLE_WITH_CUDA
#include <cudnn.h>
#endif
#include <glog/logging.h>

namespace paddle {

void PaddlePassBuilder::AppendPass(const std::string &pass_type) {
  passes_.push_back(pass_type);
}

void PaddlePassBuilder::TurnOnDebug() {
  std::vector<std::string> passes;
  auto it = std::begin(passes_);
  while (it != std::end(passes_)) {
    if (*it != "graph_viz_pass") {
      it = passes_.insert(it + 1, "graph_viz_pass");
    } else {
      ++it;
    }
  }
}

std::string PaddlePassBuilder::DebugString() {
  std::stringstream ss;
  ss << "Passes to apply:\n";
  for (auto &pass : passes_) {
    ss << "  - " << pass << '\n';
  }
  return ss.str();
}

void PaddlePassBuilder::DeletePass(const std::string &pass_type) {
  auto it = std::begin(passes_);
  while (it != std::end(passes_)) {
    if (*it == pass_type) {
      it = passes_.erase(it);
    } else {
      ++it;
    }
  }
}

void PaddlePassBuilder::InsertPass(size_t idx, const std::string &pass_type) {
  passes_.insert(std::begin(passes_) + idx, pass_type);
}

void PaddlePassBuilder::DeletePass(size_t idx) {
  passes_.erase(std::begin(passes_) + idx);
}

void PaddlePassBuilder::AppendAnalysisPass(const std::string &pass) {
  analysis_passes_.push_back(pass);
}

void PaddlePassBuilder::ClearPasses() { passes_.clear(); }

// The following passes works for Anakin sub-graph engine.
const std::vector<std::string> kAnakinSubgraphPasses({
    "infer_clean_graph_pass",                       //
    "graph_viz_pass",                               //
    "quant_conv2d_dequant_fuse_pass",               //
    "graph_viz_pass",                               //
    "simplify_anakin_priorbox_detection_out_pass",  //
    "fillconstant_elementwisemul_fuse",             //
    "fc_fuse_pass",                                 //
    "conv_elementwise_add_fuse_pass",               //
    // "conv_bn_fuse_pass",                            //
    // "conv_elementwise_add_fuse_pass",               //
    "fc_gru_fuse_pass",      //
    "graph_viz_pass",        //
    "anakin_subgraph_pass",  //
    "graph_viz_pass",        //
    "fc_gru_fuse_pass",      //
    "graph_viz_pass",        //
});

GpuPassStrategy::GpuPassStrategy() : PassStrategy({}) {
  passes_.assign({
    "infer_clean_graph_pass",  //
        //   "identity_scale_op_clean_pass",              //
        "conv_affine_channel_fuse_pass",             //
        "conv_eltwiseadd_affine_channel_fuse_pass",  //
        "conv_bn_fuse_pass",                         //
#if CUDNN_VERSION >= 7100  // To run conv_fusion, the version of cudnn must be
                           // guaranteed at least v7
        "conv_elementwise_add_act_fuse_pass",   //
        "conv_elementwise_add2_act_fuse_pass",  //
        "conv_elementwise_add_fuse_pass",       //
#endif                                          //
        "transpose_flatten_concat_fuse_pass",
        // following two passes should be located in the last, since they will
        // work on all fused ops.
        "expected_kernel_cache_pass",  //
        "runtime_context_cache_pass"
  });

  use_gpu_ = true;
}

void GpuPassStrategy::EnableMKLDNN() {
  LOG(ERROR) << "GPU not support MKLDNN yet";
}

void GpuPassStrategy::EnableMkldnnQuantizer() {
  LOG(ERROR) << "GPU not support MKL-DNN quantization";
}

CpuPassStrategy::CpuPassStrategy() : PassStrategy({}) {
  // NOTE the large fusions should be located in the front, so that they will
  // not be damaged by smaller ones.
  passes_.assign({"infer_clean_graph_pass",         //
                  "attention_lstm_fuse_pass",       //
                  "seqconv_eltadd_relu_fuse_pass",  //
                  // "seqpool_concat_fuse_pass",    //
                  // "embedding_fc_lstm_fuse_pass", //
                  "fc_lstm_fuse_pass",             //
                  "mul_lstm_fuse_pass",            //
                  "fc_gru_fuse_pass",              //
                  "mul_gru_fuse_pass",             //
                  "seq_concat_fc_fuse_pass",       //
                  "fc_fuse_pass",                  //
                  "repeated_fc_relu_fuse_pass",    //
                  "squared_mat_sub_fuse_pass",     //
                  "conv_bn_fuse_pass",             //
                  "conv_eltwiseadd_bn_fuse_pass",  //
                  "is_test_pass",                  //
                  // following two passes should be located in the last, since
                  // they will work on all fused ops.
                  "expected_kernel_cache_pass",  //
                  "runtime_context_cache_pass"});

  use_gpu_ = false;
}

void CpuPassStrategy::EnableMKLDNN() {
// TODO(Superjomn) Consider the way to mix CPU with GPU.
#ifdef PADDLE_WITH_MKLDNN
  if (!use_mkldnn_) {
    passes_.insert(passes_.begin(), "mkldnn_placement_pass");

    for (auto &pass : std::vector<std::string>(
             {"depthwise_conv_mkldnn_pass",    //
              "conv_bn_fuse_pass",             // Execute BN passes again to
              "conv_eltwiseadd_bn_fuse_pass",  // preserve correct pass order
              "conv_bias_mkldnn_fuse_pass",    //
              "conv3d_bias_mkldnn_fuse_pass",  //
              "conv_elementwise_add_mkldnn_fuse_pass",
              "conv_relu_mkldnn_fuse_pass"})) {
      passes_.push_back(pass);
    }
  }
  use_mkldnn_ = true;
#else
  use_mkldnn_ = false;
#endif
}

void CpuPassStrategy::EnableMkldnnQuantizer() {
#ifdef PADDLE_WITH_MKLDNN
  if (!use_mkldnn_quantizer_) {
    passes_.push_back("cpu_quantize_placement_pass");
  }
  use_mkldnn_quantizer_ = true;
#else
  use_mkldnn_quantizer_ = false;
#endif
}

}  // namespace paddle
