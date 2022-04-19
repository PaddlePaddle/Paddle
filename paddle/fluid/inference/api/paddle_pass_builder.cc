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
#ifdef PADDLE_WITH_HIP
#include <miopen/miopen.h>
#endif
#include <glog/logging.h>
#include <algorithm>
#include <sstream>

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

size_t PaddlePassBuilder::GetPassIndex(const std::string &pass_type) {
  auto iter = std::find(std::begin(passes_), std::end(passes_), pass_type);
  if (iter == std::end(passes_)) return -1;
  return std::distance(std::begin(passes_), iter);
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

const std::vector<std::string> kTRTSubgraphPasses({
  "adaptive_pool2d_convert_global_pass",
      "shuffle_channel_detect_pass",           //
      "quant_conv2d_dequant_fuse_pass",        //
      "delete_quant_dequant_op_pass",          //
      "delete_quant_dequant_filter_op_pass",   //
      "delete_weight_dequant_linear_op_pass",  //
      "delete_quant_dequant_linear_op_pass",   //
      "add_support_int8_pass",                 //
      // "fc_fuse_pass",                        //
      "simplify_with_basic_ops_pass",                 //
      "embedding_eltwise_layernorm_fuse_pass",        //
      "preln_embedding_eltwise_layernorm_fuse_pass",  //
      "multihead_matmul_fuse_pass_v2",                //
      "multihead_matmul_fuse_pass_v3",                //
      "skip_layernorm_fuse_pass",                     //
      "preln_skip_layernorm_fuse_pass",               //
      "conv_bn_fuse_pass",                            //
      "unsqueeze2_eltwise_fuse_pass",                 //
      "trt_squeeze2_matmul_fuse_pass",                //
      "trt_reshape2_matmul_fuse_pass",                //
      "trt_flatten2_matmul_fuse_pass",                //
      "trt_map_matmul_v2_to_mul_pass",                //
      "trt_map_matmul_v2_to_matmul_pass",             //
      "trt_map_matmul_to_mul_pass",                   //
      "fc_fuse_pass",                                 //
      "conv_elementwise_add_fuse_pass",               //
      "tensorrt_subgraph_pass",                       //
      "conv_bn_fuse_pass",                            //
#if CUDNN_VERSION >= 7100  // To run conv_fusion, the version of cudnn must be
                           // guaranteed at least v7
// cudnn8.0 has memory leak problem in conv + eltwise + act, so we
// disable the pass.
#if !(CUDNN_VERSION >= 8000 && CUDNN_VERSION < 8100)
      "conv_elementwise_add_act_fuse_pass",   //
      "conv_elementwise_add2_act_fuse_pass",  //
#endif
#endif
      "transpose_flatten_concat_fuse_pass",
});

const std::vector<std::string> kDlnneSubgraphPasses({
    "is_test_pass",                  //
    "delete_dropout_op_pass"         //
    "simplify_with_basic_ops_pass",  //
    "conv_bn_fuse_pass",             //
    "depthwise_conv_bn_fuse_pass",   //
    "shuffle_channel_detect_pass",   //
    "dlnne_subgraph_pass",           //
});

const std::vector<std::string> kLiteSubgraphPasses({
#ifdef PADDLE_WITH_LITE
    "lite_subgraph_pass",
#endif
});

GpuPassStrategy::GpuPassStrategy() : PassStrategy({}) {
  passes_.assign({
    //   "identity_scale_op_clean_pass",             //
    "is_test_pass",                               //
        "simplify_with_basic_ops_pass",           //
        "conv_bn_fuse_pass",                      //
        "conv_eltwiseadd_bn_fuse_pass",           //
        "embedding_eltwise_layernorm_fuse_pass",  //
        "multihead_matmul_fuse_pass_v2",          //
        "gpu_cpu_squeeze2_matmul_fuse_pass",      //
        "gpu_cpu_reshape2_matmul_fuse_pass",      //
        "gpu_cpu_flatten2_matmul_fuse_pass",      //
        "gpu_cpu_map_matmul_v2_to_mul_pass",      //
        "gpu_cpu_map_matmul_v2_to_matmul_pass",   //
        "gpu_cpu_map_matmul_to_mul_pass",         //
        "fc_fuse_pass",                           //
        "fc_elementwise_layernorm_fuse_pass",     //
#if CUDNN_VERSION >= 7100  // To run conv_fusion, the version of cudnn must be
                           // guaranteed at least v7
// cudnn8.0 has memory leak problem in conv + eltwise + act, so we
// disable the pass.
#if !(CUDNN_VERSION >= 8000 && CUDNN_VERSION < 8100)
        "conv_elementwise_add_act_fuse_pass",   //
        "conv_elementwise_add2_act_fuse_pass",  //
#endif
        "conv_elementwise_add_fuse_pass",      //
#endif                                         //
        "transpose_flatten_concat_fuse_pass",  //
        // following pass should be located in the last, since it will
        // work on all fused ops.
        "runtime_context_cache_pass"
  });

  use_gpu_ = true;
}

void GpuPassStrategy::EnableCUDNN() {
  if (!use_cudnn_) {
    passes_.insert(passes_.begin(), "cudnn_placement_pass");
  }
  use_cudnn_ = true;
}

void GpuPassStrategy::Exp_EnableUseGpuFp16() {
  passes_.assign({
    "is_test_pass",                               //
        "simplify_with_basic_ops_pass",           //
        "conv_bn_fuse_pass",                      //
        "conv_eltwiseadd_bn_fuse_pass",           //
        "embedding_eltwise_layernorm_fuse_pass",  //
        "multihead_matmul_fuse_pass_v2",          //
        "gpu_cpu_squeeze2_matmul_fuse_pass",      //
        "gpu_cpu_reshape2_matmul_fuse_pass",      //
        "gpu_cpu_flatten2_matmul_fuse_pass",      //
        "gpu_cpu_map_matmul_v2_to_mul_pass",      //
        "gpu_cpu_map_matmul_v2_to_matmul_pass",   //
        "gpu_cpu_map_matmul_to_mul_pass",         //
        // "fc_fuse_pass",                        //
        "fc_elementwise_layernorm_fuse_pass",  //
#if CUDNN_VERSION >= 7100  // To run conv_fusion, the version of cudnn must be
                           // guaranteed at least v7
// cudnn8.0 has memory leak problem in conv + eltwise + act, so we
// disable the pass.
#if !(CUDNN_VERSION >= 8000 && CUDNN_VERSION < 8100)
        "conv_elementwise_add_act_fuse_pass",   //
        "conv_elementwise_add2_act_fuse_pass",  //
#endif
        "conv_elementwise_add_fuse_pass",      //
#endif                                         //
        "transpose_flatten_concat_fuse_pass",  //
        "mixed_precision_configure_pass",      //
        "runtime_context_cache_pass"           //
  });

  use_gpu_fp16_ = true;
}

void GpuPassStrategy::EnableMKLDNN() {
  LOG(ERROR) << "GPU not support MKLDNN yet";
}

void GpuPassStrategy::EnableMkldnnQuantizer() {
  LOG(ERROR) << "GPU not support MKL-DNN quantization";
}

void GpuPassStrategy::EnableMkldnnBfloat16() {
  LOG(ERROR) << "GPU not support MKL-DNN bfloat16";
}

void GpuPassStrategy::EnableMkldnnInt8() {
  LOG(ERROR) << "GPU not support MKL-DNN int8";
}

CpuPassStrategy::CpuPassStrategy() : PassStrategy({}) {
  // NOTE the large fusions should be located in the front, so that they will
  // not be damaged by smaller ones.
  passes_.assign({"simplify_with_basic_ops_pass",  //
                  "layer_norm_fuse_pass",
                  "attention_lstm_fuse_pass",       //
                  "seqconv_eltadd_relu_fuse_pass",  //
                  // "seqpool_concat_fuse_pass",    //
                  "seqpool_cvm_concat_fuse_pass",  //
                  // "embedding_fc_lstm_fuse_pass", //
                  // TODO(wilber): fix correctness problem.
                  // "fc_lstm_fuse_pass",                    //
                  "mul_lstm_fuse_pass",                      //
                  "fc_gru_fuse_pass",                        //
                  "mul_gru_fuse_pass",                       //
                  "seq_concat_fc_fuse_pass",                 //
                  "gpu_cpu_squeeze2_matmul_fuse_pass",       //
                  "gpu_cpu_reshape2_matmul_fuse_pass",       //
                  "gpu_cpu_flatten2_matmul_fuse_pass",       //
                  "matmul_v2_scale_fuse_pass",               //
                  "gpu_cpu_map_matmul_v2_to_mul_pass",       //
                  "gpu_cpu_map_matmul_v2_to_matmul_pass",    //
                  "matmul_scale_fuse_pass",                  //
                  "gpu_cpu_map_matmul_to_mul_pass",          //
                  "fc_fuse_pass",                            //
                  "repeated_fc_relu_fuse_pass",              //
                  "squared_mat_sub_fuse_pass",               //
                  "conv_bn_fuse_pass",                       //
                  "conv_eltwiseadd_bn_fuse_pass",            //
                  "conv_transpose_bn_fuse_pass",             //
                  "conv_transpose_eltwiseadd_bn_fuse_pass",  //
                  "is_test_pass",                            //
                  // following pass should be located in the last, since
                  // it will work on all fused ops.
                  "runtime_context_cache_pass"});

  use_gpu_ = false;
}

void CpuPassStrategy::EnableCUDNN() { LOG(ERROR) << "CPU not support cuDNN"; }

void CpuPassStrategy::EnableMKLDNN() {
// TODO(Superjomn) Consider the way to mix CPU with GPU.
#ifdef PADDLE_WITH_MKLDNN
  if (!use_mkldnn_) {
    passes_.insert(passes_.begin(), "mkldnn_placement_pass");

    for (auto &pass : std::vector<std::string>({
             "depthwise_conv_mkldnn_pass",    //
             "conv_bn_fuse_pass",             // Execute BN passes again to
             "conv_eltwiseadd_bn_fuse_pass",  // preserve correct pass order
             "conv_transpose_bn_fuse_pass",   //
             "conv_transpose_eltwiseadd_bn_fuse_pass",  //
             "conv_bias_mkldnn_fuse_pass",              //
             "conv_transpose_bias_mkldnn_fuse_pass",
             // TODO(baoachun): Need to support 5-dimensional input.
             // "conv3d_bias_mkldnn_fuse_pass",  //
             "conv_elementwise_add_mkldnn_fuse_pass",
             "conv_concat_relu_mkldnn_fuse_pass",
             "conv_relu_mkldnn_fuse_pass",          //
             "conv_leaky_relu_mkldnn_fuse_pass",    //
             "conv_relu6_mkldnn_fuse_pass",         //
             "conv_swish_mkldnn_fuse_pass",         //
             "conv_hard_swish_mkldnn_fuse_pass",    //
             "conv_mish_mkldnn_fuse_pass",          //
             "conv_hard_sigmoid_mkldnn_fuse_pass",  //
             // TODO(baoachun) fix int8 accuracy
             "conv_gelu_mkldnn_fuse_pass",
             "scale_matmul_fuse_pass",                        //
             "reshape_transpose_matmul_mkldnn_fuse_pass",     //
             "reshape_transpose_matmul_v2_mkldnn_fuse_pass",  //
             "matmul_transpose_reshape_fuse_pass",            //
             "matmul_v2_transpose_reshape_fuse_pass",         //
             // Disabled due to topology-dependent speed-up
             //  "fc_mkldnn_pass",
             //  "fc_act_mkldnn_fuse_pass",
             "fc_elementwise_add_mkldnn_fuse_pass",   //
             "batch_norm_act_fuse_pass",              //
             "softplus_activation_mkldnn_fuse_pass",  //
             "shuffle_channel_mkldnn_detect_pass",    //
             "elt_act_mkldnn_fuse_pass",              //
             // TODO(intel): Please fix the bug on windows.
             // https://github.com/PaddlePaddle/Paddle/issues/29710
             // "mkldnn_inplace_pass",  // This pass should be activated after
             // fuses. Disabled by default due to
             // little gain and lots of problems
         })) {
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

void CpuPassStrategy::EnableMkldnnBfloat16() {
#ifdef PADDLE_WITH_MKLDNN
  if (!use_mkldnn_bfloat16_) {
    passes_.push_back("cpu_bfloat16_placement_pass");
    passes_.push_back("cpu_bfloat16_pass");
    passes_.push_back("cpu_quantize_squash_pass");
  }
  use_mkldnn_bfloat16_ = true;
#else
  use_mkldnn_bfloat16_ = false;
#endif
}

void CpuPassStrategy::EnableMkldnnInt8() {
#ifdef PADDLE_WITH_MKLDNN
  if (!use_mkldnn_int8_) {
    passes_.clear();
    passes_.push_back("quant_dequant_mkldnn_pass");
    passes_.push_back("layer_norm_fuse_pass");
    passes_.push_back("attention_lstm_fuse_pass");
    passes_.push_back("seqconv_eltadd_relu_fuse_pass");
    passes_.push_back("fc_lstm_fuse_pass");
    passes_.push_back("mul_lstm_fuse_pass");
    passes_.push_back("fc_gru_fuse_pass");
    passes_.push_back("mul_gru_fuse_pass");
    passes_.push_back("multi_gru_fuse_pass");
    passes_.push_back("multi_gru_seq_fuse_pass");
    passes_.push_back("seq_concat_fc_fuse_pass");
    passes_.push_back("gpu_cpu_squeeze2_matmul_fuse_pass");
    passes_.push_back("gpu_cpu_reshape2_matmul_fuse_pass");
    passes_.push_back("gpu_cpu_flatten2_matmul_fuse_pass");
    passes_.push_back("matmul_v2_scale_fuse_pass");
    passes_.push_back("squared_mat_sub_fuse_pass");
    passes_.push_back("is_test_pass");
    passes_.push_back("gpu_cpu_map_matmul_v2_to_mul_pass");
    passes_.push_back("gpu_cpu_map_matmul_v2_to_matmul_pass");
    passes_.push_back("matmul_scale_fuse_pass");
    passes_.push_back("gpu_cpu_map_matmul_to_mul_pass");
    passes_.push_back("repeated_fc_relu_fuse_pass");
    passes_.push_back("mkldnn_placement_pass");
    passes_.push_back("depthwise_conv_mkldnn_pass");
    passes_.push_back("conv_bn_fuse_pass");
    passes_.push_back("conv_eltwiseadd_bn_fuse_pass");
    passes_.push_back("conv_transpose_bn_fuse_pass");
    passes_.push_back("conv_transpose_eltwiseadd_bn_fuse_pass");
    passes_.push_back("conv_bias_mkldnn_fuse_pass");
    passes_.push_back("conv_transpose_bias_mkldnn_fuse_pass");
    passes_.push_back("conv_elementwise_add_mkldnn_fuse_pass");
    passes_.push_back("conv_concat_relu_mkldnn_fuse_pass");
    passes_.push_back("conv_relu_mkldnn_fuse_pass");
    passes_.push_back("conv_leaky_relu_mkldnn_fuse_pass");
    passes_.push_back("conv_relu6_mkldnn_fuse_pass");
    passes_.push_back("conv_swish_mkldnn_fuse_pass");
    passes_.push_back("conv_hard_swish_mkldnn_fuse_pass");
    passes_.push_back("conv_mish_mkldnn_fuse_pass");
    passes_.push_back("conv_hard_sigmoid_mkldnn_fuse_pass");
    passes_.push_back("conv_gelu_mkldnn_fuse_pass");
    passes_.push_back("fc_fuse_pass");
    passes_.push_back("repeated_fc_relu_fuse_pass");
    passes_.push_back("fc_mkldnn_pass");
    passes_.push_back("fc_act_mkldnn_fuse_pass");
    passes_.push_back("matmul_transpose_reshape_fuse_pass");
    passes_.push_back("matmul_v2_transpose_reshape_fuse_pass");
    passes_.push_back("batch_norm_act_fuse_pass");
    passes_.push_back("softplus_activation_mkldnn_fuse_pass");
    passes_.push_back("compute_propagate_scales_mkldnn_pass");
    passes_.push_back("scale_matmul_fuse_pass");
    passes_.push_back("reshape_transpose_matmul_mkldnn_fuse_pass");
    passes_.push_back("reshape_transpose_matmul_v2_mkldnn_fuse_pass");
    passes_.push_back("cpu_quantize_placement_pass");
    passes_.push_back("cpu_quantize_pass");
    passes_.push_back("cpu_quantize_squash_pass");
    passes_.push_back("simplify_with_basic_ops_pass");
    passes_.push_back("mkldnn_inplace_pass");
    passes_.push_back("runtime_context_cache_pass");
  }
  use_mkldnn_int8_ = true;
#else
  use_mkldnn_int8_ = false;
#endif
}

IpuPassStrategy::IpuPassStrategy() : PassStrategy({}) {
  passes_.assign({"inference_process_pass"});
}

}  // namespace paddle
