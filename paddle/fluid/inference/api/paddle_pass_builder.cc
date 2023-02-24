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
  deleted_passes_.insert(pass_type);
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
  "trt_support_nhwc_pass",
      "adaptive_pool2d_convert_global_pass",       //
      "shuffle_channel_detect_pass",               //
      "quant_conv2d_dequant_fuse_pass",            //
      "delete_fill_constant_op_pass",              //
      "delete_quant_dequant_op_pass",              //
      "delete_quant_dequant_filter_op_pass",       //
      "trt_delete_weight_dequant_linear_op_pass",  //
      "delete_quant_dequant_linear_op_pass",       //
      "identity_scale_op_clean_pass",              //
      "add_support_int8_pass",                     //
      // "fc_fuse_pass",                        //
      "simplify_with_basic_ops_pass",                 //
      "trt_embedding_eltwise_layernorm_fuse_pass",    //
      "preln_embedding_eltwise_layernorm_fuse_pass",  //
      "delete_c_identity_op_pass",                    //
      "trt_multihead_matmul_fuse_pass_v2",            //
      "trt_multihead_matmul_fuse_pass_v3",            //
      "multihead_matmul_roformer_fuse_pass",          //
      "constant_folding_pass",                        //
      "trt_flash_multihead_matmul_fuse_pass",         //
      "trt_cross_multihead_matmul_fuse_pass",         //
      "vit_attention_fuse_pass",                      //
#if defined _WIN32  // Windows CI is TensorRT7.0. Remove this after upgrading.
#else
      "trt_skip_layernorm_fuse_pass",          //
      "preln_skip_layernorm_fuse_pass",        //
#endif
      "layernorm_shift_partition_fuse_pass",  //
      "merge_layernorm_fuse_pass",            //
      "preln_residual_bias_fuse_pass",        //
      "preln_layernorm_x_fuse_pass",          //
      "reverse_roll_fuse_pass",               //
      "conv_bn_fuse_pass",                    //
      "unsqueeze2_eltwise_fuse_pass",         //
      "trt_squeeze2_matmul_fuse_pass",        //
      "trt_flatten2_matmul_fuse_pass",        //
      "trt_map_matmul_v2_to_mul_pass",        //
      "trt_map_matmul_v2_to_matmul_pass",     //
      "trt_map_matmul_to_mul_pass",           //
      "fc_fuse_pass",                         //
      "conv_elementwise_add_fuse_pass",       //
#if defined _WIN32  // Windows CI is TensorRT7.0. Remove this after upgrading.
#else
      "trans_layernorm_fuse_pass",             //
#endif
      "remove_padding_recover_padding_pass",         //
      "delete_remove_padding_recover_padding_pass",  //
      // "yolo_box_fuse_pass",      //
      "dense_fc_to_sparse_pass",                //
      "dense_multihead_matmul_to_sparse_pass",  //
#if defined _WIN32  // Windows CI is TensorRT7.0. Remove this after upgrading.
#else
      "elementwise_groupnorm_act_pass",        //
      "preln_elementwise_groupnorm_act_pass",  //
      "groupnorm_act_pass",                    //
      "elementwiseadd_transpose_pass",         //
#endif
      "tensorrt_subgraph_pass",  //
      "conv_bn_fuse_pass",       //
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
    "delete_dropout_op_pass",        //
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

// TODO(inference): Most of the existing pass fusion operators do not
// support fp16/bf16 precision, temporarily use low precision pass to prevent
// running errors. After fusion operator supports low precision, delete this.
const std::vector<std::string> kGpuLowerPrecisionPasses{
    "map_op_to_another_pass",
    "identity_scale_op_clean_pass",
    "simplify_with_basic_ops_pass",
    "silu_fuse_pass",
    "delete_quant_dequant_linear_op_pass",
    "delete_weight_dequant_linear_op_pass",
    "conv_bn_fuse_pass",
    "conv_eltwiseadd_bn_fuse_pass",
    "conv_elementwise_add_act_fuse_pass",
    "conv_elementwise_add2_act_fuse_pass",
    "conv_elementwise_add_fuse_pass",
    "conv2d_fusion_layout_transfer_pass",
    "multihead_matmul_fuse_pass_v2",
    "fused_multi_transformer_encoder_pass",
    "fused_multi_transformer_decoder_pass",
    "fused_multi_transformer_encoder_fuse_qkv_pass",
    "fused_multi_transformer_decoder_fuse_qkv_pass",
    "multi_devices_fused_multi_transformer_encoder_pass",
    "multi_devices_fused_multi_transformer_encoder_fuse_qkv_pass",
    "multi_devices_fused_multi_transformer_decoder_fuse_qkv_pass",
    "fuse_multi_transformer_layer_pass",
    "gpu_cpu_map_matmul_v2_to_mul_pass",
    "gpu_cpu_map_matmul_v2_to_matmul_pass",
    "gpu_cpu_map_matmul_to_mul_pass",
    "fc_fuse_pass",
    // "fc_elementwise_layernorm_fuse_pass",
    "embedding_eltwise_layernorm_fuse_pass",
    "inplace_op_var_pass"};

const std::vector<std::string> kTrtLowerPrecisionPasses{
    "simplify_with_basic_ops_pass",
    // "conv_bn_fuse_pass",
    // "conv_eltwiseadd_bn_fuse_pass",
    "trt_embedding_eltwise_layernorm_fuse_pass",
    "trt_skip_layernorm_fuse_pass",
    "trt_map_matmul_v2_to_mul_pass",
    "trt_map_matmul_v2_to_matmul_pass",
    "trt_map_matmul_to_mul_pass",
    "fc_fuse_pass",
    "tensorrt_subgraph_pass",
};

const std::vector<std::string> kCINNCompilerPasses{
    "gpu_cpu_map_matmul_v2_to_mul_pass",
    "gpu_cpu_map_matmul_v2_to_matmul_pass",
    "gpu_cpu_map_matmul_to_mul_pass",
    "build_cinn_pass",
};

GpuPassStrategy::GpuPassStrategy() : PassStrategy({}) {
  passes_.assign({
    "map_op_to_another_pass",                                           //
        "identity_scale_op_clean_pass",                                 //
        "is_test_pass",                                                 //
        "simplify_with_basic_ops_pass",                                 //
        "delete_quant_dequant_linear_op_pass",                          //
        "delete_weight_dequant_linear_op_pass",                         //
        "constant_folding_pass",                                        //
        "silu_fuse_pass",                                               //
        "conv_bn_fuse_pass",                                            //
        "conv_eltwiseadd_bn_fuse_pass",                                 //
        "embedding_eltwise_layernorm_fuse_pass",                        //
        "multihead_matmul_fuse_pass_v2",                                //
        "vit_attention_fuse_pass",                                      //
        "fused_multi_transformer_encoder_pass",                         //
        "fused_multi_transformer_decoder_pass",                         //
        "fused_multi_transformer_encoder_fuse_qkv_pass",                //
        "fused_multi_transformer_decoder_fuse_qkv_pass",                //
        "multi_devices_fused_multi_transformer_encoder_pass",           //
        "multi_devices_fused_multi_transformer_encoder_fuse_qkv_pass",  //
        "multi_devices_fused_multi_transformer_decoder_fuse_qkv_pass",  //
        "fuse_multi_transformer_layer_pass",                            //
        "gpu_cpu_squeeze2_matmul_fuse_pass",                            //
        "gpu_cpu_reshape2_matmul_fuse_pass",                            //
        "gpu_cpu_flatten2_matmul_fuse_pass",                            //
        "gpu_cpu_map_matmul_v2_to_mul_pass",                            //
        "gpu_cpu_map_matmul_v2_to_matmul_pass",                         //
        "matmul_scale_fuse_pass",                                       //
        "multihead_matmul_fuse_pass_v3",                                //
        "gpu_cpu_map_matmul_to_mul_pass",                               //
        "fc_fuse_pass",                                                 //
        "fc_elementwise_layernorm_fuse_pass",                           //
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
        // TODO(liuyuanle): rewrite this pass with new logic
        // "conv2d_fusion_layout_transfer_pass",  //
        "auto_mixed_precision_pass",  //
        "inplace_op_var_pass",        // should be the last pass.
  });

  use_gpu_ = true;
}

void GpuPassStrategy::EnableCUDNN() {
  if (!use_cudnn_) {
    passes_.insert(passes_.begin(), "cudnn_placement_pass");
  }
  use_cudnn_ = true;
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

void GpuPassStrategy::DisableMkldnnFcPasses() {
  LOG(ERROR) << "GPU not support MKL-DNN fc";
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
                  "constant_folding_pass"});

  use_gpu_ = false;
}

void CpuPassStrategy::EnableCUDNN() { LOG(ERROR) << "CPU not support cuDNN"; }

void CpuPassStrategy::EnableMKLDNN() {
// TODO(Superjomn) Consider the way to mix CPU with GPU.
#ifdef PADDLE_WITH_MKLDNN
  if (!use_mkldnn_) {
    passes_.insert(passes_.begin(), "mkldnn_placement_pass");

    for (auto &pass : std::vector<std::string>({
             "squeeze2_transpose2_onednn_fuse_pass",
             "depthwise_conv_mkldnn_pass",    //
             "conv_bn_fuse_pass",             // Execute BN passes again to
             "conv_eltwiseadd_bn_fuse_pass",  // preserve correct pass order
             "conv_affine_channel_mkldnn_fuse_pass",    //
             "conv_transpose_bn_fuse_pass",             //
             "conv_transpose_eltwiseadd_bn_fuse_pass",  //
             "conv_bias_mkldnn_fuse_pass",              //
             "conv_transpose_bias_mkldnn_fuse_pass",
             "interpolate_mkldnn_pass",
             // TODO(baoachun): Need to support 5-dimensional input.
             // "conv3d_bias_mkldnn_fuse_pass",  //
             "conv_elementwise_add_mkldnn_fuse_pass",
             "conv_activation_mkldnn_fuse_pass",           //
             "scale_matmul_fuse_pass",                     //
             "reshape_transpose_matmul_mkldnn_fuse_pass",  //
             "matmul_transpose_reshape_mkldnn_fuse_pass",  //
             "matmul_elementwise_add_mkldnn_fuse_pass",    //
             "matmul_activation_mkldnn_fuse_pass",         //
             // Disabled due to topology-dependent speed-up
             "fc_mkldnn_pass",
             "fc_act_mkldnn_fuse_pass",
             "fc_elementwise_add_mkldnn_fuse_pass",   //
             "batch_norm_act_fuse_pass",              //
             "softplus_activation_mkldnn_fuse_pass",  //
             "shuffle_channel_mkldnn_detect_pass",    //
             "elt_act_mkldnn_fuse_pass",              //
             "layer_norm_onednn_optimization_pass",   //
             "operator_scale_onednn_fuse_pass",       //
             "operator_unsqueeze2_onednn_fuse_pass",  //
             "operator_reshape2_onednn_fuse_pass",    //
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
    passes_.push_back("fc_mkldnn_pass");
    passes_.push_back("fc_act_mkldnn_fuse_pass");
    passes_.push_back("fc_elementwise_add_mkldnn_fuse_pass");

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
    passes_.push_back("simplify_with_basic_ops_pass");
    passes_.push_back("quant_dequant_mkldnn_pass");
    passes_.push_back("mkldnn_placement_pass");
    passes_.push_back("constant_folding_pass");
    passes_.push_back("squeeze2_transpose2_onednn_fuse_pass");
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
    passes_.push_back("depthwise_conv_mkldnn_pass");
    passes_.push_back("conv_bn_fuse_pass");
    passes_.push_back("conv_eltwiseadd_bn_fuse_pass");
    passes_.push_back("conv_affine_channel_mkldnn_fuse_pass");
    passes_.push_back("conv_transpose_bn_fuse_pass");
    passes_.push_back("conv_transpose_eltwiseadd_bn_fuse_pass");
    passes_.push_back("conv_bias_mkldnn_fuse_pass");
    passes_.push_back("conv_transpose_bias_mkldnn_fuse_pass");
    passes_.push_back("conv_elementwise_add_mkldnn_fuse_pass");
    passes_.push_back("conv_activation_mkldnn_fuse_pass");
    passes_.push_back("fc_fuse_pass");
    passes_.push_back("repeated_fc_relu_fuse_pass");
    passes_.push_back("fc_mkldnn_pass");
    passes_.push_back("fc_act_mkldnn_fuse_pass");
    passes_.push_back("fc_elementwise_add_mkldnn_fuse_pass");
    passes_.push_back("matmul_transpose_reshape_mkldnn_fuse_pass");
    passes_.push_back("batch_norm_act_fuse_pass");
    passes_.push_back("softplus_activation_mkldnn_fuse_pass");
    passes_.push_back("compute_propagate_scales_mkldnn_pass");
    passes_.push_back("scale_matmul_fuse_pass");
    passes_.push_back("reshape_transpose_matmul_mkldnn_fuse_pass");
    passes_.push_back("matmul_elementwise_add_mkldnn_fuse_pass");
    passes_.push_back("layer_norm_onednn_optimization_pass");
    passes_.push_back("operator_scale_onednn_fuse_pass");
    passes_.push_back("operator_unsqueeze2_onednn_fuse_pass");
    passes_.push_back("operator_reshape2_onednn_fuse_pass");
    passes_.push_back("cpu_quantize_placement_pass");
    passes_.push_back("cpu_quantize_pass");
    passes_.push_back("cpu_quantize_squash_pass");
    passes_.push_back("quant_transpose2_dequant_onednn_fuse_pass");
    passes_.push_back("int8_scale_calculation_mkldnn_pass");
    passes_.push_back("params_quantization_mkldnn_pass");
  }
  use_mkldnn_int8_ = true;
#else
  use_mkldnn_int8_ = false;
#endif
}

void CpuPassStrategy::DisableMkldnnFcPasses() {
#ifdef PADDLE_WITH_MKLDNN
  if (!disable_mkldnn_fc_passes_) {
    EraseFcMkldnnPasses();
  }
  disable_mkldnn_fc_passes_ = true;
#else
  disable_mkldnn_fc_passes_ = false;
#endif
}

void CpuPassStrategy::EraseFcMkldnnPasses() {
  std::vector<std::string> fc_passes_to_erase(
      {"fc_mkldnn_pass",
       "fc_act_mkldnn_fuse_pass",
       "fc_elementwise_add_mkldnn_fuse_pass"});
  for (const auto &pass : fc_passes_to_erase) {
    int idx = GetPassIndex(pass);
    if (idx != -1) {
      passes_.erase(std::begin(passes_) + idx);
    }
  }
}

XpuPassStrategy::XpuPassStrategy() : PassStrategy({}) {
  passes_.assign({
      "delete_dropout_op_pass",
      "identity_scale_op_clean_pass",
      "generate_sequence_xpu_fuse_pass",
      "multi_encoder_xpu_fuse_pass",
      "multi_encoder_xpu_slice_fuse_pass",
      "embedding_with_eltwise_add_xpu_fuse_pass",
      "fc_xpu_fuse_pass",
      "link_xpu_op_max_pass",
  });
  use_xpu_ = true;
}

IpuPassStrategy::IpuPassStrategy() : PassStrategy({}) {
  passes_.assign({"inference_process_pass"});
}

}  // namespace paddle
