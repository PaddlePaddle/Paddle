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

#pragma once

#include <sstream>
#include <string>
#include <vector>

namespace paddle {
/*
 * This is a pass builder based on string. It is part of inference API.
 */
class PaddlePassBuilder {
 public:
  void AppendPass(const std::string &pass_type);

  void InsertPass(size_t idx, const std::string &pass_type);

  // Delete the `idx`-th pass.
  void DeletePass(size_t idx);

  // Delete all the passes that has type `pass_type`.
  void DeletePass(const std::string &pass_type);

  // Append an analysis pass.
  void AppendAnalysisPass(const std::string &pass);

  // Visualize the computation graph after each pass by generating a DOT
  // language file, one can draw them with the Graphviz toolkit.
  void TurnOnDebug();

  // Human-readible information.
  std::string DebugString();

  const std::vector<std::string> &AllPasses() const { return passes_; }
  const std::vector<std::string> &AnalysisPasses() const {
    return analysis_passes_;
  }

 protected:
  std::vector<std::string> analysis_passes_{{"ir_analysis_compose_pass"}};
  std::vector<std::string> passes_;
};

/*
 * Pass strategy to help control the IR passes.
 */
class PassStrategy : public PaddlePassBuilder {
 public:
  // The MKLDNN control exists in both CPU and GPU mode, because there can be
  // still some CPU kernels running in CPU mode.
  virtual void EnableMKLDNN() {}

  virtual ~PassStrategy() = default;
};

/*
 * The CPU passes controller, it is used in AnalysisPredictor with CPU mode.
 */
class CpuPassStrategy : public PassStrategy {
 public:
  CpuPassStrategy() {
    // NOTE the large fusions should be located in the front, so that they will
    // not be damaged by smaller ones.
    passes_.assign({
        "infer_clean_graph_pass",         //
        "attention_lstm_fuse_pass",       //
        "seqconv_eltadd_relu_fuse_pass",  //
        // "embedding_fc_lstm_fuse_pass", //
        "fc_lstm_fuse_pass",             //
        "mul_lstm_fuse_pass",            //
        "fc_gru_fuse_pass",              //
        "mul_gru_fuse_pass",             //
        "seq_concat_fc_fuse_pass",       //
        "fc_fuse_pass",                  //
        "conv_bn_fuse_pass",             //
        "conv_eltwiseadd_bn_fuse_pass",  //
        "is_test_pass",                  //
    });
  }

  virtual ~CpuPassStrategy() = default;

  void EnableMKLDNN() override {
// TODO(Superjomn) Consider the way to mix CPU with GPU.
#ifdef PADDLE_WITH_MKLDNN
    passes_.insert(passes_.begin(), "mkldnn_placement_pass");

    for (auto &pass :
         std::vector<std::string>({"depthwise_conv_mkldnn_pass",  //
                                   "conv_bias_mkldnn_fuse_pass",  //
                                   "conv_relu_mkldnn_fuse_pass",  //
                                   "conv_elementwise_add_mkldnn_fuse_pass"})) {
      passes_.push_back(pass);
    }
#endif
  }
};

/*
 * The GPU passes strategy, it is used in
 */
class GpuPassStrategy : public PassStrategy {
 public:
  GpuPassStrategy() : PassStrategy({}) {
    // TODO(NHZlX) Problem with Data synchronization between GPU and CPU
    // When running in GPU mode, the parameters are all on GPU. But the
    // opearations of "conv_bn_fuse_pass" are on CPU.
    passes_.assign({
        "infer_clean_graph_pass",
        // "infer_clean_graph_pass", "conv_bn_fuse_pass",
    });
  }

  void EnableMKLDNN() override;

  virtual ~GpuPassStrategy() = default;
};

}  // namespace paddle
