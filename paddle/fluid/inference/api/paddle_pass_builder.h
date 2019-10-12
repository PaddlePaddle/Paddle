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

/*! \file */

/*! \namespace paddle */
namespace paddle {

/** This is a pass builder based on string. It is part of inference API.
 */
class PaddlePassBuilder {
 public:
  explicit PaddlePassBuilder(const std::vector<std::string> &passes)
      : passes_(passes) {}

  void SetPasses(std::initializer_list<std::string> passes) {
    passes_ = passes;
  }

  /** Append a pass to the end of the passes. */
  void AppendPass(const std::string &pass_type);

  /** Insert a pass to a specific position.
   * @param idx the position to insert.
   * @param pass_type the pass key.
   */
  void InsertPass(size_t idx, const std::string &pass_type);

  /** Delete the `idx`-th pass. */
  void DeletePass(size_t idx);

  /** Delete all the passes that has type `pass_type`. */
  void DeletePass(const std::string &pass_type);

  void ClearPasses();
  /** Append an analysis pass. */
  void AppendAnalysisPass(const std::string &pass);

  /** Visualize the computation graph after each pass by generating a DOT
   * language file, one can draw them with the Graphviz toolkit.
   */
  void TurnOnDebug();

  /** Human-readible information. */
  std::string DebugString();

  const std::vector<std::string> &AllPasses() const { return passes_; }
  std::vector<std::string> AnalysisPasses() const {
    auto passes = analysis_passes_;
    // To make sure the ir_graph_to_program should be the last pass so any
    // modication of IR will persist to the program.
    passes.push_back("ir_graph_to_program_pass");
    return passes;
  }

 protected:
  std::vector<std::string> analysis_passes_{
      {"ir_graph_build_pass", "ir_graph_clean_pass", "ir_analysis_pass",
       "ir_params_sync_among_devices_pass", "adjust_cudnn_workspace_size_pass",
       "inference_op_replace_pass"}};
  std::vector<std::string> passes_;
};

/**Pass strategy to help control the IR passes.
 */
class PassStrategy : public PaddlePassBuilder {
 public:
  explicit PassStrategy(const std::vector<std::string> &passes)
      : PaddlePassBuilder(passes) {}

  /** Enable the use of cuDNN kernel
   */
  virtual void EnableCUDNN() {}

  /** The MKLDNN control exists in both CPU and GPU mode, because there can be
   * still some CPU kernels running in CPU mode.
   */
  virtual void EnableMKLDNN() {}

  /** Enable NGRAPH optimization
   */
  virtual void EnableNgraph() {}

  /** Enable MKLDNN quantize optimization
   */
  virtual void EnableMkldnnQuantizer() {}

  bool use_gpu() const { return use_gpu_; }

  virtual ~PassStrategy() = default;

 protected:
  bool use_ngraph_{false};
  bool use_gpu_{false};
  bool use_mkldnn_{false};
};

/** The CPU passes controller, it is used in AnalysisPredictor with CPU mode.
 */
class CpuPassStrategy : public PassStrategy {
 public:
  CpuPassStrategy();

  explicit CpuPassStrategy(const CpuPassStrategy &other)
      : PassStrategy(other.AllPasses()) {
    use_gpu_ = other.use_gpu_;
    use_ngraph_ = other.use_ngraph_;
    use_mkldnn_ = other.use_mkldnn_;
    use_mkldnn_quantizer_ = other.use_mkldnn_quantizer_;
  }

  virtual ~CpuPassStrategy() = default;

  void EnableCUDNN() override;
  void EnableNgraph() override;
  void EnableMKLDNN() override;
  void EnableMkldnnQuantizer() override;

 protected:
  bool use_ngraph_{false};
  bool use_mkldnn_quantizer_{false};
};

/** The GPU passes strategy, it is used in AnalysisPredictor with GPU mode.
 */
class GpuPassStrategy : public PassStrategy {
 public:
  GpuPassStrategy();

  explicit GpuPassStrategy(const GpuPassStrategy &other)
      : PassStrategy(other.AllPasses()) {
    use_gpu_ = true;
    use_cudnn_ = other.use_cudnn_;
  }

  void EnableCUDNN() override;
  void EnableNgraph() override;
  void EnableMKLDNN() override;
  void EnableMkldnnQuantizer() override;

  virtual ~GpuPassStrategy() = default;

 protected:
  bool use_cudnn_{false};
};

extern const std::vector<std::string> kTRTSubgraphPasses;
extern const std::vector<std::string> kAnakinSubgraphPasses;

}  // namespace paddle
