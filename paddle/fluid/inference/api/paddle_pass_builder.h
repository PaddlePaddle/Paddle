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

///
/// \file paddle_pass_builder.h
///
/// \brief Class Paddle Passs Builder and its subclasses(pass strategies).
/// \section sec_intro Introduction
/// This class aims to build passes for paddle and define passes' strategies.
///
/// \author paddle-infer@baidu.com
/// \date 2020-3-23
/// \since 1.7

/// \namespace paddle
namespace paddle {

/// \class PaddlePassBuilder
/// \brief This class build passes based on vector<string> input. It is part of
/// inference API. Users can build passes, insert new passes, delete passes
/// using this class and its functions.
///
/// Example Usage:
///     Build a new pass.
/// \code{cpp}
/// const vector<string> passes(1, "conv_relu_mkldnn_fuse_pass");
/// PaddlePassBuilder builder(passes);
/// \endcode
class PaddlePassBuilder {
 public:
  /// \brief Constructor of the class. It stores the input passes.
  /// \param[in] passes passes' types.
  explicit PaddlePassBuilder(const std::vector<std::string> &passes)
      : passes_(passes) {}

  /// \brief Stores the input passes.
  /// \param[in] passes passes' types.
  void SetPasses(std::initializer_list<std::string> passes) {
    passes_ = passes;
  }

  /// \brief Append a pass to the end of the passes.
  /// \param[in] pass_type the type of the new pass.
  void AppendPass(const std::string &pass_type);

  /// \brief Insert a pass to a specific position.
  /// \param[in] idx the position to insert.
  /// \param[in] pass_type the type of insert pass.
  void InsertPass(size_t idx, const std::string &pass_type);

  /// \brief Delete the pass at certain position 'idx'.
  /// \param[in] idx the position to delete.
  void DeletePass(size_t idx);

  /// \brief Delete all passes that has a certain type 'pass_type'.
  /// \param[in] pass_type the certain pass type to be deleted.
  void DeletePass(const std::string &pass_type);

  /// \brief Delete all the passes.
  void ClearPasses();

  /// \brief Append an analysis pass.
  /// \param[in] pass the type of the new analysis pass.
  void AppendAnalysisPass(const std::string &pass);

  /// \brief Visualize the computation graph after each pass by generating a DOT
  /// language file, one can draw them with the Graphviz toolkit.
  void TurnOnDebug();
  /// \brief Human-readable information of the passes.
  std::string DebugString();

  /// \brief Get information of passes.
  /// \return Return list of the passes.
  const std::vector<std::string> &AllPasses() const { return passes_; }

  /// \brief Get information of analysis passes.
  /// \return Return list of analysis passes.
  std::vector<std::string> AnalysisPasses() const {
    auto passes = analysis_passes_;
    // To make sure the ir_graph_to_program should be the last pass so any
    // modication of IR will persist to the program.
    passes.push_back("ir_graph_to_program_pass");
    return passes;
  }

 protected:
  /// \cond Protected
  std::vector<std::string> analysis_passes_{
      {"ir_graph_build_pass", "ir_graph_clean_pass", "ir_analysis_pass",
       "ir_params_sync_among_devices_pass", "adjust_cudnn_workspace_size_pass",
       "inference_op_replace_pass"}};
  std::vector<std::string> passes_;
  /// \endcond
};

/// \class PassStrategy
/// \brief This class defines the pass strategies like whether to use gpu/cuDNN
/// kernel/MKLDNN.
class PassStrategy : public PaddlePassBuilder {
 public:
  /// \brief Constructor of PassStrategy class. It works the same as
  /// PaddlePassBuilder class. \param[in] passes passes' types.
  explicit PassStrategy(const std::vector<std::string> &passes)
      : PaddlePassBuilder(passes) {}

  /// \brief Enable the use of cuDNN kernel.
  virtual void EnableCUDNN() {}

  /// \brief Enable the use of MKLDNN.
  /// The MKLDNN control exists in both CPU and GPU mode, because there can
  /// still be some CPU kernels running in GPU mode.
  virtual void EnableMKLDNN() {}

  /// \brief Enable MKLDNN quantize optimization.
  virtual void EnableMkldnnQuantizer() {}

  /// \brief Check if we are using gpu.
  /// \return A bool variable implying whether we are in gpu mode.
  bool use_gpu() const { return use_gpu_; }

  /// \brief Default destructor.
  virtual ~PassStrategy() = default;

 protected:
  /// \cond Protected
  bool use_gpu_{false};
  bool use_mkldnn_{false};
  /// \endcond
};

/// \class CpuPassStrategy
/// \brief The CPU passes controller, it is used in AnalysisPredictor with CPU
/// mode.
class CpuPassStrategy : public PassStrategy {
 public:
  /// \brief Default constructor of CpuPassStrategy.
  CpuPassStrategy();

  /// \brief Construct by copying another CpuPassStrategy object.
  /// \param[in] other The CpuPassStrategy object we want to copy.
  explicit CpuPassStrategy(const CpuPassStrategy &other)
      : PassStrategy(other.AllPasses()) {
    use_gpu_ = other.use_gpu_;
    use_mkldnn_ = other.use_mkldnn_;
    use_mkldnn_quantizer_ = other.use_mkldnn_quantizer_;
  }
  /// \brief Default destructor.
  virtual ~CpuPassStrategy() = default;

  /// \brief Enable the use of cuDNN kernel.
  void EnableCUDNN() override;

  /// \brief Enable the use of MKLDNN.
  void EnableMKLDNN() override;

  /// \brief Enable MKLDNN quantize optimization.
  void EnableMkldnnQuantizer() override;

 protected:
  /// \cond Protected
  bool use_mkldnn_quantizer_{false};
  /// \endcond
};

/// \class GpuPassStrategy
/// \brief The GPU passes controller, it is used in AnalysisPredictor with GPU
/// mode.
class GpuPassStrategy : public PassStrategy {
 public:
  /// \brief Default constructor of GpuPassStrategy.
  GpuPassStrategy();

  /// \brief Construct by copying another GpuPassStrategy object.
  /// \param[in] other The GpuPassStrategy object we want to copy.
  explicit GpuPassStrategy(const GpuPassStrategy &other)
      : PassStrategy(other.AllPasses()) {
    use_gpu_ = true;
    use_cudnn_ = other.use_cudnn_;
  }

  /// \brief Enable the use of cuDNN kernel.
  void EnableCUDNN() override;

  /// \brief Not supported in GPU mode yet.
  void EnableMKLDNN() override;

  /// \brief Not supported in GPU mode yet.
  void EnableMkldnnQuantizer() override;

  /// \brief Default destructor.
  virtual ~GpuPassStrategy() = default;

 protected:
  /// \cond Protected
  bool use_cudnn_{false};
  /// \endcond
};
/// \brief List of tensorRT subgraph passes.
extern const std::vector<std::string> kTRTSubgraphPasses;

/// \brief List of lite subgraph passes.
extern const std::vector<std::string> kLiteSubgraphPasses;

}  // namespace paddle
