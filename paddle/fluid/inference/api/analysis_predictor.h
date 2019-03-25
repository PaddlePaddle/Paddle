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
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/inference/analysis/analyzer.h"
#include "paddle/fluid/inference/api/api_impl.h"
#include "paddle/fluid/inference/api/details/reset_tensor_array.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/string/printf.h"
#ifdef PADDLE_WITH_TESTING
#include <gtest/gtest.h>
#include <gtest/gtest_prod.h>
#endif
namespace paddle {

using inference::analysis::Argument;
using inference::analysis::Analyzer;
using framework::proto::ProgramDesc;
using framework::NaiveExecutor;

/*
 * Map variable name to tensor of scaling factors scaling it to MAX=1.0.
 * bool denotes whether quantization of the variable should be done to unsigned
 * type.
 */
using VarQuantScale =
    std::unordered_map<std::string, std::pair<bool, framework::LoDTensor>>;

/** \brief This predictor is based on the original native predictor with IR and
 * Analysis support.
 *
 * It will optimize IR and Parameters in the runtime.
 *
 * TODO(Superjomn) Replace the Navive predictor?
 */
class AnalysisPredictor : public PaddlePredictor {
 public:
  explicit AnalysisPredictor(const AnalysisConfig &config) : config_(config) {}
  ~AnalysisPredictor();

  bool Init(const std::shared_ptr<framework::Scope> &parent_scope,
            const std::shared_ptr<framework::ProgramDesc> &program = nullptr);

  bool Run(const std::vector<PaddleTensor> &inputs,
           std::vector<PaddleTensor> *output_data,
           int batch_size = -1) override;

  std::vector<std::string> GetInputNames();
  std::vector<std::string> GetOutputNames();

  std::unique_ptr<ZeroCopyTensor> GetInputTensor(
      const std::string &name) override;
  std::unique_ptr<ZeroCopyTensor> GetOutputTensor(
      const std::string &name) override;

  bool ZeroCopyRun() override;

  void CreateFeedFetchVar(framework::Scope *scope);
  void PrepareFeedFetch();

  void PrepareArgument();
  void OptimizeInferenceProgram();

  Argument &analysis_argument() { return argument_; }

  std::unique_ptr<PaddlePredictor> Clone() override;

  framework::Scope *scope() { return scope_.get(); }
  framework::ProgramDesc &program() { return *inference_program_; }

  void SetMkldnnThreadID(int tid);

  std::string GetSerializedProgram() const override;

  bool Quantize();

 protected:
  // For memory optimization.
  bool need_collect_var_shapes_for_memory_optim();
  void CollectVarShapes();
  void SerializeBatchVarShapes(const std::string &path);

  bool PrepareProgram(const std::shared_ptr<framework::ProgramDesc> &program);
  bool PrepareScope(const std::shared_ptr<framework::Scope> &parent_scope);
  bool CreateExecutor();
  bool PrepareExecutor();

  bool LoadProgramDesc();
  bool LoadParameters();

  bool SetFeed(const std::vector<PaddleTensor> &input_datas,
               framework::Scope *scope);
  bool GetFetch(std::vector<PaddleTensor> *output_data,
                framework::Scope *scope);
  template <typename T>
  void GetFetchOne(const framework::LoDTensor &fetchs,
                   PaddleTensor *output_data);

#if PADDLE_WITH_TENSORRT
  // When we use Paddle-TRT INT8 engine, we need to generate calibration table
  // data first,
  // the calibration table contains the range for each op's input and output,
  // this whole process can be divided into several steps:
  //
  // 1. Builds a 32-bit engine, runs it on the calibration set, and records a
  // histogram for each
  // tensor of the distribution of activation values.
  // 2. Builds a calibration table from the histograms.
  //
  // After step 2, we need to store the calibration table on disk
  bool SaveTrtCalibToDisk();
#endif

// Some more detailed tests, they are made the friends of the predictor, so that
// the all the details can be tested.
#if PADDLE_WITH_TESTING
  FRIEND_TEST(AnalysisPredictor, analysis_off);
  FRIEND_TEST(AnalysisPredictor, analysis_on);
  FRIEND_TEST(AnalysisPredictor, with_gpu);

  friend class QuantizerTest;
#endif

 private:
#if PADDLE_WITH_MKLDNN
  // Helper class to perform quantization
  class Quantizer;
#endif

 private:
  AnalysisConfig config_;
  Argument argument_;
  std::unique_ptr<NaiveExecutor> executor_;
  platform::Place place_;
  std::shared_ptr<framework::Scope> scope_;
  framework::Scope *sub_scope_{nullptr};
  std::shared_ptr<framework::ProgramDesc> inference_program_;
  std::vector<framework::OpDesc *> feeds_;
  std::map<std::string, size_t> feed_names_;
  // Sorted according to the idx.
  std::map<size_t, std::string> idx2feeds_;
  std::vector<framework::OpDesc *> fetches_;
  std::map<size_t, std::string> idx2fetches_;
#if PADDLE_WITH_MKLDNN
  std::unique_ptr<Quantizer> quantizer_;
#endif

  // Memory buffer for feed inputs. The temporary LoDTensor will cause serious
  // concurrency problems, wrong results and memory leak, so cache them.
  std::vector<framework::LoDTensor> feed_tensors_;
  details::TensorArrayBatchCleaner tensor_array_batch_cleaner_;
  // A mutex help to make Clone thread safe.
  std::mutex clone_mutex_;

  // For memory optimization.
  const size_t max_shape_collect_count_{1000};
  int need_collect_var_shapes_{-1};  // -1 for default, 0 for false, 1 for true.
  std::vector<std::map<std::string, std::vector<int>>> batch_var_shapes_;

 private:
  // Some status here that help to determine the status inside the predictor.
  bool status_program_optimized_{false};
  bool status_is_cloned_{false};
  bool status_use_gpu_{false};
  bool status_ir_optim_enabled_{false};
};

#if PADDLE_WITH_MKLDNN
class AnalysisPredictor::Quantizer {
 public:
  explicit Quantizer(AnalysisPredictor &predictor,  // NOLINT
                     const std::shared_ptr<QuantizerConfig> &qconfig)
      : predictor_(predictor), qconfig_(qconfig) {}

  // Execute full quantization procedure.
  bool Quantize();

#if PADDLE_WITH_TESTING
  friend class QuantizerTest;
#endif

 private:
  // Run single warmup iteration
  bool RunWarmup() const;
  // Gather data from variables and calculate scales for them.
  bool CalculateScales();
  // Calculate a scale for tensor based on ScaleAlgo rules.
  void CalculateSingleScale(const std::string &op_name,
                            const std::string &conn_name,
                            const std::string &var_name,
                            const framework::LoDTensor &var_tensor,
                            bool is_unsigned);
  void PrepareArgument() const;
  bool RunQuantizePasses() const;

  std::vector<int> ExpandQuantizedBins(std::vector<int> quantized_bins,
                                       std::vector<int> reference_bins) const;

  // Using the KL-divergence method get the most precise scaling factor.
  std::pair<bool, framework::LoDTensor> GetKLScalingFactor(
      const framework::LoDTensor &var_tensor, bool is_unsigned) const;

  std::pair<bool, framework::LoDTensor> GetMaxChScalingFactor(
      const framework::LoDTensor &var_tensor, bool is_unsigned) const;

  std::pair<bool, framework::LoDTensor> GetMaxScalingFactor(
      const framework::LoDTensor &var_tensor, bool is_unsigned) const;

  // Returns histogram and bin width
  std::pair<std::vector<int>, float> Histogram(
      const framework::LoDTensor &var_tensor, float min_val, float max_val,
      size_t num_bins = 2048) const;

  // Calculate the entropy.
  float SafeEntropy(std::vector<int> reference_distr_P, int P_sum,
                    std::vector<int> candidate_distr_Q, int Q_sum) const;

 private:
  AnalysisPredictor &predictor_;
  const std::shared_ptr<QuantizerConfig> qconfig_;

  // A map: variable name -> scale
  VarQuantScale scales_;
};
#endif

}  // namespace paddle
