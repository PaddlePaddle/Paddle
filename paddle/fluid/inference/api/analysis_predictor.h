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
#include <string>
#include <vector>
#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/inference/analysis/analyzer.h"
#include "paddle/fluid/inference/api/api_impl.h"
#include "paddle/fluid/inference/api/details/reset_tensor_array.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/string/printf.h"

namespace paddle {

using inference::analysis::Argument;
using inference::analysis::Analyzer;
using framework::proto::ProgramDesc;
using framework::NaiveExecutor;
using contrib::AnalysisConfig;

/* This predictor is based on the original native predictor with IR and Analysis
 * support. It will optimize IR and Parameters in the runtime.
 * TODO(Superjomn) Replace the Navive predictor?
 */
class AnalysisPredictor : public PaddlePredictor {
 public:
  explicit AnalysisPredictor(const AnalysisConfig &config) : config_(config) {}

  bool Init(const std::shared_ptr<framework::Scope> &parent_scope,
            const std::shared_ptr<framework::ProgramDesc> &program = nullptr);

  bool Run(const std::vector<PaddleTensor> &inputs,
           std::vector<PaddleTensor> *output_data,
           int batch_size = -1) override;

  std::unique_ptr<ZeroCopyTensor> GetInputTensor(
      const std::string &name) override;
  std::unique_ptr<ZeroCopyTensor> GetOutputTensor(
      const std::string &name) override;

  bool ZeroCopyRun() override;

  void PrepareFeedFetch();

  void OptimizeInferenceProgram();

  Argument &analysis_argument() { return argument_; }

  std::unique_ptr<PaddlePredictor> Clone() override;

  framework::Scope *scope() { return executor_->scope(); }
  framework::ProgramDesc &program() { return *inference_program_; }

 protected:
  bool LoadProgramDesc();

  bool SetFeed(const std::vector<PaddleTensor> &input_datas,
               framework::Scope *scope);
  bool GetFetch(std::vector<PaddleTensor> *output_data,
                framework::Scope *scope);
  template <typename T>
  void GetFetchOne(const framework::LoDTensor &fetchs,
                   PaddleTensor *output_data);
  ~AnalysisPredictor();

 private:
  contrib::AnalysisConfig config_;
  Argument argument_;
  std::unique_ptr<NaiveExecutor> executor_;
  platform::Place place_;
  std::shared_ptr<framework::Scope> scope_;
  framework::Scope *sub_scope_{nullptr};
  std::shared_ptr<framework::ProgramDesc> inference_program_;
  std::vector<framework::OpDesc *> feeds_;
  std::map<std::string, size_t> feed_names_;
  std::vector<framework::OpDesc *> fetchs_;
  // Memory buffer for feed inputs. The temporary LoDTensor will cause serious
  // concurrency problems, so cache them.
  std::vector<framework::LoDTensor> feed_tensors_;
  details::TensorArrayBatchCleaner tensor_array_batch_cleaner_;
};

}  // namespace paddle
