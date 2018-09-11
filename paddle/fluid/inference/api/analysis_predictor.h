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

#include <string>
#include <vector>
#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/inference/analysis/analyzer.h"
#include "paddle/fluid/inference/api/api_impl.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"

namespace paddle {

using inference::analysis::Argument;
using inference::analysis::Analyzer;
using framework::proto::ProgramDesc;
using framework::NaiveExecutor;

/* This predictor is based on the original native predictor with IR and Analysis
 * support. It will optimize IR and Parameters in the runtime.
 * TODO(Superjomn) Replace the Navive predictor?
 */
class AnalysisPredictor : public PaddlePredictor {
 public:
  explicit AnalysisPredictor(const AnalysisConfig &config) : config_(config) {}

  bool Init(const std::shared_ptr<framework::Scope> &parent_scope);

  bool Run(const std::vector<PaddleTensor> &inputs,
           std::vector<PaddleTensor> *output_data,
           int batch_size = -1) override;

  void PrepareFeedFetch();

  void OptimizeInferenceProgram();

  Argument &analysis_argument() { return argument_; }

  std::unique_ptr<PaddlePredictor> Clone() override {
    PADDLE_THROW("Not Implemented");
  }

 protected:
  bool LoadProgramDesc() {
    // Initialize the inference program
    std::unique_ptr<framework::Executor> tmp_exe(
        new framework::Executor(platform::CPUPlace()));
    if (!config_.model_dir.empty()) {
      // Parameters are saved in separate files sited in
      // the specified `dirname`.
      inference_program_ = paddle::inference::Load(
          static_cast<framework::Executor *>(tmp_exe.get()), scope_.get(),
          config_.model_dir);
    } else if (!config_.prog_file.empty() && !config_.param_file.empty()) {
      // All parameters are saved in a single file.
      // The file names should be consistent with that used
      // in Python API `fluid.io.save_inference_model`.
      inference_program_ = paddle::inference::Load(
          static_cast<framework::Executor *>(tmp_exe.get()), scope_.get(),
          config_.prog_file, config_.param_file);
    } else {
      LOG(ERROR) << "fail to load inference model.";
      return false;
    }
    return true;
  }

  bool SetFeed(const std::vector<PaddleTensor> &input_datas,
               framework::Scope *scope);
  bool GetFetch(std::vector<PaddleTensor> *output_data,
                framework::Scope *scope);
  template <typename T>
  void GetFetchOne(const framework::LoDTensor &fetchs,
                   PaddleTensor *output_data);

 private:
  AnalysisConfig config_;
  Argument argument_;
  std::unique_ptr<NaiveExecutor> executor_;
  platform::Place place_;
  std::shared_ptr<framework::Scope> scope_;
  framework::Scope *sub_scope_;
  std::unique_ptr<framework::ProgramDesc> inference_program_;
  std::vector<framework::OpDesc *> feeds_;
  std::map<std::string, size_t> feed_names_;
  std::vector<framework::OpDesc *> fetchs_;
};

}  // namespace paddle
