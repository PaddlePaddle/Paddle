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

#include "paddle/fluid/inference/api/mkldnn_analysis_predictor.h"
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/api/paddle_inference_pass.h"
#include "paddle/fluid/inference/utils/singleton.h"
#include "paddle/fluid/platform/profiler.h"

DECLARE_bool(profile);

namespace paddle {

// Check environment flag whether MKL-DNN should be used
static bool IsMKLDNNSetOn() {
  const char* flag = std::getenv("FLAGS_use_mkldnn");
  if (flag) {
    std::string flag_str(flag);
    std::transform(flag_str.begin(), flag_str.end(), flag_str.begin(),
                   ::toupper);
    return !flag_str.compare("ON") || !flag_str.compare("TRUE") ||
           !flag_str.compare("1");
  }
  return false;
}

bool MKLDNNAnalysisPredictor::Init(
    const std::shared_ptr<framework::Scope>& parent_scope) {
  VLOG(3) << "Predictor::init()";
#if !defined(_WIN32)
  if (FLAGS_profile) {
    LOG(WARNING) << "Profiler is actived, might affect the performance";
    LOG(INFO) << "You can turn off by set gflags '-profile false'";
    auto tracking_device = config_.use_gpu ? platform::ProfilerState::kAll
                                           : platform::ProfilerState::kCPU;
    platform::EnableProfiler(tracking_device);
  }
#endif

  if (config_.use_gpu) {
    place_ = paddle::platform::CUDAPlace(config_.device);
    LOG(WARNING) << "ir optimize only supports CPU currently";
    config_.enable_ir_optim = false;
  } else {
    place_ = paddle::platform::CPUPlace();
  }
  if (parent_scope) {
    scope_ = parent_scope;
    sub_scope_ = &(parent_scope->NewScope());
  } else {
    paddle::framework::InitDevices(false);
    scope_.reset(new paddle::framework::Scope());
  }

  executor_.reset(new paddle::framework::Executor(place_));

  // Initialize the inference program
  if (!config_.model_dir.empty()) {
    // Parameters are saved in separate files sited in
    // the specified `dirname`.
    inference_program_ = paddle::inference::Load(executor_.get(), scope_.get(),
                                                 config_.model_dir);
  } else if (!config_.prog_file.empty() && !config_.param_file.empty()) {
    // All parameters are saved in a single file.
    // The file names should be consistent with that used
    // in Python API `fluid.io.save_inference_model`.
    inference_program_ = paddle::inference::Load(
        executor_.get(), scope_.get(), config_.prog_file, config_.param_file);
  } else {
    LOG(ERROR) << "fail to load inference model.";
    return false;
  }

  if (IsMKLDNNSetOn()) {
    LOG(INFO) << "MKL-DNN enabled";
    config_.use_mkldnn = true;
    executor_->EnableMKLDNN(*inference_program_);
  }

  OptimizeInferenceProgram();
  ctx_ = executor_->Prepare(*inference_program_, 0);

  VLOG(5) << "to create variables";
  PADDLE_ENFORCE(scope_.get());
  executor_->CreateVariables(*inference_program_,
                             sub_scope_ ? sub_scope_ : scope_.get(), 0);
  // Get the feed_target_names and fetch_target_names
  PrepareFeedFetch();
  return true;
}

void MKLDNNAnalysisPredictor::OptimizeInferenceProgram() {
  LOG(INFO) << "== optimize begin ==";
  FLAGS_IA_enable_ir = config_.enable_ir_optim;
  FLAGS_IA_enable_tensorrt_subgraph_engine = false;
  FLAGS_IA_output_storage_path = "";  // Don't output the model.
  // Analyze inference_program
  if (!config_.model_dir.empty()) {
    analysis_argument().fluid_model_dir.reset(
        new std::string(config_.model_dir));
  } else {
    PADDLE_ENFORCE(
        !config_.param_file.empty(),
        "Either model_dir or (param_file, prog_file) should be set.");
    PADDLE_ENFORCE(!config_.prog_file.empty());
    analysis_argument().fluid_model_program_path.reset(
        new std::string(config_.prog_file));
    analysis_argument().fluid_model_param_path.reset(
        new std::string(config_.param_file));
  }
  analysis_argument().origin_program_desc.reset(
      new ProgramDesc(*inference_program_->Proto()));

  PADDLE_ENFORCE(config_.ir_mode == AnalysisConfig::IrPassMode::kInclude,
                 "Only kInclude is supported with MKLDNNAnalysisPredictor.");
  if (config_.use_mkldnn)
    MKLDNNAnalyzer()
        .SetIrPasses(config_.ir_mkldnn_passes)
        .Run(&analysis_argument());
  else
    MKLDNNAnalyzer().SetIrPasses(config_.ir_passes).Run(&analysis_argument());

  CHECK(analysis_argument().transformed_program_desc);
  VLOG(5) << "to prepare executor";
  inference_program_.reset(new framework::ProgramDesc(
      *analysis_argument().transformed_program_desc));
  if (analysis_argument().Has(framework::ir::kParamScopeAttr)) {
    // Update scope.
    scope_.reset(analysis_argument().Release<framework::Scope>(
        framework::ir::kParamScopeAttr));
  }
  LOG(INFO) << "== optimize end ==";
}

template <>
std::unique_ptr<PaddlePredictor>
CreatePaddlePredictor<MKLDNNAnalysisConfig, PaddleEngineKind::kAnalysis>(
    const MKLDNNAnalysisConfig& config) {
  VLOG(3) << "create MKLDNNAnalysisConfig";
  std::unique_ptr<PaddlePredictor> predictor(
      new MKLDNNAnalysisPredictor(config));
  if (!dynamic_cast<MKLDNNAnalysisPredictor*>(predictor.get())->Init(nullptr)) {
    return nullptr;
  }
  return predictor;
}

}  // namespace paddle
