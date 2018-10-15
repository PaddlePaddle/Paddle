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

#include "paddle/fluid/inference/analysis/analyzer.h"
#include "paddle/fluid/inference/api/api_impl.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/utils/singleton.h"
#include "paddle/fluid/operators/tensorrt_engine_op.h"

namespace paddle {

using inference::analysis::Argument;
using inference::Singleton;
using inference::analysis::Analyzer;
using framework::proto::ProgramDesc;
using paddle::contrib::MixedRTConfig;

class TensorRTSubgraphPredictor : public NativePaddlePredictor {
 public:
  explicit TensorRTSubgraphPredictor(const MixedRTConfig& config)
      : NativePaddlePredictor(config), config_(config) {}

  bool Init(const std::shared_ptr<framework::Scope>& parent_scope) {
    FLAGS_IA_enable_tensorrt_subgraph_engine = true;
    VLOG(3) << "Predictor::init()";
    if (config_.use_gpu) {
      place_ = paddle::platform::CUDAPlace(config_.device);
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
      inference_program_ = paddle::inference::Load(
          executor_.get(), scope_.get(), config_.model_dir);
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

    OptimizeInferenceProgram();
    ctx_ = executor_->Prepare(*inference_program_, 0);

    VLOG(5) << "to create variables";
    executor_->CreateVariables(*inference_program_,
                               sub_scope_ ? sub_scope_ : scope_.get(), 0);
    // Get the feed_target_names and fetch_target_names
    PrepareFeedFetch();
    return true;
  }

  bool Run(const std::vector<PaddleTensor>& inputs,
           std::vector<PaddleTensor>* output_data,
           int batch_size = -1) override {
    PADDLE_ENFORCE_GT(batch_size, 0,
                      "TensorRT engine needs the argument batch_size set");
    FLAGS_tensorrt_engine_batch_size = batch_size;
    return NativePaddlePredictor::Run(inputs, output_data, batch_size);
  }

  void OptimizeInferenceProgram() {
    // Analyze inference_program
    Argument argument;

    argument.Set<int>("minimum_subgraph_size",
                      new int(config_.minimum_subgraph_size));
    argument.Set<int>("max_batch_size", new int(config_.max_batch_size));
    argument.Set<int>("workspace_size", new int(config_.workspace_size));
    argument.Set<std::string>("precision_mode",
                              new std::string(config_.precision_mode));

    if (!config_.model_dir.empty()) {
      argument.fluid_model_dir.reset(new std::string(config_.model_dir));
    } else {
      PADDLE_ENFORCE(
          !config_.param_file.empty(),
          "Either model_dir or (param_file, prog_file) should be set.");
      PADDLE_ENFORCE(!config_.prog_file.empty());
      argument.fluid_model_program_path.reset(
          new std::string(config_.prog_file));
      argument.fluid_model_param_path.reset(
          new std::string(config_.param_file));
    }
    argument.origin_program_desc.reset(
        new ProgramDesc(*inference_program_->Proto()));
    Singleton<Analyzer>::Global().Run(&argument);
    CHECK(argument.transformed_program_desc);
    VLOG(5) << "transformed program:\n"
            << argument.transformed_program_desc->SerializeAsString();
    VLOG(5) << "to prepare executor";
    inference_program_.reset(
        new framework::ProgramDesc(*argument.transformed_program_desc));
  }

 private:
  MixedRTConfig config_;
};

template <>
std::unique_ptr<PaddlePredictor>
CreatePaddlePredictor<MixedRTConfig, PaddleEngineKind::kAutoMixedTensorRT>(
    const MixedRTConfig& config) {
  VLOG(3) << "create TensorRTSubgraphPredictor";
  if (config.use_gpu) {
    // 1. GPU memeroy
    PADDLE_ENFORCE_GT(
        config.fraction_of_gpu_memory, 0.f,
        "fraction_of_gpu_memory in the config should be set to range (0., 1.]");
    PADDLE_ENFORCE_GE(config.device, 0, "Invalid device id %d", config.device);
    std::vector<std::string> flags;
    if (config.fraction_of_gpu_memory >= 0.0f ||
        config.fraction_of_gpu_memory <= 0.95f) {
      flags.push_back("dummpy");
      std::string flag = "--fraction_of_gpu_memory_to_use=" +
                         std::to_string(config.fraction_of_gpu_memory);
      flags.push_back(flag);
      VLOG(3) << "set flag: " << flag;
      framework::InitGflags(flags);
    }
  }

  std::unique_ptr<PaddlePredictor> predictor(
      new TensorRTSubgraphPredictor(config));
  if (!dynamic_cast<TensorRTSubgraphPredictor*>(predictor.get())
           ->Init(nullptr)) {
    return nullptr;
  }
  return std::move(predictor);
}

template <>
std::unique_ptr<PaddlePredictor> CreatePaddlePredictor<MixedRTConfig>(
    const MixedRTConfig& config) {
  return CreatePaddlePredictor<MixedRTConfig,
                               PaddleEngineKind::kAutoMixedTensorRT>(config);
}

}  // namespace paddle

USE_TRT_CONVERTER(elementwise_add_weight);
USE_TRT_CONVERTER(elementwise_add_tensor);
USE_TRT_CONVERTER(elementwise_sub_tensor);
USE_TRT_CONVERTER(elementwise_div_tensor);
USE_TRT_CONVERTER(elementwise_mul_tensor);
USE_TRT_CONVERTER(elementwise_max_tensor);
USE_TRT_CONVERTER(elementwise_min_tensor);
USE_TRT_CONVERTER(elementwise_pow_tensor);
USE_TRT_CONVERTER(mul);
USE_TRT_CONVERTER(conv2d);
USE_TRT_CONVERTER(relu);
USE_TRT_CONVERTER(sigmoid);
USE_TRT_CONVERTER(tanh);
USE_TRT_CONVERTER(fc);
USE_TRT_CONVERTER(pool2d);
USE_TRT_CONVERTER(softmax);
USE_TRT_CONVERTER(batch_norm);
USE_TRT_CONVERTER(concat);
USE_TRT_CONVERTER(dropout);
USE_TRT_CONVERTER(pad);
