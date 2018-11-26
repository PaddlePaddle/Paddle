/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "paddle/fluid/inference/tests/api/tester_helper.h"

namespace paddle {
namespace inference {

DEFINE_bool(use_tensorrt, true, "Test the performance of TensorRT engine.");
DEFINE_string(prog_filename, "", "Name of model file.");
DEFINE_string(param_filename, "", "Name of parameters file.");

template <typename ConfigType>
void SetConfig(ConfigType* config, std::string model_dir, bool use_gpu,
               bool use_tensorrt = false, int batch_size = -1) {
  if (!FLAGS_prog_filename.empty() && !FLAGS_param_filename.empty()) {
    config->prog_file = model_dir + "/" + FLAGS_prog_filename;
    config->param_file = model_dir + "/" + FLAGS_param_filename;
  } else {
    config->model_dir = model_dir;
  }
  if (use_gpu) {
    config->use_gpu = true;
    config->device = 0;
    config->fraction_of_gpu_memory = 0.15;
  }
}

template <>
void SetConfig<contrib::AnalysisConfig>(contrib::AnalysisConfig* config,
                                        std::string model_dir, bool use_gpu,
                                        bool use_tensorrt, int batch_size) {
  if (!FLAGS_prog_filename.empty() && !FLAGS_param_filename.empty()) {
    config->prog_file = model_dir + "/" + FLAGS_prog_filename;
    config->param_file = model_dir + "/" + FLAGS_param_filename;
  } else {
    config->model_dir = model_dir;
  }
  if (use_gpu) {
    config->use_gpu = true;
    config->device = 0;
    config->fraction_of_gpu_memory = 0.15;
    if (use_tensorrt) {
      config->EnableTensorRtEngine(1 << 10, batch_size);
      config->pass_builder()->DeletePass("conv_bn_fuse_pass");
      config->pass_builder()->DeletePass("fc_fuse_pass");
      config->pass_builder()->TurnOnDebug();
    } else {
      config->enable_ir_optim = true;
    }
  }
}

void profile(std::string model_dir, bool use_analysis, bool use_tensorrt) {
  std::vector<std::vector<PaddleTensor>> inputs_all;
  if (!FLAGS_prog_filename.empty() && !FLAGS_param_filename.empty()) {
    SetFakeImageInput(&inputs_all, model_dir, true, FLAGS_prog_filename,
                      FLAGS_param_filename);
  } else {
    SetFakeImageInput(&inputs_all, model_dir, false, "__model__", "");
  }

  std::vector<PaddleTensor> outputs;
  if (use_analysis || use_tensorrt) {
    contrib::AnalysisConfig config(true);
    SetConfig<contrib::AnalysisConfig>(&config, model_dir, true, use_tensorrt,
                                       FLAGS_batch_size);
    TestPrediction(reinterpret_cast<PaddlePredictor::Config*>(&config),
                   inputs_all, &outputs, FLAGS_num_threads, true);
  } else {
    NativeConfig config;
    SetConfig<NativeConfig>(&config, model_dir, true, false);
    TestPrediction(reinterpret_cast<PaddlePredictor::Config*>(&config),
                   inputs_all, &outputs, FLAGS_num_threads, false);
  }
}

void compare(std::string model_dir, bool use_tensorrt) {
  std::vector<std::vector<PaddleTensor>> inputs_all;
  if (!FLAGS_prog_filename.empty() && !FLAGS_param_filename.empty()) {
    SetFakeImageInput(&inputs_all, model_dir, true, FLAGS_prog_filename,
                      FLAGS_param_filename);
  } else {
    SetFakeImageInput(&inputs_all, model_dir, false, "__model__", "");
  }

  std::vector<PaddleTensor> native_outputs;
  NativeConfig native_config;
  SetConfig<NativeConfig>(&native_config, model_dir, true, false,
                          FLAGS_batch_size);
  TestOneThreadPrediction(
      reinterpret_cast<PaddlePredictor::Config*>(&native_config), inputs_all,
      &native_outputs, false);

  std::vector<PaddleTensor> analysis_outputs;
  contrib::AnalysisConfig analysis_config(true);
  SetConfig<contrib::AnalysisConfig>(&analysis_config, model_dir, true,
                                     use_tensorrt, FLAGS_batch_size);
  TestOneThreadPrediction(
      reinterpret_cast<PaddlePredictor::Config*>(&analysis_config), inputs_all,
      &analysis_outputs, true);

  CompareResult(native_outputs, analysis_outputs);
}

TEST(TensorRT_mobilenet, compare) {
  std::string model_dir = FLAGS_infer_model + "/mobilenet";
  compare(model_dir, /* use_tensorrt */ true);
}

TEST(TensorRT_resnet50, compare) {
  std::string model_dir = FLAGS_infer_model + "/resnet50";
  compare(model_dir, /* use_tensorrt */ true);
}

TEST(TensorRT_resnext50, compare) {
  std::string model_dir = FLAGS_infer_model + "/resnext50";
  compare(model_dir, /* use_tensorrt */ true);
}

TEST(TensorRT_resnext50, profile) {
  std::string model_dir = FLAGS_infer_model + "/resnext50";
  profile(model_dir, /* use_analysis */ true, FLAGS_use_tensorrt);
}

TEST(TensorRT_mobilenet, analysis) {
  std::string model_dir = FLAGS_infer_model + "/" + "mobilenet";
  compare(model_dir, /* use_tensorrt */ false);
}

}  // namespace inference
}  // namespace paddle
