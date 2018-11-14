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

#define PADDLE_WITHOUT_ANALYSIS_TESTER
#define PADDLE_WITH_TENSORRT_TESTER
#include "paddle/fluid/inference/tests/api/tester_helper.h"

namespace paddle {
namespace inference {

DEFINE_bool(use_tensorrt, true, "Test the performance of TensorRT engine.");
DEFINE_string(prog_filename, "", "Name of model file.");
DEFINE_string(param_filename, "", "Name of parameters file.");

template <typename ConfigType>
ConfigType GetConfig(std::string model_dir, bool use_gpu, int batch_size = -1) {
  ConfigType config;
  if (!FLAGS_prog_filename.empty() && !FLAGS_param_filename.empty()) {
    config.prog_file = model_dir + "/" + FLAGS_prog_filename;
    config.param_file = model_dir + "/" + FLAGS_param_filename;
  } else {
    config.model_dir = model_dir;
  }
  if (use_gpu) {
    config.use_gpu = true;
    config.device = 0;
    config.fraction_of_gpu_memory = 0.45;
  }
  return config;
}

template <>
contrib::MixedRTConfig GetConfig<contrib::MixedRTConfig>(std::string model_dir,
                                                         bool use_gpu,
                                                         int batch_size) {
  contrib::MixedRTConfig config;
  if (!FLAGS_prog_filename.empty() && !FLAGS_param_filename.empty()) {
    config.prog_file = model_dir + "/" + FLAGS_prog_filename;
    config.param_file = model_dir + "/" + FLAGS_param_filename;
  } else {
    config.model_dir = model_dir;
  }
  config.use_gpu = true;
  config.device = 0;
  config.fraction_of_gpu_memory = 0.2;
  if (batch_size > 0) {
    config.max_batch_size = batch_size;
  } else {
    config.max_batch_size = 3;
  }
  return config;
}

void profile(std::string model_dir, bool use_tensorrt) {
  std::vector<std::vector<PaddleTensor>> inputs_all;
  if (!FLAGS_prog_filename.empty() && !FLAGS_param_filename.empty()) {
    SetFakeImageInput(&inputs_all, model_dir, true, FLAGS_prog_filename,
                      FLAGS_param_filename);
  } else {
    SetFakeImageInput(&inputs_all, model_dir, false, "__model__", "");
  }

  std::vector<PaddleTensor> outputs;
  if (use_tensorrt) {
    contrib::MixedRTConfig config =
        GetConfig<contrib::MixedRTConfig>(model_dir, true, FLAGS_batch_size);
    TestPrediction(reinterpret_cast<PaddlePredictor::Config*>(&config),
                   inputs_all, &outputs, FLAGS_num_threads, false,
                   use_tensorrt);
  } else {
    NativeConfig config = GetConfig<NativeConfig>(model_dir, true);
    TestPrediction(reinterpret_cast<PaddlePredictor::Config*>(&config),
                   inputs_all, &outputs, FLAGS_num_threads, false,
                   use_tensorrt);
  }
}

void compare(std::string model_dir) {
  std::vector<std::vector<PaddleTensor>> inputs_all;
  if (!FLAGS_prog_filename.empty() && !FLAGS_param_filename.empty()) {
    SetFakeImageInput(&inputs_all, model_dir, true, FLAGS_prog_filename,
                      FLAGS_param_filename);
  } else {
    SetFakeImageInput(&inputs_all, model_dir, false, "__model__", "");
  }

  std::vector<PaddleTensor> native_outputs;
  NativeConfig native_config = GetConfig<NativeConfig>(model_dir, true);
  TestOneThreadPrediction(
      reinterpret_cast<PaddlePredictor::Config*>(&native_config), inputs_all,
      &native_outputs, false, false);

  std::vector<PaddleTensor> mixedrt_outputs;
  contrib::MixedRTConfig mixedrt_config =
      GetConfig<contrib::MixedRTConfig>(model_dir, true, FLAGS_batch_size);
  TestOneThreadPrediction(
      reinterpret_cast<PaddlePredictor::Config*>(&mixedrt_config), inputs_all,
      &mixedrt_outputs, false, true);

  CompareResult(native_outputs, mixedrt_outputs);
}

TEST(TensorRT_mobilenet, compare) {
  std::string model_dir = FLAGS_infer_model + "/mobilenet";
  compare(model_dir);
}

TEST(TensorRT_resnet50, compare) {
  std::string model_dir = FLAGS_infer_model + "/resnet50";
  compare(model_dir);
}

TEST(TensorRT_resnext50, compare) {
  std::string model_dir = FLAGS_infer_model + "/resnext50";
  compare(model_dir);
}

TEST(TensorRT_resnext50, profile) {
  std::string model_dir = FLAGS_infer_model + "/resnext50";
  profile(model_dir, FLAGS_use_tensorrt);
}

}  // namespace inference
}  // namespace paddle
