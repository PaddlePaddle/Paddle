/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "paddle/common/flags.h"
#include "test/cpp/inference/api/trt_test_helper.h"

namespace paddle {
namespace inference {

TEST(TensorRT, split_converter) {
  std::string model_dir = FLAGS_infer_model + "/split_converter";
  std::string opt_cache_dir = model_dir + "/_opt_cache";
  delete_cache_files(opt_cache_dir);

  AnalysisConfig config;
  int batch_size = 4;
  int channels = 4;
  int height = 4;
  int width = 4;
  config.EnableUseGpu(100, 0);
  config.SetModel(model_dir);
  config.EnableTensorRtEngine(
      1 << 20, batch_size, 1, AnalysisConfig::Precision::kInt8, false, true);

  std::map<std::string, std::vector<int>> input_shape;
  input_shape["x"] = {batch_size, channels, height, width};
  config.SetTRTDynamicShapeInfo(input_shape, input_shape, input_shape, false);
  auto predictor = CreatePaddlePredictor(config);
}

}  // namespace inference
}  // namespace paddle
