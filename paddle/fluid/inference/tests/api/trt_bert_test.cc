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

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "paddle/fluid/inference/tests/api/trt_test_helper.h"

namespace paddle {
namespace inference {

TEST(TensorRT, split_converter) {
  AnalysisConfig config;
  int batch_size = 1;
  config.SetModel(FLAGS_infer_model);
  config.EnableUseGpu(1200, 0);
  config.SwitchUseFeedFetchOps(false);
  config.EnableTensorRtEngine(1 << 30, batch_size, 10,
                              AnalysisConfig::Precision::kFloat32, false,
                              false);
  auto predictor = CreatePaddlePredictor<AnalysisConfig>(config);
  int64_t i0[128] = {
      96,  54,  78, 37,  106, 35,  122, 33,  95, 63,  81, 60, 65, 68,  45, 96,
      117, 61,  43, 15,  12,  64,  91,  100, 90, 74,  99, 23, 22, 91,  83, 13,
      28,  71,  59, 15,  40,  26,  66,  18,  31, 87,  85, 11, 55, 67,  28, 126,
      7,   89,  39, 67,  88,  29,  66,  38,  98, 1,   66, 38, 95, 56,  48, 95,
      9,   38,  90, 82,  101, 6,   75,  46,  42, 89,  98, 12, 6,  101, 82, 55,
      81,  113, 33, 91,  44,  73,  41,  39,  12, 113, 13, 86, 36, 91,  53, 68,
      103, 67,  65, 92,  27,  76,  24,  107, 54, 94,  63, 10, 15, 32,  91, 45,
      37,  126, 49, 118, 73,  127, 122, 119, 28, 96,  92, 79, 21, 90,  11, 40};
  int64_t i1[128] = {
      0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,
      15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
      30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,
      45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
      60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,
      75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,
      90,  91,  92,  93,  94,  95,  96,  97,  98,  99,  100, 101, 102, 103, 104,
      105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
      120, 121, 122, 123, 124, 125, 126, 127};
  int64_t i2[128] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  float i3[128 * 128] = {0.0};
  int64_t i4[1] = {0};

  auto input_names = predictor->GetInputNames();

  auto input_t0 = predictor->GetInputTensor(input_names[0]);
  input_t0->Reshape({batch_size, 128, 1});
  input_t0->copy_from_cpu(i0);
  auto input_t1 = predictor->GetInputTensor(input_names[1]);
  input_t1->Reshape({batch_size, 128, 1});
  input_t1->copy_from_cpu(i1);
  auto input_t2 = predictor->GetInputTensor(input_names[2]);
  input_t2->Reshape({batch_size, 128, 1});
  input_t2->copy_from_cpu(i2);
  auto input_t3 = predictor->GetInputTensor(input_names[3]);
  input_t3->Reshape({batch_size, 128, 128});
  input_t3->copy_from_cpu(i3);
  auto input_t4 = predictor->GetInputTensor(input_names[4]);
  input_t4->Reshape({batch_size, 1});
  input_t4->copy_from_cpu(i4);

  ASSERT_TRUE(predictor->ZeroCopyRun());
}

}  // namespace inference
}  // namespace paddle
