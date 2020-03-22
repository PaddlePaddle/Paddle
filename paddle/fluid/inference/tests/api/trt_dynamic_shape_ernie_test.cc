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

#include "paddle/fluid/inference/tests/api/trt_test_helper.h"

namespace paddle {
namespace inference {

void run(const AnalysisConfig& config, std::vector<float>* out_data) {
  auto predictor = CreatePaddlePredictor(config);
  auto input_names = predictor->GetInputNames();

  int run_batch = 1;
  int run_seq_len = 5;

  std::vector<int64_t> tmp_input;
  std::vector<float> tmp_four_input;
  tmp_input.reserve(run_batch * run_seq_len);
  tmp_four_input.reserve(run_batch * run_seq_len);

  // first input
  for (int i = 0; i < run_batch * run_seq_len; i++) {
    tmp_input[i] = i % 1800;
  }
  auto input_t = predictor->GetInputTensor(input_names[0]);
  input_t->Reshape({run_batch, run_seq_len, 1});
  input_t->copy_from_cpu(tmp_input.data());

  // second input
  for (int i = 0; i < run_batch * run_seq_len; i++) {
    tmp_input[i] = i % 513;
  }
  auto input_t2 = predictor->GetInputTensor(input_names[1]);
  input_t2->Reshape({run_batch, run_seq_len, 1});
  input_t2->copy_from_cpu(tmp_input.data());

  // third input.
  for (int i = 0; i < run_batch * run_seq_len; i++) {
    tmp_input[i] = i % 3;
  }
  auto input_t3 = predictor->GetInputTensor(input_names[2]);
  input_t3->Reshape({run_batch, run_seq_len, 1});
  input_t3->copy_from_cpu(tmp_input.data());

  for (int i = 0; i < run_batch * run_seq_len; i++) {
    tmp_four_input[i] = (i % 255) / 255.;
  }
  auto input_t4 = predictor->GetInputTensor(input_names[3]);
  input_t4->Reshape({run_batch, run_seq_len, 1});
  input_t4->copy_from_cpu(tmp_four_input.data());

  ASSERT_TRUE(predictor->ZeroCopyRun());

  auto output_names = predictor->GetOutputNames();
  auto output_t = predictor->GetOutputTensor(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                std::multiplies<int>());
  out_data->resize(out_num);
  output_t->copy_to_cpu(out_data->data());
}

void trt_ernie(bool with_fp16, std::vector<float> result) {
  AnalysisConfig config;
  std::string model_dir = FLAGS_infer_model;
  SetConfig(&config, model_dir, true /* use_gpu */);

  config.SwitchUseFeedFetchOps(false);

  int head_number = 12;
  int batch = 1;
  int min_seq_len = 1;
  int max_seq_len = 128;
  int opt_seq_len = 128;

  std::vector<int> min_shape = {batch, min_seq_len, 1};
  std::vector<int> max_shape = {batch, max_seq_len, 1};
  std::vector<int> opt_shape = {batch, opt_seq_len, 1};
  // Set the input's min, max, opt shape
  std::map<std::string, std::vector<int>> min_input_shape = {
      {"placeholder_0", min_shape},
      {"placeholder_1", min_shape},
      {"placeholder_2", min_shape},
      {"stack_0.tmp_0", {batch, head_number, min_seq_len, min_seq_len}}};
  std::map<std::string, std::vector<int>> max_input_shape = {
      {"placeholder_0", max_shape},
      {"placeholder_1", max_shape},
      {"placeholder_2", max_shape},
      {"stack_0.tmp_0", {batch, head_number, max_seq_len, max_seq_len}}};
  std::map<std::string, std::vector<int>> opt_input_shape = {
      {"placeholder_0", opt_shape},
      {"placeholder_1", opt_shape},
      {"placeholder_2", opt_shape},
      {"stack_0.tmp_0", {batch, head_number, opt_seq_len, opt_seq_len}}};

  auto precision = AnalysisConfig::Precision::kFloat32;
  if (with_fp16) {
    precision = AnalysisConfig::Precision::kHalf;
  }
  config.EnableTensorRtEngine(1 << 30, 1, 5, precision, false, true,
                              min_input_shape, max_input_shape,
                              opt_input_shape);
  std::vector<float> out_data;
  run(config, &out_data);
  for (size_t i = 0; i < out_data.size(); i++) {
    EXPECT_NEAR(result[i], out_data[i], 1e-6);
  }
}

TEST(AnalysisPredictor, no_fp16) {
  std::vector<float> result = {-0.43203, 0.771308, -0.72201};
  trt_ernie(false, result);
}

/*
TEST(AnalysisPredictor, fp16) {
#ifdef SUPPORT_CUDA_FP16
 std::vector<float> result = {-0.43203, 0.771308, -0.72201};
 trt_ernie(true, result);
#endif
}
*/

}  // namespace inference
}  // namespace paddle
