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

#ifdef USE_NVINFER_PLUGIN
void run(const AnalysisConfig& config, std::vector<float>* out_data) {
  auto predictor = CreatePaddlePredictor(config);
  auto input_names = predictor->GetInputNames();

  const int run_batch = 2;
  const int run_seq_len = 71;
  const int max_seq_len = 128;

  int32_t i0[run_seq_len] = {
      1,    3558, 4,   75,  491, 89, 340, 313, 93,   4,   255,   10, 75,    321,
      4095, 1902, 4,   134, 49,  75, 311, 14,  44,   178, 543,   15, 12043, 2,
      75,   201,  340, 9,   14,  44, 486, 218, 1140, 279, 12043, 2,
      //
      101, 2054, 2234, 2046, 2486, 2044, 1996, 2047, 4552, 2001, 9536, 1029,  102, 2004, 1997, 2008,
      2154, 1010, 1996, 2047, 4552, 9536, 2075, 1996, 2117, 3072, 2234, 2046, 2486, 1012, 102,
  };
  int32_t i1[run_seq_len] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      //
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  };
  int32_t i2[3] = {0,  40, 71};
  int32_t i3[max_seq_len] = {0};

  // first input
  auto input_t = predictor->GetInputTensor(input_names[0]);
  input_t->Reshape({run_seq_len});
  input_t->copy_from_cpu(i0);

  // second input
  auto input_t2 = predictor->GetInputTensor(input_names[1]);
  input_t2->Reshape({run_seq_len});
  input_t2->copy_from_cpu(i1);

  // third input.
  auto input_t3 = predictor->GetInputTensor(input_names[2]);
  input_t3->Reshape({run_batch + 1});
  input_t3->copy_from_cpu(i2);

  auto input_t4 = predictor->GetInputTensor(input_names[3]);
  input_t4->Reshape({1, max_seq_len, 1});
  input_t4->copy_from_cpu(i3);

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

  int max_batch = 32;
  int min_seq_len = 1;
  int max_seq_len = 128;
  int opt_seq_len = 64;

  //std::string input_name0 = "eval_placeholder_0";
  //std::string input_name1 = "eval_placeholder_1";
  //std::string input_name2 = "eval_placeholder_2";
  //std::string input_name3 = "eval_placeholder_3";

  std::string input_name0 = "read_file_0.tmp_0";
  std::string input_name1 = "read_file_0.tmp_1";
  std::string input_name2 = "read_file_0.tmp_2";
  std::string input_name3 = "read_file_0.tmp_4";

  std::vector<int> min_shape = {min_seq_len};
  std::vector<int> max_shape = {max_seq_len};
  std::vector<int> opt_shape = {opt_seq_len};
  // Set the input's min, max, opt shape
  std::map<std::string, std::vector<int>> min_input_shape = {
      {input_name0, min_shape},
      {input_name1, min_shape},
      {input_name2, {1}},
      {input_name3, {1, 1, 1}}};
  std::map<std::string, std::vector<int>> max_input_shape = {
      {input_name0, max_shape},
      {input_name1, max_shape},
      {input_name2, {max_batch + 1}},
      {input_name3, {1, 128, 1}}};
  std::map<std::string, std::vector<int>> opt_input_shape = {
      {input_name0, opt_shape},
      {input_name1, opt_shape},
      {input_name2, {max_batch + 1}},
      {input_name3, {1, 128, 1}}};

  auto precision = AnalysisConfig::Precision::kFloat32;
  if (with_fp16) {
    std::cout << "fp16 model-------->" << std::endl;
    precision = AnalysisConfig::Precision::kHalf;
  }
  config.EnableTensorRtEngine(1 << 30, 1, 5, precision, false, false);
  config.SetTRTDynamicShapeInfo(min_input_shape, max_input_shape,
                                opt_input_shape);
  std::vector<float> out_data;
  run(config, &out_data);
  for (size_t i = 0; i < out_data.size(); i++) {
    std::cout << "predict out : " << out_data[i] << std::endl;
    //EXPECT_NEAR(result[i], out_data[i], 1e-5);
  }
}

TEST(AnalysisPredictor, fp16) {
  std::vector<float> result = {0.598336, 0.219558, 0.182106};
  trt_ernie(true, result);
}

#else

void run(const AnalysisConfig& config, std::vector<float>* out_data) {
  auto predictor = CreatePaddlePredictor(config);
  auto input_names = predictor->GetInputNames();

  int run_batch = 1;
  const int run_seq_len = 128;

  std::vector<int64_t> tmp_input;
  std::vector<float> tmp_four_input;
  tmp_input.reserve(run_batch * run_seq_len);
  tmp_four_input.reserve(run_batch * run_seq_len);

  int64_t i0[run_seq_len] = {
      1,    3558, 4,   75,  491, 89, 340, 313, 93,   4,   255,   10, 75,    321,
      4095, 1902, 4,   134, 49,  75, 311, 14,  44,   178, 543,   15, 12043, 2,
      75,   201,  340, 9,   14,  44, 486, 218, 1140, 279, 12043, 2};
  int64_t i1[run_seq_len] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  int64_t i2[run_seq_len] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                             10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                             20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                             30, 31, 32, 33, 34, 35, 36, 37, 38, 39};
  float i3[run_seq_len] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

  // first input
  auto input_t = predictor->GetInputTensor(input_names[0]);
  input_t->Reshape({run_batch, run_seq_len, 1});
  input_t->copy_from_cpu(i0);

  // second input
  auto input_t2 = predictor->GetInputTensor(input_names[1]);
  input_t2->Reshape({run_batch, run_seq_len, 1});
  input_t2->copy_from_cpu(i1);

  // third input.
  auto input_t3 = predictor->GetInputTensor(input_names[2]);
  input_t3->Reshape({run_batch, run_seq_len, 1});
  input_t3->copy_from_cpu(i2);

  auto input_t4 = predictor->GetInputTensor(input_names[3]);
  input_t4->Reshape({run_batch, run_seq_len, 1});
  input_t4->copy_from_cpu(i3);

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

  int batch = 1;
  int min_seq_len = 1;
  int max_seq_len = 128;
  int opt_seq_len = 128;

  std::vector<int> min_shape = {batch, min_seq_len, 1};
  std::vector<int> max_shape = {batch, max_seq_len, 1};
  std::vector<int> opt_shape = {batch, opt_seq_len, 1};
  // Set the input's min, max, opt shape
  std::map<std::string, std::vector<int>> min_input_shape = {
      {"read_file_0.tmp_0", min_shape},
      {"read_file_0.tmp_1", min_shape},
      {"read_file_0.tmp_2", min_shape},
      {"matmul_0.tmp_0", {batch, min_seq_len, min_seq_len}}};
  std::map<std::string, std::vector<int>> max_input_shape = {
      {"read_file_0.tmp_0", max_shape},
      {"read_file_0.tmp_1", max_shape},
      {"read_file_0.tmp_2", max_shape},
      {"matmul_0.tmp_0", {batch, max_seq_len, max_seq_len}}};
  std::map<std::string, std::vector<int>> opt_input_shape = {
      {"read_file_0.tmp_0", opt_shape},
      {"read_file_0.tmp_1", opt_shape},
      {"read_file_0.tmp_2", opt_shape},
      {"matmul_0.tmp_0", {batch, opt_seq_len, opt_seq_len}}};

  auto precision = AnalysisConfig::Precision::kFloat32;
  if (with_fp16) {
    precision = AnalysisConfig::Precision::kHalf;
  }
  config.EnableTensorRtEngine(1 << 30, 1, 5, precision, false, false);
  config.SetTRTDynamicShapeInfo(min_input_shape, max_input_shape,
                                opt_input_shape);
  std::vector<float> out_data;
  run(config, &out_data);
  for (size_t i = 0; i < out_data.size(); i++) {
    EXPECT_NEAR(result[i], out_data[i], 1e-5);
  }
}

TEST(AnalysisPredictor, no_fp16) {
  std::vector<float> result = {0.597841, 0.219972, 0.182187};
  trt_ernie(false, result);
}

TEST(AnalysisPredictor, fp16) {
#ifdef SUPPORTS_CUDA_FP16
  std::vector<float> result = {0.598336, 0.219558, 0.182106};
  trt_ernie(true, result);
#endif
}

#endif // end of USE_NVINFER_PLUGIN

}  // namespace inference
}  // namespace paddle
