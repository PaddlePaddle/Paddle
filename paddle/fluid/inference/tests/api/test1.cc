// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
//
// See the License for the specific language governing permissions and
// limitations under the License.

#include <glog/logging.h>
#include <thread>
#include "gflags/gflags.h"
#include "gtest/gtest.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"

DEFINE_string(model, "", "");
DEFINE_int32(batch_size, 1, "");

namespace paddle {

const int kTimeMaxSentNum = 30;
const int kPositionFeatTypeNum = 20;

void PrepareWordInput(PaddleTensor* tensor) {
  tensor->shape.assign({1, 4, 40, 1});
  std::vector<int> datas(
      {30614, 39264, 49386, 16129, 32130, 2,     2,     36761, 2,     12286,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,  //
       2,     44317, 44317, 51311, 42921, 10516, 15747, 0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,  //
       29210, 12088, 19825, 2,     41007, 46655, 22153, 349,   2,     50293,
       1520,  40597, 16429, 15747, 0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,  //
       43550, 49852, 24901, 2,     10772, 40597, 30614, 36052, 32130, 25622,
       30614, 28250, 24809, 32130, 2,     21931, 52075, 24346, 2,     26732,
       28250, 17322, 32130, 50961, 466,   47480, 2,     7135,  47480, 6689,
       25709, 49716, 40597, 2,     32130, 37811, 3977,  24826, 37042, 15747});
  auto* data = tensor->mutable_data<int64_t>();
  for (int i = 0; i < datas.size(); i++) {
    data[i] = datas[i];
  }
}

void PreparePostag(PaddleTensor* tensor) {
  tensor->shape.assign({1, 4, 40, 1});
  std::vector<int> datas({
      7,  7, 19, 17, 4,  3,  8, 19, 8, 7,  0, 0, 0,  0, 0,  0,  0, 0, 0, 0,  0,
      0,  0, 0,  0,  0,  0,  0, 0,  0, 0,  0, 0, 0,  0, 0,  0,  0, 0, 0,  //
      3,  4, 4,  8,  19, 17, 4, 0,  0, 0,  0, 0, 0,  0, 0,  0,  0, 0, 0, 0,  0,
      0,  0, 0,  0,  0,  0,  0, 0,  0, 0,  0, 0, 0,  0, 0,  0,  0, 0, 0,  //
      7,  7, 4,  3,  8,  15, 8, 15, 7, 7,  7, 6, 7,  4, 0,  0,  0, 0, 0, 0,  0,
      0,  0, 0,  0,  0,  0,  0, 0,  0, 0,  0, 0, 0,  0, 0,  0,  0, 0, 0,  //
      19, 8, 8,  17, 7,  6,  7, 22, 4, 16, 7, 8, 20, 4, 11, 16, 8, 8, 7, 12, 8,
      20, 4, 8,  8,  17, 7,  4, 17, 7, 13, 7, 6, 11, 4, 13, 16, 7, 8, 4  //
  });

  auto* data = tensor->mutable_data<int64_t>();
  for (int i = 0; i < datas.size(); i++) {
    data[i] = datas[i];
  }
}

void PreparePosition(PaddleTensor* tensor) {
  tensor->shape.assign({1, 3, 4, 1});
  std::vector<int> datas({3, 4, 5, 6, 3, 4, 7, 8, 3, 4, 10, 11});

  auto* data = tensor->mutable_data<int64_t>();
  for (int i = 0; i < datas.size(); i++) {
    data[i] = datas[i];
  }
}

void PrepareUnigram(PaddleTensor* tensor) {
  tensor->shape.assign({1, 4, 40, 4, 1});

  std::vector<int> datas{
      {// data1
       1187, 1299, 0, 0, 896, 1388, 0, 0, 2305, 114, 0, 0, 1440, 0, 0, 0, 2626,
       0, 0, 0, 2169, 3646, 5606, 0, 4419, 910, 0, 0, 4609, 0, 0, 0, 3086, 5310,
       0, 0, 5023, 473, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0,
       // data2
       169, 3646, 5606, 0, 46, 0, 0, 0, 2146, 0, 0, 0, 3379, 731, 0, 0, 4020,
       403, 0, 0, 4586, 0, 0, 0, 1270, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       // data3
       4503, 229, 0, 0, 3451, 2488, 0, 0, 3073, 0, 0, 0, 2169, 3646, 5606, 0,
       731, 0, 0, 0, 3672, 3265, 0, 0, 2707, 2292, 0, 0, 2149, 3265, 0, 0, 5355,
       4587, 1827, 0, 3345, 4297, 0, 0, 1769, 2518, 0, 0, 541, 0, 0, 0, 2172,
       3789, 5184, 0, 1270, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       // data4
       1786, 0, 0, 0, 3806, 5262, 0, 0, 5436, 0, 0, 0, 2047, 1305, 0, 0, 1831,
       3517, 0, 0, 541, 0, 0, 0, 1187, 1299, 0, 0, 4272, 0, 0, 0, 2626, 0, 0, 0,
       43, 0, 0, 0, 1187, 1299, 0, 0, 1533, 212, 0, 0, 3094, 2680, 0, 0, 2626,
       0, 0, 0, 2179, 1299, 1881, 0, 4092, 0, 0, 0, 5228, 1181, 0, 0, 2457,
       3104, 0, 0, 1578, 2680, 1730, 0, 4144, 5664, 0, 0, 1533, 212, 0, 0, 5,
       11, 0, 0, 2626, 0, 0, 0, 212, 2468, 0, 0, 768, 2292, 0, 0, 3672, 2774, 0,
       0, 5027, 1346, 0, 0, 3001, 0, 0, 0, 3672, 2774, 0, 0, 6, 1346, 0, 0, 66,
       0, 0, 0, 809, 4067, 0, 0, 541, 0, 0, 0, 5355, 4587, 1827, 3588, 2626, 0,
       0, 0, 4969, 3009, 0, 0, 1658, 0, 0, 0, 3094, 2814, 0, 0, 883, 4576, 0, 0,
       1270, 0, 0, 0}};
  ASSERT_EQ(datas.size(), 640);

  auto* data = tensor->mutable_data<int64_t>();
  for (int i = 0; i < datas.size(); i++) {
    data[i] = datas[i];
  }
}

void PrepareLength(PaddleTensor* tensor) {
  tensor->shape.assign({1, 3, 1});
  std::vector<int> datas({6, 4, 9});

  auto* data = tensor->mutable_data<int64_t>();

  for (int i = 0; i < datas.size(); i++) {
    data[i] = datas[i];
  }
}

void PrepareDense(PaddleTensor* tensor) {
  tensor->shape.assign({1, 3, 5});
  std::vector<float> datas({
      0, 0, 0, 0, 0.999999, 0, 0, 0, 0, 0.999999, 0, 0, 0, 0, 0,
  });

  auto* data = tensor->mutable_data<float>();
  for (int i = 0; i < datas.size(); i++) {
    data[i] = datas[i];
  }
}

void Prepare_lcs_with_title_pr(PaddleTensor* tensor) {
  tensor->shape.assign({1, 3, 2});
  auto* data = tensor->mutable_data<float>();

  std::vector<float> datas(
      {0.272727, 0.166667, 0.111111, 0.166667, 0.0422535, 0.166667});
  for (int i = 0; i < datas.size(); i++) {
    data[i] = datas[i];
  }
}

void Prepare_effective_unigram_num(PaddleTensor* tensor) {
  tensor->shape.assign({1, 4, 40});
  auto* data = tensor->mutable_data<float>();

  std::vector<float> datas(
      {2, 2, 2, 1, 1, 3, 2, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 1, 2, 2, 1,
       1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 3, 1, 2, 2, 2, 3, 2, 2, 1,
       3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 2, 1, 2, 2, 1, 2, 1, 1, 1, 2, 2, 2, 1, 3, 1, 2, 2,
       3, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 1, 2, 1, 4, 1, 2, 1, 2, 2, 1});

  for (int i = 0; i < datas.size(); i++) {
    data[i] = datas[i];
  }
}

void Prepare_sent_num_mask(PaddleTensor* tensor) {
  tensor->shape.assign({1, 3});
  auto* data = tensor->mutable_data<float>();

  std::vector<float> datas({1, 1, 1});

  for (int i = 0; i < datas.size(); i++) {
    data[i] = datas[i];
  }
}

void PrepareInputs(std::vector<PaddleTensor>& inputs) {
  inputs.emplace_back();
  PrepareWordInput(&inputs.back());

  inputs.emplace_back();
  PreparePostag(&inputs.back());

  inputs.emplace_back();
  PrepareUnigram(&inputs.back());

  inputs.emplace_back();

  PreparePosition(&inputs.back());

  inputs.emplace_back();
  PrepareLength(&inputs.back());

  inputs.emplace_back();
  PrepareDense(&inputs.back());

  inputs.emplace_back();
  Prepare_lcs_with_title_pr(&inputs.back());

  inputs.emplace_back();
  Prepare_effective_unigram_num(&inputs.back());

  inputs.emplace_back();
  Prepare_sent_num_mask(&inputs.back());
}

TEST(test, test) {
  LOG(INFO) << "model: " << FLAGS_model;
  AnalysisConfig config(FLAGS_model);
  config.SwitchIrDebug();
  config.pass_builder()->DeletePass("identity_scale_op_clean_pass");

  auto predictor = CreatePaddlePredictor(config);

  std::vector<PaddleTensor> inputs, outputs;
  PrepareInputs(inputs);

  ASSERT_TRUE(predictor->Run(inputs, &outputs));

  for (auto& output : outputs) {
    LOG(INFO) << output.data.length();
  }
}

TEST(test, test_multi_threads) {
  LOG(INFO) << "model: " << FLAGS_model;
  AnalysisConfig config(FLAGS_model);
  config.SwitchIrDebug();
  config.pass_builder()->DeletePass("identity_scale_op_clean_pass");

  auto predictor = CreatePaddlePredictor(config);

  std::vector<PaddleTensor> inputs, outputs;
  PrepareInputs(inputs);

  std::vector<std::thread> threads;
  for (int i = 0; i < 10; i++) {
    threads.emplace_back([&] {
      auto p = predictor->Clone();
      ASSERT_TRUE(p->Run(inputs, &outputs));

      for (auto& output : outputs) {
        LOG(INFO) << output.data.length();
      }
    });
  }

  for (auto& t : threads) t.join();
}
}
