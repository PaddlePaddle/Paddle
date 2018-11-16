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

#include "paddle/fluid/inference/api/analysis_predictor.h"
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <thread>  // NOLINT
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"

DEFINE_string(dirname, "", "dirname to tests.");

namespace paddle {
using contrib::AnalysisConfig;

TEST(AnalysisPredictor, analysis_off) {
  AnalysisConfig config(false);
  config.model_dir = FLAGS_dirname;
  config.enable_ir_optim = false;

  auto _predictor = CreatePaddlePredictor<AnalysisConfig>(config);
  auto* predictor = static_cast<AnalysisPredictor*>(_predictor.get());

  // Without analysis, the scope_ and sub_scope_ are created by predictor
  // itself.
  ASSERT_TRUE(predictor->scope_);
  ASSERT_TRUE(predictor->sub_scope_);
  ASSERT_EQ(predictor->scope_->parent(), nullptr);
  ASSERT_EQ(predictor->sub_scope_->parent(), predictor->scope_.get());
  // ir is turned off, so program shouldn't be optimized.
  ASSERT_FALSE(predictor->status_program_optimized_);
  LOG(INFO) << "scope parameters " << predictor->scope_->LocalVarNames().size();

  // 2. Dummy Input Data
  int64_t data[4] = {1, 2, 3, 4};
  PaddleTensor tensor;
  tensor.shape = std::vector<int>({4, 1});
  tensor.data.Reset(data, sizeof(data));
  tensor.dtype = PaddleDType::INT64;

  std::vector<PaddleTensor> inputs(4, tensor);
  std::vector<PaddleTensor> outputs;
  ASSERT_TRUE(predictor->Run(inputs, &outputs));
}

TEST(AnalysisPredictor, analysis_on) {
  AnalysisConfig config(false);
  config.model_dir = FLAGS_dirname;
  config.enable_ir_optim = true;

  auto _predictor = CreatePaddlePredictor<AnalysisConfig>(config);
  auto* predictor = static_cast<AnalysisPredictor*>(_predictor.get());

  ASSERT_TRUE(predictor->scope_);
  ASSERT_TRUE(predictor->sub_scope_);
  ASSERT_EQ(predictor->scope_->parent(), nullptr);
  ASSERT_EQ(predictor->sub_scope_->parent(), predictor->scope_.get());
  // ir is turned on, so program should be optimized.
  ASSERT_TRUE(predictor->status_program_optimized_);
  // 2. Dummy Input Data
  int64_t data[4] = {1, 2, 3, 4};
  PaddleTensor tensor;
  tensor.shape = std::vector<int>({4, 1});
  tensor.data.Reset(data, sizeof(data));
  tensor.dtype = PaddleDType::INT64;

  std::vector<PaddleTensor> inputs(4, tensor);
  std::vector<PaddleTensor> outputs;
  ASSERT_TRUE(predictor->Run(inputs, &outputs));

  for (auto& output : outputs) {
    LOG(INFO) << inference::DescribeTensor(output);
  }

  // compare with NativePredictor
  auto naive_predictor = CreatePaddlePredictor<NativeConfig>(config);
  std::vector<PaddleTensor> naive_outputs;
  ASSERT_TRUE(naive_predictor->Run(inputs, &naive_outputs));
  ASSERT_EQ(naive_outputs.size(), 1UL);
  inference::CompareTensor(outputs.front(), naive_outputs.front());
}

TEST(AnalysisPredictor, ZeroCopy) {
  AnalysisConfig config;
  config.model_dir = FLAGS_dirname;
  config.use_feed_fetch_ops = false;

  auto predictor = CreatePaddlePredictor<AnalysisConfig>(config);

  auto w0 = predictor->GetInputTensor("firstw");
  auto w1 = predictor->GetInputTensor("secondw");
  auto w2 = predictor->GetInputTensor("thirdw");
  auto w3 = predictor->GetInputTensor("forthw");

  w0->Reshape({4, 1});
  w1->Reshape({4, 1});
  w2->Reshape({4, 1});
  w3->Reshape({4, 1});

  auto* w0_data = w0->mutable_data<int64_t>(PaddlePlace::kCPU);
  auto* w1_data = w1->mutable_data<int64_t>(PaddlePlace::kCPU);
  auto* w2_data = w2->mutable_data<int64_t>(PaddlePlace::kCPU);
  auto* w3_data = w3->mutable_data<int64_t>(PaddlePlace::kCPU);

  for (int i = 0; i < 4; i++) {
    w0_data[i] = i;
    w1_data[i] = i;
    w2_data[i] = i;
    w3_data[i] = i;
  }

  predictor->ZeroCopyRun();

  auto out = predictor->GetOutputTensor("fc_1.tmp_2");
  PaddlePlace place;
  int size = 0;
  auto* out_data = out->data<float>(&place, &size);
  LOG(INFO) << "output size: " << size / sizeof(float);
  LOG(INFO) << "output_data: " << out_data;
}

TEST(AnalysisPredictor, Clone) {
  AnalysisConfig config;
  config.model_dir = FLAGS_dirname;
  config.use_feed_fetch_ops = true;
  config.enable_ir_optim = true;

  std::vector<std::unique_ptr<PaddlePredictor>> predictors;
  predictors.emplace_back(CreatePaddlePredictor(config));

  LOG(INFO) << "************** to clone ************************";
  const int num_threads = 3;
  for (int i = 1; i < num_threads; i++) {
    predictors.emplace_back(predictors.front()->Clone());
  }

  auto* root_scope =
      static_cast<AnalysisPredictor*>(predictors[0].get())->scope();
  ASSERT_FALSE(root_scope->kids().empty());
  LOG(INFO) << "***** scope ******\n"
            << framework::GenScopeTreeDebugInfo(root_scope);

  // 2. Dummy Input Data
  int64_t data[4] = {1, 2, 3, 4};
  PaddleTensor tensor;
  tensor.shape = std::vector<int>({4, 1});
  tensor.data.Reset(data, sizeof(data));
  tensor.dtype = PaddleDType::INT64;

  std::vector<PaddleTensor> inputs(4, tensor);
  std::vector<PaddleTensor> outputs;
  predictors[0]->Run(inputs, &outputs);

  LOG(INFO) << "Run with single thread";
  for (int i = 0; i < num_threads; i++) {
    LOG(INFO) << "run predictor " << i;
    ASSERT_TRUE(predictors[i]->Run(inputs, &outputs));
  }

  LOG(INFO) << "Run with multiple threads";
  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; i++) {
    threads.emplace_back([&predictors, &inputs, i] {
      LOG(INFO) << "thread #" << i << " running";
      std::vector<PaddleTensor> outputs;
      for (int j = 0; j < 10; j++) {
        ASSERT_TRUE(predictors[i]->Run(inputs, &outputs));
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }
}

}  // namespace paddle
