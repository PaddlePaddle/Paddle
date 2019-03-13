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
#include "gmock/gmock.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/tests/api/tester_helper.h"

DEFINE_string(dirname, "", "dirname to tests.");

namespace paddle {

TEST(AnalysisPredictor, analysis_off) {
  AnalysisConfig config;
  config.SetModel(FLAGS_dirname);
  config.SwitchIrOptim(false);

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
  AnalysisConfig config;
  config.SetModel(FLAGS_dirname);
  config.SwitchIrOptim(true);
#ifdef PADDLE_WITH_CUDA
  config.EnableUseGpu(100, 0);
#else
  config.DisableGpu();
#endif

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
  auto naive_predictor =
      CreatePaddlePredictor<NativeConfig>(config.ToNativeConfig());
  std::vector<PaddleTensor> naive_outputs;
  ASSERT_TRUE(naive_predictor->Run(inputs, &naive_outputs));
  ASSERT_EQ(naive_outputs.size(), 1UL);
  inference::CompareTensor(outputs.front(), naive_outputs.front());
}

TEST(AnalysisPredictor, ZeroCopy) {
  AnalysisConfig config;
  config.SetModel(FLAGS_dirname);
  config.SwitchUseFeedFetchOps(false);
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
  config.SetModel(FLAGS_dirname);
  config.SwitchUseFeedFetchOps(true);
  config.SwitchIrOptim(true);

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
      auto predictor = predictors.front()->Clone();
      for (int j = 0; j < 10; j++) {
        ASSERT_TRUE(predictor->Run(inputs, &outputs));
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }
}

TEST(AnalysisPredictor, memory_optim) {
  AnalysisConfig config(FLAGS_dirname);
  config.DisableGpu();
  config.EnableMemoryOptim(true);
  config.SwitchIrDebug();

  auto native_predictor =
      CreatePaddlePredictor<NativeConfig>(config.ToNativeConfig());

  // 2. Dummy Input Data
  int64_t data[4] = {1, 2, 3, 4};
  PaddleTensor tensor;
  tensor.shape = std::vector<int>({4, 1});
  tensor.data.Reset(data, sizeof(data));
  tensor.dtype = PaddleDType::INT64;

  std::vector<PaddleTensor> inputs(4, tensor);
  std::vector<PaddleTensor> output, output1;

  {
    // The first predictor help to cache the memory optimize strategy.
    auto predictor = CreatePaddlePredictor<AnalysisConfig>(config);
    LOG(INFO) << "serialized program: " << predictor->GetSerializedProgram();
    ASSERT_FALSE(predictor->GetSerializedProgram().empty());

    // Run several times to check the parameters are not reused by mistake.
    for (int i = 0; i < 5; i++) {
      ASSERT_TRUE(predictor->Run(inputs, &output));
    }
  }

  {
    output.clear();
    // The second predictor to perform memory optimization.
    config.EnableMemoryOptim(false);
    auto predictor = CreatePaddlePredictor<AnalysisConfig>(config);

    // Run with memory optimization
    ASSERT_TRUE(predictor->Run(inputs, &output));
  }

  // Run native
  ASSERT_TRUE(native_predictor->Run(inputs, &output1));

  LOG(INFO) << "the output " << inference::DescribeTensor(output.front());
  LOG(INFO) << "the native output "
            << inference::DescribeTensor(output1.front());

  inference::CompareResult(output, output1);
}

TEST(Quantizer, expand_quantized_bins) {
  FAIL() << "Test not implemented yet.";
}

class QuantizerTest : public testing::Test {
 public:
  QuantizerTest() {
    AnalysisConfig config(FLAGS_dirname);

    auto _predictor = CreatePaddlePredictor<AnalysisConfig>(config);
    auto* predictor = static_cast<AnalysisPredictor*>(_predictor.get());

    auto qconfig = std::make_shared<QuantizerConfig>();

    quantizer.reset(new AnalysisPredictor::Quantizer(*predictor, qconfig));
  }

  std::pair<std::vector<int>, float> Histogram(
      const framework::LoDTensor& var_tensor, float min_val, float max_val,
      int num_bins) const {
    return quantizer->Histogram(var_tensor, min_val, max_val, num_bins);
  }

  std::pair<QuantMax, framework::LoDTensor> GetMaxScalingFactor(
      const framework::LoDTensor& var_tensor) const {
    return quantizer->GetMaxScalingFactor(var_tensor);
  }

 protected:
  std::unique_ptr<AnalysisPredictor::Quantizer> quantizer;
  float abs_error = 1e-6;
  static const std::array<float, 5> non_negative_values;
  static const std::array<float, 5> positive_and_negative_values;
};

const std::array<float, 5> QuantizerTest::non_negative_values = {
    0.5e6f, 1e3f, 0.f, 0.5e-3f, 1e-4f};
const std::array<float, 5> QuantizerTest::positive_and_negative_values = {
    -1e2f, 1e2f, -1e1f, 1e1f, 1e-4f};

TEST_F(QuantizerTest, histogram_inverted_min_max) {
  const std::array<float, 5>& values = non_negative_values;
  float min_val = *std::min_element(values.begin(), values.end());
  float max_val = *std::max_element(values.begin(), values.end());

  framework::LoDTensor var_tensor;
  var_tensor.Resize(framework::make_dim(values.size()));
  std::copy(begin(values), end(values),
            var_tensor.mutable_data<float>(platform::CPUPlace()));

  ASSERT_THROW(Histogram(var_tensor, max_val, min_val, 3),
               platform::EnforceNotMet);
}

TEST_F(QuantizerTest, histogram_non_negative_5_to_3) {
  // all non-negative values
  const std::array<float, 5>& values = non_negative_values;
  float min_val = *std::min_element(values.begin(), values.end());
  float max_val = *std::max_element(values.begin(), values.end());

  framework::LoDTensor var_tensor;
  var_tensor.Resize(framework::make_dim(values.size()));
  std::copy(begin(values), end(values),
            var_tensor.mutable_data<float>(platform::CPUPlace()));

  std::vector<int> histogram;
  float bin_width;

  std::tie(histogram, bin_width) = Histogram(var_tensor, min_val, max_val, 3);

  ASSERT_NEAR(bin_width, std::abs(max_val - min_val) / 3.f, abs_error)
      << "Improperly calculated bin_width.";

  ASSERT_THAT(histogram, testing::ElementsAre(4, 0, 1))
      << "Improperly calculated histogram.";
}

TEST_F(QuantizerTest, histogram_positive_and_negative_5_to_3) {
  const std::array<float, 5>& values = positive_and_negative_values;
  float min_val = *std::min_element(values.begin(), values.end());
  float max_val = *std::max_element(values.begin(), values.end());

  framework::LoDTensor var_tensor;
  var_tensor.Resize(framework::make_dim(values.size()));
  std::copy(begin(values), end(values),
            var_tensor.mutable_data<float>(platform::CPUPlace()));

  std::vector<int> histogram;
  float bin_width;

  std::tie(histogram, bin_width) = Histogram(var_tensor, min_val, max_val, 3);

  ASSERT_NEAR(bin_width, std::abs(max_val - min_val) / 3.0f, abs_error)
      << "Improperly calculated bin_width.";

  ASSERT_THAT(histogram, testing::ElementsAre(1, 3, 1))
      << "Improperly calculated histogram.";
}

TEST_F(QuantizerTest, histogram_zero_bins) {
  const std::array<float, 5>& values = non_negative_values;
  float min_val = *std::min_element(values.begin(), values.end());
  float max_val = *std::max_element(values.begin(), values.end());

  framework::LoDTensor var_tensor;
  var_tensor.Resize(framework::make_dim(values.size()));
  std::copy(begin(values), end(values),
            var_tensor.mutable_data<float>(platform::CPUPlace()));

  ASSERT_THROW(Histogram(var_tensor, min_val, max_val, 0),
               platform::EnforceNotMet);
}

TEST_F(QuantizerTest, histogram_empty) {
  // empty tensor
  ASSERT_THROW(Histogram({}, -1, 1, 1), platform::EnforceNotMet);

  // zero tensor
  framework::LoDTensor var_tensor;
  var_tensor.Resize({0});
  ASSERT_TRUE(var_tensor.mutable_data<float>(platform::CPUPlace()));

  ASSERT_THROW(Histogram(var_tensor, -1, 1, 1), platform::EnforceNotMet);
}

TEST_F(QuantizerTest, kl_scaling_factor) {
  FAIL() << "Test not implemented yet.";
}

TEST_F(QuantizerTest, max_scaling_factor_signed) {
  const std::array<float, 5>& values = positive_and_negative_values;
  float max_val = *std::max_element(values.begin(), values.end());

  framework::LoDTensor var_tensor;
  var_tensor.Resize(framework::make_dim(values.size()));
  std::copy(begin(values), end(values),
            var_tensor.mutable_data<float>(platform::CPUPlace()));

  QuantMax quant_max;
  framework::LoDTensor lod_tensor;

  std::tie(quant_max, lod_tensor) = GetMaxScalingFactor(var_tensor);

  ASSERT_EQ(quant_max, QuantMax::S8_MAX);
  ASSERT_EQ(lod_tensor.numel(), 1);
  ASSERT_NEAR(lod_tensor.data<float>()[0],
              static_cast<float>(QuantMax::S8_MAX) / max_val, abs_error);
}

TEST_F(QuantizerTest, max_scaling_factor_unsigned) {
  const std::array<float, 5>& values = non_negative_values;
  float max_val = *std::max_element(values.begin(), values.end());

  framework::LoDTensor var_tensor;
  var_tensor.Resize(framework::make_dim(values.size()));
  std::copy(begin(values), end(values),
            var_tensor.mutable_data<float>(platform::CPUPlace()));

  QuantMax quant_max;
  framework::LoDTensor lod_tensor;

  std::tie(quant_max, lod_tensor) = GetMaxScalingFactor(var_tensor);

  ASSERT_EQ(quant_max, QuantMax::U8_MAX);
  ASSERT_EQ(lod_tensor.numel(), 1);
  ASSERT_NEAR(lod_tensor.data<float>()[0],
              static_cast<float>(QuantMax::U8_MAX) / max_val, abs_error);
}

TEST_F(QuantizerTest, safe_entropy) { FAIL() << "test not implemented yet."; }

}  // namespace paddle
