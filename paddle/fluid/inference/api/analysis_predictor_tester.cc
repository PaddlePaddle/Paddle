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
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/tests/api/tester_helper.h"
#include "paddle/fluid/platform/cpu_info.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/inference/api/mkldnn_quantizer.h"
#endif

DEFINE_string(dirname, "", "dirname to tests.");

namespace paddle {

TEST(AnalysisPredictor, analysis_off) {
  AnalysisConfig config;
  config.SetModel(FLAGS_dirname);
  config.SwitchIrOptim(false);
  LOG(INFO) << config.Summary();

  auto _predictor = CreatePaddlePredictor<AnalysisConfig>(config);
  auto* predictor = static_cast<AnalysisPredictor*>(_predictor.get());

  // Without analysis, the scope_ and sub_scope_ are created by predictor
  // itself.
  ASSERT_TRUE(predictor->scope_);
  ASSERT_TRUE(predictor->sub_scope_);
  ASSERT_EQ(predictor->scope_->parent(), nullptr);
  ASSERT_EQ(predictor->sub_scope_->parent(), predictor->scope_.get());
  // ir is turned off, so program shouldn't be optimized.
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
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  config.EnableUseGpu(100, 0);
#else
  config.DisableGpu();
#endif
  LOG(INFO) << config.Summary();

  auto _predictor = CreatePaddlePredictor<AnalysisConfig>(config);
  auto* predictor = static_cast<AnalysisPredictor*>(_predictor.get());

  ASSERT_TRUE(predictor->scope_);
  ASSERT_TRUE(predictor->sub_scope_);
  ASSERT_EQ(predictor->scope_->parent(), nullptr);
  ASSERT_EQ(predictor->sub_scope_->parent(), predictor->scope_.get());
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
  LOG(INFO) << config.Summary();
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
  predictor->TryShrinkMemory();
}

TEST(AnalysisPredictor, Clone) {
  AnalysisConfig config;
  config.SetModel(FLAGS_dirname);
  config.SwitchUseFeedFetchOps(true);
  config.SwitchIrOptim(true);
  LOG(INFO) << config.Summary();

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

// This function is not released yet, will fail on some machine.
// TODO(Superjomn) Turn on it latter.
/*
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
*/

#ifdef PADDLE_WITH_MKLDNN
class MkldnnQuantizerTest : public testing::Test {
 public:
  MkldnnQuantizerTest() {
    AnalysisConfig config(FLAGS_dirname);
    predictor = std::move(CreatePaddlePredictor(config));
    auto* predictor_p = static_cast<AnalysisPredictor*>(predictor.get());

    auto qconfig = new MkldnnQuantizerConfig();

    mkldnn_quantizer.reset(
        new AnalysisPredictor::MkldnnQuantizer(*predictor_p, qconfig));
  }

  std::pair<std::vector<int>, float> Histogram(
      const framework::LoDTensor& var_tensor, float min_val, float max_val,
      int num_bins) const {
    return mkldnn_quantizer->Histogram(var_tensor, min_val, max_val, num_bins);
  }

  std::pair<bool, framework::LoDTensor> GetMaxScalingFactor(
      const framework::LoDTensor& var_tensor, bool is_unsigned) const {
    return mkldnn_quantizer->GetMaxScalingFactor(var_tensor, is_unsigned);
  }

  std::pair<bool, framework::LoDTensor> GetMaxChScalingFactor(
      const framework::LoDTensor& var_tensor, bool is_unsigned) const {
    return mkldnn_quantizer->GetMaxChScalingFactor(var_tensor, is_unsigned, 0);
  }

  std::pair<bool, framework::LoDTensor> GetKLScalingFactor(
      const framework::LoDTensor& var_tensor, bool is_unsigned) const {
    return mkldnn_quantizer->GetKLScalingFactor(var_tensor, is_unsigned);
  }

 protected:
  std::unique_ptr<PaddlePredictor> predictor;
  std::unique_ptr<AnalysisPredictor::MkldnnQuantizer> mkldnn_quantizer;
  float abs_error = 1e-6;
  static const std::array<float, 10> non_negative_values;
  static const std::array<float, 10> positive_and_negative_values;
};

const std::array<float, 10> MkldnnQuantizerTest::non_negative_values = {
    0.0158671, 0.026459,   0.0280772,  0.00962479, 0.0131628,
    0.016704,  0.00118407, 0.00765726, 0.0123213,  0.00944741};
const std::array<float, 10> MkldnnQuantizerTest::positive_and_negative_values =
    {-0.0482659, -0.0102493, -0.00794221, -0.00387115, -0.00674586,
     -0.0495346, 0.0629528,  -0.00531285, -0.0230353,  0.0269089};

TEST_F(MkldnnQuantizerTest, histogram_inverted_min_max) {
  const auto& values = non_negative_values;
  auto min_val = *std::min_element(values.begin(), values.end());
  auto max_val = *std::max_element(values.begin(), values.end());

  framework::LoDTensor var_tensor;
  var_tensor.Resize(framework::make_dim(values.size()));
  std::copy(begin(values), end(values),
            var_tensor.mutable_data<float>(platform::CPUPlace()));

  ASSERT_THROW(Histogram(var_tensor, max_val, min_val, 3),
               platform::EnforceNotMet);
}

TEST_F(MkldnnQuantizerTest, histogram_non_negative_to_3) {
  // all non-negative values
  const auto& values = non_negative_values;
  auto min_val = *std::min_element(values.begin(), values.end());
  auto max_val = *std::max_element(values.begin(), values.end());

  framework::LoDTensor var_tensor;
  var_tensor.Resize(framework::make_dim(values.size()));
  std::copy(begin(values), end(values),
            var_tensor.mutable_data<float>(platform::CPUPlace()));

  std::vector<int> histogram;
  float bin_width;

  std::tie(histogram, bin_width) = Histogram(var_tensor, min_val, max_val, 3);

  ASSERT_NEAR(bin_width, std::abs(max_val - min_val) / 3.f, abs_error)
      << "Improperly calculated bin_width.";

  ASSERT_EQ(histogram[0], 4);
  ASSERT_EQ(histogram[1], 4);
  ASSERT_EQ(histogram[2], 2);
}

TEST_F(MkldnnQuantizerTest, histogram_positive_and_negative_to_3) {
  const auto& values = positive_and_negative_values;
  auto min_val = *std::min_element(values.begin(), values.end());
  auto max_val = *std::max_element(values.begin(), values.end());

  framework::LoDTensor var_tensor;
  var_tensor.Resize(framework::make_dim(values.size()));
  std::copy(begin(values), end(values),
            var_tensor.mutable_data<float>(platform::CPUPlace()));

  std::vector<int> histogram;
  float bin_width;

  std::tie(histogram, bin_width) = Histogram(var_tensor, min_val, max_val, 3);

  ASSERT_NEAR(bin_width, std::abs(max_val - min_val) / 3.0f, abs_error)
      << "Improperly calculated bin_width.";

  ASSERT_EQ(histogram[0], 3);
  ASSERT_EQ(histogram[1], 5);
  ASSERT_EQ(histogram[2], 2);
}

TEST_F(MkldnnQuantizerTest, histogram_zero_bins) {
  const auto& values = non_negative_values;
  auto min_val = *std::min_element(values.begin(), values.end());
  auto max_val = *std::max_element(values.begin(), values.end());

  framework::LoDTensor var_tensor;
  var_tensor.Resize(framework::make_dim(values.size()));
  std::copy(begin(values), end(values),
            var_tensor.mutable_data<float>(platform::CPUPlace()));

  ASSERT_THROW(Histogram(var_tensor, min_val, max_val, 0),
               platform::EnforceNotMet);
}

TEST_F(MkldnnQuantizerTest, histogram_empty) {
  // empty tensor
  ASSERT_THROW(Histogram({}, -1, 1, 1), platform::EnforceNotMet);

  // zero tensor
  framework::LoDTensor var_tensor;
  var_tensor.Resize({0});
  var_tensor.mutable_data<double>(platform::CPUPlace());

  ASSERT_THROW(Histogram(var_tensor, -1, 1, 1), platform::EnforceNotMet);
}

TEST_F(MkldnnQuantizerTest, kl_scaling_factor_signed) {
  const auto& values = positive_and_negative_values;

  framework::LoDTensor var_tensor;
  var_tensor.Resize(framework::make_dim(values.size()));
  std::copy(begin(values), end(values),
            var_tensor.mutable_data<float>(platform::CPUPlace()));

  bool is_unsigned;
  framework::LoDTensor lod_tensor;

  std::tie(is_unsigned, lod_tensor) = GetKLScalingFactor(var_tensor, false);

  ASSERT_EQ(is_unsigned, false);
  ASSERT_EQ(lod_tensor.numel(), 1);
  ASSERT_NEAR(lod_tensor.data<double>()[0], 1.0 / 0.0899106152344, abs_error);
}

TEST_F(MkldnnQuantizerTest, max_scaling_factor_signed) {
  const auto& values = positive_and_negative_values;
  auto max_val = *std::max_element(values.begin(), values.end());

  framework::LoDTensor var_tensor;
  var_tensor.Resize(framework::make_dim(values.size()));
  std::copy(begin(values), end(values),
            var_tensor.mutable_data<float>(platform::CPUPlace()));

  bool is_unsigned;
  framework::LoDTensor lod_tensor;

  std::tie(is_unsigned, lod_tensor) = GetMaxScalingFactor(var_tensor, false);

  ASSERT_EQ(is_unsigned, false);
  ASSERT_EQ(lod_tensor.numel(), 1);
  ASSERT_NEAR(lod_tensor.data<double>()[0], 1.0 / max_val, abs_error);
}

TEST_F(MkldnnQuantizerTest, max_scaling_factor_unsigned) {
  const auto& values = non_negative_values;
  auto max_val = *std::max_element(values.begin(), values.end());

  framework::LoDTensor var_tensor;
  var_tensor.Resize(framework::make_dim(values.size()));
  std::copy(begin(values), end(values),
            var_tensor.mutable_data<float>(platform::CPUPlace()));

  bool is_unsigned;
  framework::LoDTensor lod_tensor;

  std::tie(is_unsigned, lod_tensor) = GetMaxScalingFactor(var_tensor, true);

  ASSERT_EQ(is_unsigned, true);
  ASSERT_EQ(lod_tensor.numel(), 1);
  ASSERT_NEAR(lod_tensor.data<double>()[0], 1.0 / max_val, abs_error);
}

TEST_F(MkldnnQuantizerTest, max_scaling_factor_chwise_unsigned) {
  const auto& values = non_negative_values;
  auto max_val = *std::max_element(values.begin(), values.end());
  int channels = 3;

  framework::LoDTensor var_tensor;
  var_tensor.Resize(framework::make_dim(channels, 1, 1, values.size()));
  for (int i = 0; i < channels; i++)
    std::copy(begin(values), end(values),
              var_tensor.mutable_data<float>(platform::CPUPlace()) +
                  i * values.size());

  bool is_unsigned;
  framework::LoDTensor lod_tensor;

  std::tie(is_unsigned, lod_tensor) = GetMaxChScalingFactor(var_tensor, true);

  ASSERT_EQ(is_unsigned, true);
  ASSERT_EQ(lod_tensor.numel(), channels);
  for (int i = 0; i < channels; i++) {
    ASSERT_NEAR(lod_tensor.data<double>()[i], 1.0 / max_val, abs_error);
  }
}

TEST_F(MkldnnQuantizerTest, kl_scaling_factor_unsigned) {
  const auto& values = non_negative_values;

  framework::LoDTensor var_tensor;
  var_tensor.Resize(framework::make_dim(values.size()));
  std::copy(begin(values), end(values),
            var_tensor.mutable_data<float>(platform::CPUPlace()));

  bool is_unsigned;
  framework::LoDTensor lod_tensor;

  std::tie(is_unsigned, lod_tensor) = GetKLScalingFactor(var_tensor, true);

  ASSERT_EQ(is_unsigned, true);
  ASSERT_EQ(lod_tensor.numel(), 1);
  ASSERT_NEAR(lod_tensor.data<double>()[0], 1.0 / 0.0252845321362, abs_error);
}
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(AnalysisPredictor, bf16_gpu_pass_strategy) {
  AnalysisConfig config;
  config.SetModel(FLAGS_dirname);
  config.SwitchIrOptim(true);
  config.EnableUseGpu(100, 0);
  config.EnableMkldnnBfloat16();
#ifdef PADDLE_WITH_MKLDNN
  if (platform::MayIUse(platform::cpu_isa_t::avx512_core))
    ASSERT_EQ(config.mkldnn_bfloat16_enabled(), true);
  else
    ASSERT_EQ(config.mkldnn_bfloat16_enabled(), false);
#else
  ASSERT_EQ(config.mkldnn_bfloat16_enabled(), false);
#endif
}
#endif

TEST(AnalysisPredictor, bf16_pass_strategy) {
  std::vector<std::string> passes;
  PassStrategy passStrategy(passes);
  passStrategy.EnableMkldnnBfloat16();
}

}  // namespace paddle

namespace paddle_infer {

TEST(Predictor, Run) {
  Config config;
  config.SetModel(FLAGS_dirname);

  auto predictor = CreatePredictor(config);

  auto w0 = predictor->GetInputHandle("firstw");
  auto w1 = predictor->GetInputHandle("secondw");
  auto w2 = predictor->GetInputHandle("thirdw");
  auto w3 = predictor->GetInputHandle("forthw");

  w0->Reshape({4, 1});
  w1->Reshape({4, 1});
  w2->Reshape({4, 1});
  w3->Reshape({4, 1});

  auto* w0_data = w0->mutable_data<int64_t>(PlaceType::kCPU);
  auto* w1_data = w1->mutable_data<int64_t>(PlaceType::kCPU);
  auto* w2_data = w2->mutable_data<int64_t>(PlaceType::kCPU);
  auto* w3_data = w3->mutable_data<int64_t>(PlaceType::kCPU);

  for (int i = 0; i < 4; i++) {
    w0_data[i] = i;
    w1_data[i] = i;
    w2_data[i] = i;
    w3_data[i] = i;
  }

  predictor->Run();

  auto out = predictor->GetOutputHandle("fc_1.tmp_2");
  PlaceType place;
  int size = 0;
  out->data<float>(&place, &size);
  LOG(INFO) << "output size: " << size / sizeof(float);
  predictor->TryShrinkMemory();
}

}  // namespace paddle_infer
