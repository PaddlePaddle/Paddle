// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <random>

#include "paddle/fluid/framework/transfer_scope_cache.h"
#include "test/cpp/inference/api/tester_helper.h"

// Here add missing commands
PD_DEFINE_string(infer_model2, "", "model path");
PD_DEFINE_string(infer_model3, "", "model path");

namespace paddle {
namespace inference {

// Shape of Input to models
const int N = 1, C = 3, H = 224, W = 224;

void SetConfig(AnalysisConfig* config, const std::string& infer_model) {
  config->SetModel(infer_model + "/__model__", infer_model + "/__params__");
  config->DisableFCPadding();
  config->SwitchSpecifyInputNames(true);
}

std::unique_ptr<PaddlePredictor> InitializePredictor(
    const std::string& infer_model,
    const std::vector<float>& data,
    bool use_mkldnn) {
  AnalysisConfig cfg;
  SetConfig(&cfg, infer_model);
  if (use_mkldnn) {
    cfg.EnableMKLDNN();
  }

  auto predictor = ::paddle::CreatePaddlePredictor<AnalysisConfig>(cfg);
  auto input_name = predictor->GetInputNames()[0];
  auto input = predictor->GetInputTensor(input_name);
  std::vector<int> shape{N, C, H, W};
  input->Reshape(std::move(shape));
  input->copy_from_cpu(data.data());

  return predictor;
}

// Compare result of NativeConfig and AnalysisConfig
void compare(bool use_mkldnn = false) {
  // Create Input to models
  std::vector<float> data(N * C * H * W);
  std::default_random_engine re{1234};
  std::uniform_real_distribution<float> sampler{0.0, 1.0};
  for (auto& v : data) {
    v = sampler(re);
  }

  // Initialize Models predictors
  auto predictor_1 = InitializePredictor(FLAGS_infer_model, data, use_mkldnn);
  auto predictor_xx = InitializePredictor(FLAGS_infer_model2, data, use_mkldnn);
  auto predictor_3 = InitializePredictor(FLAGS_infer_model3, data, use_mkldnn);

  // Run single xx model
  predictor_xx->ZeroCopyRun();
  auto output =
      predictor_xx->GetOutputTensor(predictor_xx->GetOutputNames()[0]);
  auto output_shape = output->shape();
  int numel = std::accumulate(
      output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
  std::vector<float> xx_output(numel);
  output->copy_to_cpu(xx_output.data());

  // Initialize xx model's predictor to trigger oneDNN cache clearing
  predictor_xx = InitializePredictor(FLAGS_infer_model2, data, use_mkldnn);

  // Run sequence of models
  predictor_1->ZeroCopyRun();
  predictor_xx->ZeroCopyRun();
  predictor_3->ZeroCopyRun();

  // Get again output of xx model , but when all three models were executed
  std::vector<float> xx2_output(numel);
  output = predictor_xx->GetOutputTensor(predictor_xx->GetOutputNames()[0]);
  output->copy_to_cpu(xx2_output.data());

  // compare results
  auto result = std::equal(
      xx_output.begin(),
      xx_output.end(),
      xx2_output.begin(),
      [](const float& l, const float& r) { return fabs(l - r) < 1e-4; });

  PADDLE_ENFORCE_EQ(
      result,
      true,
      ::common::errors::Fatal("Results of model run independently "
                              "differs from results of the same model "
                              "run as a sequence of models"));
}

TEST(Analyzer_mmp, compare) { compare(); }
#ifdef PADDLE_WITH_DNNL
TEST(Analyzer_mmp, compare_mkldnn) { compare(true /* use_mkldnn */); }
#endif

}  // namespace inference
}  // namespace paddle
