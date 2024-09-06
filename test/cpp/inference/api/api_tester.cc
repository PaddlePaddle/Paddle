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

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <exception>
#include <string>

#include "paddle/fluid/inference/api/paddle_analysis_config.h"
#include "paddle/fluid/inference/api/paddle_api.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {

/*
 * Do not use this, just a demo indicating how to customize a config for a
 * specific predictor.
 */
struct DemoConfig : public PaddlePredictor::Config {
  float other_config;
  DemoConfig() : other_config(0) {}
};

/*
 * Do not use this, just a demo indicating how to customize a Predictor.
 */
class DemoPredictor : public PaddlePredictor {
 public:
  explicit DemoPredictor(const DemoConfig &config) {
    LOG(INFO) << "I get other_config " << config.other_config;
  }
  bool Run(const std::vector<PaddleTensor> &inputs,
           std::vector<PaddleTensor> *output_data,
           int batch_size = 0) override {
    LOG(INFO) << "Run";
    return false;
  }

  std::unique_ptr<PaddlePredictor> Clone(void *stream = nullptr) override {
    return nullptr;
  }

  ~DemoPredictor() override = default;
};

template <>
std::unique_ptr<PaddlePredictor> CreatePaddlePredictor<DemoConfig>(
    const DemoConfig &config) {
  std::unique_ptr<PaddlePredictor> x(new DemoPredictor(config));
  return x;
}

TEST(paddle_inference_api, demo) {
  DemoConfig config;
  config.other_config = 1.7;
  auto predictor = CreatePaddlePredictor(config);
  std::vector<PaddleTensor> outputs;
  predictor->Run({}, &outputs);
  predictor->TryShrinkMemory();
}

TEST(paddle_inference_api, get_version) {
  LOG(INFO) << "paddle version:\n" << get_version();
  auto version = get_version();
  ASSERT_FALSE(version.empty());
}

TEST(paddle_inference_api, UpdateDllFlag) {
  UpdateDllFlag("paddle_num_threads", "10");
  try {
    UpdateDllFlag("paddle_num_threads2", "10");
  } catch (std::exception &e) {
    LOG(INFO) << e.what();
  }
}

TEST(paddle_inference_api, AnalysisConfigCopyCtor) {
  AnalysisConfig cfg1;
  cfg1.EnableUseGpu(10);
#ifdef PADDLE_WITH_TENSORRT
  cfg1.EnableTensorRtEngine();
#endif
  std::string delete_pass("skip_layernorm_fuse_pass");
  cfg1.pass_builder()->DeletePass(delete_pass);
  AnalysisConfig cfg2(cfg1);

  auto passes = cfg2.pass_builder()->AllPasses();
  for (auto const &ps : passes) {
    PADDLE_ENFORCE_NE(ps,
                      delete_pass,
                      common::errors::InvalidArgument(
                          "Required ps shouldn't be euqal to delete_pass. "));
  }
}

#ifdef PADDLE_WITH_CRYPTO
TEST(paddle_inference_api, crypto) { paddle::MakeCipher(""); }
#endif

}  // namespace paddle
