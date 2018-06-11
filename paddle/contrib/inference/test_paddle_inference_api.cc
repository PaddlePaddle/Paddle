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

#include "paddle/contrib/inference/paddle_inference_api.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

namespace paddle {

/*
 * Do not use this, just a demo indicating how to customize a config for a
 * specific predictor.
 */
struct DemoConfig : public PaddlePredictor::Config {
  float other_config;
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
           std::vector<PaddleTensor> *output_data) override {
    LOG(INFO) << "Run";
    return false;
  }

  std::unique_ptr<PaddlePredictor> Clone() override { return nullptr; }

  ~DemoPredictor() override {}
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
}

}  // namespace paddle
