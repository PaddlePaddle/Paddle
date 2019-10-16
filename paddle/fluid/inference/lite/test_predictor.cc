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
// See the License for the specific language governing permissions and
// limitations under the License.

#include <ios>
#include <fstream>
#include <gtest/gtest.h>

#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"

#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"

namespace paddle {
namespace lite {

using paddle::AnalysisConfig;

TEST(AnalysisPredictor, Lite) {

  AnalysisConfig config;
  config.SetModel("/shixiaowei02/Paddle_lite/xingzhaolong/leaky_relu_model");
  config.SwitchUseFeedFetchOps(false);
  config.EnableUseGpu(10, 1);
  config.EnableLiteEngine(paddle::AnalysisConfig::Precision::kFloat32);
  config.pass_builder()->TurnOnDebug();

  auto predictor = CreatePaddlePredictor(config);
  PADDLE_ENFORCE_NOT_NULL(predictor.get());
}


}  // namespace lite
}  // namespace paddle 
