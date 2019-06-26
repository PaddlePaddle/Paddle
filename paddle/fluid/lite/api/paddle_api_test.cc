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

#include "paddle/fluid/lite/api/paddle_api.h"
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "paddle/fluid/lite/api/paddle_use_kernels.h"
#include "paddle/fluid/lite/api/paddle_use_ops.h"
#include "paddle/fluid/lite/api/paddle_use_passes.h"

DEFINE_string(model_dir, "", "");

namespace paddle {
namespace lite_api {

TEST(CxxApi, run) {
  lite_api::CxxConfig config;
  config.set_model_dir(FLAGS_model_dir);
  config.set_preferred_place(Place{TARGET(kX86), PRECISION(kFloat)});
  config.set_valid_places({
      Place{TARGET(kX86), PRECISION(kFloat)},
      Place{TARGET(kARM), PRECISION(kFloat)},
  });

  auto predictor = lite_api::CreatePaddlePredictor(config);

  auto input_tensor = predictor->GetInput(0);
  input_tensor->Resize(std::vector<int64_t>({100, 100}));
  auto* data = input_tensor->mutable_data<float>();
  for (int i = 0; i < 100 * 100; i++) {
    data[i] = i;
  }

  predictor->Run();

  auto output = predictor->GetOutput(0);
  auto* out = output->data<float>();
  LOG(INFO) << out[0];
  LOG(INFO) << out[1];

  EXPECT_NEAR(out[0], 50.2132, 1e-3);
  EXPECT_NEAR(out[1], -28.8729, 1e-3);

  predictor->SaveOptimizedModel(FLAGS_model_dir + ".opt2");
}

#ifdef LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
TEST(LightApi, run) {
  lite_api::MobileConfig config;
  config.set_model_dir(FLAGS_model_dir + ".opt2");

  auto predictor = lite_api::CreatePaddlePredictor(config);

  auto input_tensor = predictor->GetInput(0);
  input_tensor->Resize(std::vector<int64_t>({100, 100}));
  auto* data = input_tensor->mutable_data<float>();
  for (int i = 0; i < 100 * 100; i++) {
    data[i] = i;
  }

  predictor->Run();

  auto output = predictor->GetOutput(0);
  auto* out = output->data<float>();
  LOG(INFO) << out[0];
  LOG(INFO) << out[1];

  EXPECT_NEAR(out[0], 50.2132, 1e-3);
  EXPECT_NEAR(out[1], -28.8729, 1e-3);
}
#endif

}  // namespace lite_api
}  // namespace paddle
