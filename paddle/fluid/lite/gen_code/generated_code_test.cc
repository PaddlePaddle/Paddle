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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include "paddle/fluid/lite/gen_code/paddle_infer.h"

namespace paddle {
namespace lite {

TEST(PaddlePredictor, Init) {
  gencode::PaddlePredictor predictor;
  predictor.Init();
}

#ifdef LITE_WITH_X86
TEST(PaddlePredictor, RunX86) {
  gencode::PaddlePredictor predictor;
  predictor.Init();

  LOG(INFO) << "run the generated code";
  auto input_tensor = predictor.GetInput(0);
  input_tensor->Resize(std::vector<int64_t>({100, 100}));
  auto* data = input_tensor->mutable_data<float>();
  for (int i = 0; i < 100 * 100; i++) {
    data[i] = i;
  }

  predictor.Run();

  auto output_tensor = predictor.GetOutput(0);
  LOG(INFO) << "output: " << output_tensor->data<float>()[0];
}
#endif

#ifdef LITE_WITH_ARM
TEST(PaddlePredictor, RunARM) {
  gencode::PaddlePredictor predictor;
  predictor.Init();

  LOG(INFO) << "run the generated code";
  auto input_tensor = predictor.GetInput(0);
  input_tensor->Resize(std::vector<int64_t>({1, 100}));
  auto* data = input_tensor->mutable_data<float>();
  for (int i = 0; i < 100; i++) {
    data[i] = 1;
  }

  predictor.Run();

  std::vector<float> result({0.4350058, -0.6048313, -0.29346266, 0.40377066,
                             -0.13400325, 0.37114543, -0.3407839, 0.14574292,
                             0.4104212, 0.8938774});

  auto output_tensor = predictor.GetOutput(0);
  auto output_shape = output_tensor->shape();
  ASSERT_EQ(output_shape.size(), 2);
  ASSERT_EQ(output_shape[0], 1);
  ASSERT_EQ(output_shape[1], 500);

  int step = 50;
  for (int i = 0; i < result.size(); i += step) {
    EXPECT_NEAR(output_tensor->data<float>()[i], result[i], 1e-6);
  }
}
#endif

}  // namespace lite
}  // namespace paddle
