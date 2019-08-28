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

#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <glog/logging.h>
#include <vector>
#include "lite/api/paddle_api.h"
#include "lite/api/paddle_place.h"

namespace paddle {
namespace lite {

using paddle::lite_api::Place;

void TestModel(const std::vector<Place>& valid_places,
               const Place& preferred_place,
               bool use_npu = false) {
  lite_api::CxxConfig cfg; 
  cfg.set_model_dir("/shixiaowei02/Paddle_lite/xingzhaolong/leaky_relu_model");
  cfg.set_preferred_place(preferred_place);
  cfg.set_valid_places(valid_places);
  auto predictor = lite_api::CreatePaddlePredictor(cfg);

  auto input_tensor = predictor->GetInput(0);
  input_tensor->Resize(std::vector<int64_t>({1, 1, 3, 3}));
  auto* data = input_tensor->mutable_data<float>();

  auto input_shape = input_tensor->shape(); 
  int item_size = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int64_t>()); 
  for (int i = 0; i < item_size; i++) {
    data[i] = -1.;
  }

  predictor->Run();

  auto out = predictor->GetOutput(0);
  for (int i = 0; i < item_size; i++) {
    LOG(INFO) << out->data<float>()[i];
  }
}

TEST(MobileNetV1, test_arm) {
  std::vector<Place> valid_places({
      Place{TARGET(kHost), PRECISION(kFloat)},
      Place{TARGET(kCUDA), PRECISION(kFloat)},
  });

  TestModel(valid_places, Place({TARGET(kCUDA), PRECISION(kFloat)}));
}

}  // namespace lite
}  // namespace paddle
