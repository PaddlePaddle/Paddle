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

TEST(PaddlePredictor, Run) {
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

}  // namespace lite
}  // namespace paddle
