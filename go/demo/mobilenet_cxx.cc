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
#include <paddle_inference_api.h>
#include <fstream>
#include <iostream>

void SetConfig(paddle::AnalysisConfig *);

int main(int argc, char *argv[]) {
  paddle::AnalysisConfig config;
  SetConfig(&config);
  auto predictor = paddle::CreatePaddlePredictor(config);
  auto input_name = predictor->GetInputNames()[0];
  auto input = predictor->GetInputTensor(input_name);
  std::cout << predictor->GetOutputNames()[0] << std::endl;
  std::vector<int> shape{1, 3, 300, 300};
  input->Reshape(std::move(shape));
  std::vector<float> data(1 * 300 * 300 * 3);
  std::ifstream fin("data/data.txt");
  for (int i = 0; i < data.size(); i++) {
    fin >> data[i];
  }

  input->copy_from_cpu(data.data());
  predictor->ZeroCopyRun();
  auto output_name = predictor->GetOutputNames()[0];
  auto output = predictor->GetOutputTensor(output_name);
  return 0;
}

void SetConfig(paddle::AnalysisConfig *config) {
  config->SetModel("data/model/__model__", "data/model/__params__");
  config->SwitchUseFeedFetchOps(false);
  config->SwitchSpecifyInputNames(true);
  config->SwitchIrOptim(false);
}
