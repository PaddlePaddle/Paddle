/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/inference/api/paddle_inference_api.h"

DEFINE_string(model, "", "Directory of the inference model.");

namespace paddle {

contrib::AnakinConfig Config() {
  // Determine the use of memory here.
  std::map<std::string, std::vector<int>> init_inputs_shape;
  init_inputs_shape["input_0"] = std::vector<int>({1, 3, 112, 112});

  contrib::AnakinConfig config;
  config.target_type = contrib::AnakinConfig::MLU;
  config.model_file = FLAGS_model;
  config.init_inputs_shape = init_inputs_shape;

  // Determine the device execution context.
  config.device_id = 0;
  config.data_stream_id = 0;
  config.compute_stream_id = 0;

  // Set re_allocable and op_fuse TRUE.
  config.re_allocable = true;
  config.op_fuse = true;

  return config;
}

void single_test() {
  // 1. Defining basic data structures.
  auto config = paddle::Config();
  auto predictor =
      paddle::CreatePaddlePredictor<paddle::contrib::AnakinConfig,
                                    paddle::PaddleEngineKind::kAnakin>(config);

  // 2. Define the data structure of the predictor inputs and outputs.
  std::vector<paddle::PaddleTensor> input_tensors;
  std::vector<paddle::PaddleTensor> output_tensors;

  // 3. Define and fill the inputs tensor.
  int num = 1;
  int channel = 3;
  int height = 112;
  int width = 112;
  std::vector<float> input(num * channel * height * width, 1);
  std::vector<std::vector<float>> inputs({input});
  const std::vector<std::string> input_names{"input_0"};
  for (auto& name : input_names) {
    paddle::PaddleTensor tensor;
    tensor.name = name;
    tensor.dtype = PaddleDType::FLOAT32;
    input_tensors.push_back(tensor);
  }
  for (size_t j = 0; j < input_tensors.size(); j++) {
    input_tensors[j].data =
        paddle::PaddleBuf(&inputs[j][0], inputs[j].size() * sizeof(float));
    // The shape of each execution can be changed.
    input_tensors[j].shape = std::vector<int>({num, channel, height, width});
  }

  // 4. Set the output placeholder of predictor.
  PaddleTensor predict_out, score_out;
  predict_out.name = "landmark_predict_out";
  score_out.name = "landmark_score_out";
  output_tensors.push_back(predict_out);
  output_tensors.push_back(score_out);

  // 5. Execution predict.
  predictor->Run(input_tensors, &output_tensors);

  // 6. Take out the output data.
  for (auto out : output_tensors) {
    float* data_o = static_cast<float*>(out.data.data());
    LOG(INFO) << out.name << " size = " << out.data.length() / sizeof(float);
  }
}
}  // namespace paddle

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  paddle::single_test();
  return 0;
}
