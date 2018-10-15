// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <chrono>
#include <iostream>
#include <fstream>
#include "paddle/fluid/inference/api/paddle_inference_api.h"

namespace paddle {

std::string DIRNAME = "./LB_icnet_model";
//std::string DIRNAME = "./infer_models";
NativeConfig GetConfig() {
  NativeConfig config;
  config.prog_file=DIRNAME + "/__model__";
  config.param_file=DIRNAME + "/__params__";
  config.fraction_of_gpu_memory = 0.8;
  config.use_gpu = true;
  config.device = 0;
  return config;
}

using Time = decltype(std::chrono::high_resolution_clock::now());
Time time() { return std::chrono::high_resolution_clock::now(); };
double time_diff(Time t1, Time t2) {
  typedef std::chrono::microseconds ms;
  auto diff = t2 - t1;
  ms counter = std::chrono::duration_cast<ms>(diff);
  return counter.count() / 1000.0;
}

void test_naive(int batch_size){
  NativeConfig config = GetConfig();
  // config.model_dir = model_path;
  auto predictor = CreatePaddlePredictor<NativeConfig>(config);
  int height = 449;
  int width = 581;
  //int height = 3;
  //int width = 3;
  int num_sum = height * width * 3 * batch_size;
  
  std::vector<float> data;
  
  for(int i = 0; i < num_sum; i++) {
    data.push_back(0.0);
  }
  
  PaddleTensor tensor;
  tensor.shape = std::vector<int>({batch_size, 3, height, width});
  tensor.data.Resize(sizeof(float) * batch_size * 3 * height * width);
  std::copy(data.begin(), data.end(), static_cast<float*>(tensor.data.data()));
  tensor.dtype = PaddleDType::FLOAT32;
  std::vector<PaddleTensor> paddle_tensor_feeds(1, tensor);
  PaddleTensor tensor_out;

  std::vector<PaddleTensor> outputs(1, tensor_out);

  predictor->Run(paddle_tensor_feeds, &outputs, batch_size);
  std::cout << "start predict123:" << std::endl;
  auto time1 = time(); 
  
  for(size_t i = 0; i < 2; i++) {
    predictor->Run(paddle_tensor_feeds, &outputs, batch_size);
    std::cout << "pass " << i;
  } 

  auto time2 = time(); 
  std::ofstream ofresult("naive_test_result.txt", std::ios::app);

  std::cout <<"batch: " << batch_size << " predict cost: " << time_diff(time1, time2) / 100.0 << "ms" << std::endl;
  std::cout << outputs.size() << std::endl;
  /*
  int64_t * data_o = static_cast<int64_t*>(outputs[0].data.data());
  for (size_t j = 0; j < outputs[0].data.length() / sizeof(int64_t); ++j) {
    ofresult << std::to_string(data_o[j]) << " ";
  }
  ofresult << std::endl;
  ofresult.close();
  */
}
}  // namespace paddle

int main(int argc, char** argv) {
  paddle::test_naive(1 << 0);
  return 0;
}