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

#define GOOGLE_GLOG_DLL_DECL

#include <gflags/gflags.h>
#include <glog/logging.h>
//#include <gtest/gtest.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include <thread>  // NOLINT

#define ASSERT_TRUE(x) x
#define ASSERT_EQ(x, y) assert(x == y)

namespace paddle {

// DEFINE_string(dirname, "./LB_icnet_model",
//               "Directory of the inference model.");

NativeConfig GetConfig() {
  NativeConfig config;
  config.prog_file= "./dzh_lb/__model__";
  config.param_file= "./dzh_lb/__params__";
  config.fraction_of_gpu_memory = 0.08;
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

void test_naive(int batch_size, std::string model_path){
  PaddlePredictor* pres[2];
  
  NativeConfig config = GetConfig();
  // config.model_dir = model_path;
  auto predictor0 = CreatePaddlePredictor<NativeConfig>(config);
  auto predictor1 = CreatePaddlePredictor<NativeConfig>(config);
  pres[0] = predictor0.get();
  pres[1] = predictor1.get();

  int height = 449;
  int width = 581;
  
  std::vector<float> data;
  for (int i = 0; i < 3 * height * width; i++) {
    data.push_back(0);
  }
  
  PaddleTensor tensor;
  tensor.shape = std::vector<int>({batch_size, 3, height, width});
  tensor.data.Resize(sizeof(float) * batch_size * 3 * height * width);
  std::copy(data.begin(), data.end(), static_cast<float*>(tensor.data.data()));
  tensor.dtype = PaddleDType::FLOAT32;
  std::vector<PaddleTensor> paddle_tensor_feeds(1, tensor);

  constexpr int num_jobs = 5;  // each job run 1 batch
  std::vector<std::thread> threads;
  for (int tid = 0; tid < num_jobs; ++tid) {
    threads.emplace_back([&, tid]() {
      auto predictor = pres[tid];
      std::vector<PaddleTensor> local_outputs;
     for(size_t i = 0; i < 1000; i++) {
      ASSERT_TRUE(predictor->Run(paddle_tensor_feeds, &local_outputs));
      std::cout << "run: " << tid << std::endl; 
      }
      ASSERT_EQ(local_outputs.size(), 1UL);
    });
  }
  for (int i = 0; i < num_jobs; ++i) {
    threads[i].join();
  }
}

//TEST(alexnet, naive) {
//  test_naive(1 << 0, "./trt_models/vgg19");
//}

}  // namespace paddle

int main(int argc, char** argv) {
	paddle::test_naive(1 << 0, "");
}

