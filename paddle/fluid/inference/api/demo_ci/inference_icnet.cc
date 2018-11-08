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
#include <chrono>  // NOLINT
#include <fstream>
#include <iostream>
#include <thread>  // NOLINT
#include <utility>
#include "paddle/fluid/inference/paddle_inference_api.h"

namespace paddle {

NativeConfig GetConfig() {
  NativeConfig config;
  config.prog_file = "hs_lb_without_bn_cudnn/__model__";
  config.param_file = "hs_lb_without_bn_cudnn/__params__";
  config.fraction_of_gpu_memory = 0.0;
  config.use_gpu = true;
  config.device = 0;
  return config;
}

using Time = decltype(std::chrono::high_resolution_clock::now());
Time TimeNow() { return std::chrono::high_resolution_clock::now(); }
double TimeDiff(Time t1, Time t2) {
  typedef std::chrono::microseconds ms;
  auto diff = t2 - t1;
  ms counter = std::chrono::duration_cast<ms>(diff);
  return counter.count() / 1000.0;
}

std::vector<PaddleTensor> PrepareData() {
  int height = 449;
  int width = 581;
  std::vector<float> data;
  for (int i = 0; i < 3 * height * width; ++i) {
    data.push_back(0.0);
  }
  PaddleTensor tensor;
  tensor.shape = std::vector<int>({batch_size, 3, height, width});
  tensor.data.Resize(sizeof(float) * batch_size * 3 * height * width);
  std::copy(data.begin(), data.end(), static_cast<float*>(tensor.data.data()));
  tensor.dtype = PaddleDType::FLOAT32;
  std::vector<PaddleTensor> paddle_tensor_feeds(1, tensor);
  return std::move(paddle_tensor_feeds);
}

void TestNaive(int batch_size, int thread_num) {
  NativeConfig config = GetConfig();

  int num_jobs = thread_num;   // parallel jobs.
  constexpr int epoches = 10;  // each job run epoches.
  std::vector<std::thread> threads;
  std::vector<std::unique_ptr<PaddlePredictor>> predictors;
  for (int tid = 0; tid < num_jobs; ++tid) {
    auto& pred = CreatePaddlePredictor<NativeConfig>(config);
    predictors.emplace_back(std::move(pred));
  }

  auto time1 = TimeNow();
  for (int tid = 0; tid < num_jobs; ++tid) {
    threads.emplace_back([&, tid]() {
      auto& predictor = predictors[tid];
      PaddleTensor tensor_out;
      std::vector<PaddleTensor> outputs(1, tensor_out);
      for (size_t i = 0; i < epoches; i++) {
        ASSERT_TRUE(predictor->Run(paddle_tensor_feeds, &outputs));
        VLOG(3) << "tid : " << tid << " run: " << i << "finished";
        ASSERT_EQ(outputs.size(), 1UL);
      }
    });
  }
  for (int i = 0; i < num_jobs; ++i) {
    threads[i].join();
  }
  auto time2 = TimeNow();
  VLOG(3) << "Thread num " << thread_num << "total time cost"
          << (time2 - time1);
}
}  // namespace paddle

int main(int argc, char** argv) {
  paddle::TestNaive(1, 1);  // single thread.
  paddle::TestNaive(1, 5);  // 5 threads.
  return 0;
}
