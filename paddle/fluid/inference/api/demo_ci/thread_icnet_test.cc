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
#include <thread>  // NOLINT
#include <utility>
#include "paddle/fluid/inference/api/paddle_inference_api.h"

#define ASSERT_TRUE(x) x
#define ASSERT_EQ(x, y) assert(x == y)

// DEFINE_string(dirname, "./LB_icnet_model",
//               "Directory of the inference model.");
namespace paddle {
NativeConfig GetConfig() {
  NativeConfig config;
  config.prog_file = "./hs_lb_without_bn_cuda/__model__";
  config.param_file = "./hs_lb_without_bn_cuda/__params__";
  config.fraction_of_gpu_memory = 0.0;
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

void test_naive(int batch_size, std::string model_path) {
  NativeConfig config = GetConfig();
  int height = 449;
  int width = 581;
  std::vector<float> data;
  for (int i = 0; i < 3 * height * width; ++i) {
    data.push_back(0.0);
  }

  // read data
  // std::ifstream infile("new_file.list");
  // std::string temp_s;
  // std::vector<std::string> all_files;
  // while (!infile.eof()) {
  //   infile >> temp_s;
  //   all_files.push_back(temp_s);
  // }

  // // size_t file_num = all_files.size();
  // infile.close();
  // // =============read file list =============
  // for (size_t f_k = 0; f_k < 1; f_k++) {
  //   std::ifstream in_img(all_files[f_k]);
  //   std::cout << all_files[f_k] << std::endl;
  //   float temp_v;

  //   float sum_n = 0.0;
  //   std::vector<float> data;
  //   while (!in_img.eof()) {
  //     in_img >> temp_v;
  //     data.push_back(float(temp_v));

  //     sum_n += temp_v;
  //   }
  //   in_img.close();
  //   std::cout << "sum: " << sum_n << std::endl;

  PaddleTensor tensor;
  tensor.shape = std::vector<int>({batch_size, 3, height, width});
  tensor.data.Resize(sizeof(float) * batch_size * 3 * height * width);
  std::copy(data.begin(), data.end(), static_cast<float*>(tensor.data.data()));
  tensor.dtype = PaddleDType::FLOAT32;
  std::vector<PaddleTensor> paddle_tensor_feeds(1, tensor);

  constexpr int num_jobs = 5;  // each job run 1 batch
  std::vector<std::thread> threads;
  // using PtrPred = std::vector<std::unique_ptr<PaddlePredictor>>;
  std::vector<std::unique_ptr<PaddlePredictor>> predictors;
  for (int tid = 0; tid < num_jobs; ++tid) {
    auto& pred = CreatePaddlePredictor<NativeConfig>(config);
    predictors.emplace_back(std::move(pred));
  }

  using namespace std::chrono_literals;
  // std::this_thread::sleep_for(std::chrono::seconds(20));
  std::cout << "before start predict";

  int epoches = 100000;
  for (int tid = 0; tid < num_jobs; ++tid) {
    threads.emplace_back([&, tid]() {
      // auto predictor = CreatePaddlePredictor<NativeConfig>(config);
      auto& predictor = predictors[tid];
      // auto& predictor = predictors[tid];
      // auto predictor = preds[tid];
      // std::this_thread::sleep_for(std::chrono::seconds(20));
      PaddleTensor tensor_out;
      std::vector<PaddleTensor> outputs(1, tensor_out);
      for (size_t i = 0; i < epoches; i++) {
        ASSERT_TRUE(predictor->Run(paddle_tensor_feeds, &outputs));
        VLOG(0) << "tid : " << tid << " run: " << i << "finished";
        // std::cout <<"tid : " << tid << " run: " << i << "finished" <<
        // std::endl;
        ASSERT_EQ(outputs.size(), 1UL);
        // int64_t* data_o = static_cast<int64_t*>(outputs[0].data.data());
        // int64_t sum_out = 0;
        // for (size_t j = 0; j < outputs[0].data.length() / sizeof(int64_t);
        //      ++j) {
        //   sum_out += data_o[j];
        // }
        // std::cout << "tid : " << tid << "pass : " << i << " " << sum_out
        //           << std::endl;
      }
    });
  }
  for (int i = 0; i < num_jobs; ++i) {
    threads[i].join();
  }
}
// }
}  // namespace paddle

int main(int argc, char** argv) {
  paddle::test_naive(1 << 0, "");
  return 0;
}
