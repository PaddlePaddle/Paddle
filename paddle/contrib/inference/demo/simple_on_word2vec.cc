/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

/*
 * This file contains a simple demo for how to take a model for inference.
 */

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <memory>
#include <thread>
#include "paddle/contrib/inference/paddle_inference_api.h"
namespace paddle {
namespace demo {

DEFINE_string(dirname, "", "Directory of the inference model.");

void Main(bool use_gpu) {
  //# 1. Create PaddlePredictor with a config.
  NativeConfig config;
  config.model_dir = FLAGS_dirname + "word2vec.inference.model";
  config.use_gpu = use_gpu;
  config.fraction_of_gpu_memory = 0.15;
  config.device = 0;
  auto predictor =
      CreatePaddlePredictor<NativeConfig, PaddleEngineKind::kNative>(config);

  for (int batch_id = 0; batch_id < 3; batch_id++) {
    //# 2. Prepare input.
    int64_t data[4] = {1, 2, 3, 4};

    PaddleTensor tensor{.name = "",
                        .shape = std::vector<int>({4, 1}),
                        .data = PaddleBuf(data, sizeof(data)),
                        .dtype = PaddleDType::INT64};

    // For simplicity, we set all the slots with the same data.
    std::vector<PaddleTensor> slots(4, tensor);

    //# 3. Run
    std::vector<PaddleTensor> outputs;
    CHECK(predictor->Run(slots, &outputs));

    //# 4. Get output.
    ASSERT_EQ(outputs.size(), 1UL);
    LOG(INFO) << "output buffer size: " << outputs.front().data.length();
    const size_t num_elements = outputs.front().data.length() / sizeof(float);
    // The outputs' buffers are in CPU memory.
    for (size_t i = 0; i < std::min(5UL, num_elements); i++) {
      LOG(INFO) << static_cast<float*>(outputs.front().data.data())[i];
    }
  }
}

void MainThreads(int num_threads, bool use_gpu) {
  // Multi-threads only support on CPU
  // 0. Create PaddlePredictor with a config.
  NativeConfig config;
  config.model_dir = FLAGS_dirname + "word2vec.inference.model";
  config.use_gpu = use_gpu;
  config.fraction_of_gpu_memory = 0.15;
  config.device = 0;
  auto main_predictor =
      CreatePaddlePredictor<NativeConfig, PaddleEngineKind::kNative>(config);

  std::vector<std::thread> threads;
  for (int tid = 0; tid < num_threads; ++tid) {
    threads.emplace_back([&, tid]() {
      // 1. clone a predictor which shares the same parameters
      auto predictor = main_predictor->Clone();
      constexpr int num_batches = 3;
      for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
        // 2. Dummy Input Data
        int64_t data[4] = {1, 2, 3, 4};
        PaddleTensor tensor{.name = "",
                            .shape = std::vector<int>({4, 1}),
                            .data = PaddleBuf(data, sizeof(data)),
                            .dtype = PaddleDType::INT64};
        std::vector<PaddleTensor> inputs(4, tensor);
        std::vector<PaddleTensor> outputs;
        // 3. Run
        CHECK(predictor->Run(inputs, &outputs));

        // 4. Get output.
        ASSERT_EQ(outputs.size(), 1UL);
        LOG(INFO) << "TID: " << tid << ", "
                  << "output buffer size: " << outputs.front().data.length();
        const size_t num_elements =
            outputs.front().data.length() / sizeof(float);
        // The outputs' buffers are in CPU memory.
        for (size_t i = 0; i < std::min(5UL, num_elements); i++) {
          LOG(INFO) << static_cast<float*>(outputs.front().data.data())[i];
        }
      }
    });
  }
  for (int i = 0; i < num_threads; ++i) {
    threads[i].join();
  }
}

TEST(demo, word2vec_cpu) { Main(false /*use_gpu*/); }
TEST(demo_multi_threads, word2vec_cpu_1) { MainThreads(1, false /*use_gpu*/); }
TEST(demo_multi_threads, word2vec_cpu_4) { MainThreads(4, false /*use_gpu*/); }

#ifdef PADDLE_WITH_CUDA
TEST(demo, word2vec_gpu) { Main(true /*use_gpu*/); }
TEST(demo_multi_threads, word2vec_gpu_1) { MainThreads(1, true /*use_gpu*/); }
TEST(demo_multi_threads, word2vec_gpu_4) { MainThreads(4, true /*use_gpu*/); }
#endif

}  // namespace demo
}  // namespace paddle
