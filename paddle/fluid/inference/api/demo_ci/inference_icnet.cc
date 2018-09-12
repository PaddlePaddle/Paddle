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
#include <cassert>
#include <cctype>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <thread>  //NOLINT
#include "paddle/fluid/inference/paddle_inference_api.h"

std::string DIRNAME = ""; /* "Directory of the inference model." */ // NOLINT
bool USE_GPU = false;     /*"Whether use gpu."*/

auto message_err = []() {
  std::cout << "Copyright (c) 2018 PaddlePaddle Authors." << std::endl;
  std::cout << "Demo Case for windows inference. "
            << "\n"
            << "Usage: Input your model path and use_gpu as the guide requires,"
            << "then run the demo inference, and will get a result."
            << std::endl;
  std::cout << std::endl;
};

void ParseArgs() {
  message_err();
  std::cout << "DIRNAME:[D:/Paddle/xxx/path_to_model_dir]" << std::endl;
  std::cin >> DIRNAME;
  std::cout << "USE_GPU:[yes|no]";
  std::string value;
  std::cin >> value;
  std::transform(value.begin(), value.end(), value.begin(), ::toupper);
  USE_GPU = (value == "YES") ? true : false;
}

namespace paddle {
namespace demo {
std::string ToString(const NativeConfig& config) {
  std::stringstream ss;
  ss << "Use GPU : " << (config.use_gpu ? "True" : "False") << "\n"
     << "Device : " << config.device << "\n"
     << "fraction_of_gpu_memory : " << config.fraction_of_gpu_memory << "\n"
     << "specify_input_name : "
     << (config.specify_input_name ? "True" : "False") << "\n"
     << "Program File : " << config.prog_file << "\n"
     << "Param File : " << config.param_file;
  return ss.str();
}

void Main(bool use_gpu) {
  //# 1. Create PaddlePredictor with a config.
  NativeConfig config;
  config.model_dir = DIRNAME;
  config.use_gpu = USE_GPU;
  config.fraction_of_gpu_memory = 0.15;
  config.device = 0;
  std::cout << ToString(config) << std::endl;
  auto predictor =
      CreatePaddlePredictor<NativeConfig, PaddleEngineKind::kNative>(config);

  for (int batch_id = 0; batch_id < 3; batch_id++) {
    //# 2. Prepare input.
    int64_t data[4] = {1, 2, 3, 4};

    PaddleTensor tensor;
    tensor.shape = std::vector<int>({4, 1});
    tensor.data = PaddleBuf(data, sizeof(data));
    tensor.dtype = PaddleDType::INT64;

    // For simplicity, we set all the slots with the same data.
    std::vector<PaddleTensor> slots(4, tensor);

    //# 3. Run
    std::vector<PaddleTensor> outputs;
    assert(predictor->Run(slots, &outputs) == true &&
           "Predict run expect true");

    //# 4. Get output.
    assert(outputs.size() == 1UL);
    // Check the output buffer size and result of each tid.
    assert(outputs.front().data.length() == 33168UL);
    float result[5] = {0.00129761, 0.00151112, 0.000423564, 0.00108815,
                       0.000932706};
    const size_t num_elements = outputs.front().data.length() / sizeof(float);
    // The outputs' buffers are in CPU memory.
    for (size_t i = 0; i < std::min(static_cast<size_t>(5), num_elements);
         i++) {
      assert(static_cast<float*>(outputs.front().data.data())[i] == result[i]);
      std::cout << "expect the output "
                << static_cast<float*>(outputs.front().data.data())[i]
                << std::endl;
    }
  }
}

void MainThreads(int num_threads, bool USE_GPU) {
  // Multi-threads only support on CPU
  // 0. Create PaddlePredictor with a config.
  NativeConfig config;
  config.model_dir = DIRNAME;
  config.use_gpu = USE_GPU;
  config.fraction_of_gpu_memory = 0.15;
  config.device = 0;
  std::cout << ToString(config) << std::endl;
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
        PaddleTensor tensor;
        tensor.shape = std::vector<int>({4, 1});
        tensor.data = PaddleBuf(data, sizeof(data));
        tensor.dtype = PaddleDType::INT64;

        std::vector<PaddleTensor> inputs(4, tensor);
        std::vector<PaddleTensor> outputs;
        // 3. Run
        assert(predictor->Run(inputs, &outputs) == true);

        // 4. Get output.
        assert(outputs.size() == 1UL);
        // Check the output buffer size and result of each tid.
        assert(outputs.front().data.length() == 33168UL);
        float result[5] = {0.00129761, 0.00151112, 0.000423564, 0.00108815,
                           0.000932706};
        const size_t num_elements =
            outputs.front().data.length() / sizeof(float);
        // The outputs' buffers are in CPU memory.
        for (size_t i = 0; i < std::min(static_cast<size_t>(5), num_elements);
             i++) {
          assert(static_cast<float*>(outputs.front().data.data())[i] ==
                 result[i]);
        }
      }
    });
  }
  for (int i = 0; i < num_threads; ++i) {
    threads[i].join();
  }
}

}  // namespace demo
}  // namespace paddle

int main(int argc, char** argv) {
  // ParseArgs();
  DIRNAME = "./icnet";
  USE_GPU = true;
  paddle::demo::Main(false /* USE_GPU*/);
  paddle::demo::MainThreads(1, false /* USE_GPU*/);
  paddle::demo::MainThreads(4, false /* USE_GPU*/);
  if (USE_GPU) {
    paddle::demo::Main(true /*USE_GPU*/);
    paddle::demo::MainThreads(1, true /*USE_GPU*/);
    paddle::demo::MainThreads(4, true /*USE_GPU*/);
  }
  system("pause");
  return 0;
}
