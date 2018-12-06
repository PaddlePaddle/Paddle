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

#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "inference_icnet.h"
#include "inference_icnet_preprocess.h"
#include "paddle/include/paddle_inference_api.h"

using namespace paddle;

class Predictor {
 private:
  std::unique_ptr<PaddlePredictor> predictor;
  using Time = decltype(std::chrono::high_resolution_clock::now());

  Time time() { return std::chrono::high_resolution_clock::now(); };

  double time_diff(Time t1, Time t2) {
    typedef std::chrono::microseconds ms;
    auto diff = t2 - t1;
    ms counter = std::chrono::duration_cast<ms>(diff);
    return counter.count() / 1000.0;
  }

 public:
  Predictor(const char* prog_file, const char* param_file,
            const float fraction_of_gpu_memory, const bool use_gpu,
            const int device) {
    NativeConfig config;
    config.prog_file = prog_file;
    config.param_file = param_file;
    config.fraction_of_gpu_memory = fraction_of_gpu_memory;
    config.use_gpu = use_gpu;
    config.device = device;

    predictor = CreatePaddlePredictor<NativeConfig>(config);
  }

  ~Predictor() { predictor = nullptr; }

  void predict(float* input, const int channel, const int height,
               const int width, void* output, int& output_length,
               int batch_size) {
    int intput_length = channel * height * width * batch_size;
    // initialize the input data
    PaddleTensor tensor;
    tensor.shape = std::vector<int>({batch_size, channel, height, width});
    tensor.data.Resize(sizeof(float) * batch_size * channel * height * width);
    std::memcpy(static_cast<void*>(tensor.data.data()), input,
                intput_length * sizeof(float));

    tensor.dtype = PaddleDType::FLOAT32;
    std::vector<PaddleTensor> paddle_tensor_feeds(1, tensor);

    // initialize the output data
    PaddleTensor tensor_out;
    std::vector<PaddleTensor> outputs(1, tensor_out);

    predictor->Run(paddle_tensor_feeds, &outputs, batch_size);

    // copy the data out
    std::memcpy(static_cast<void*>(output), outputs[0].data.data(),
                outputs[0].data.length());
    output_length = (int)outputs[0].data.length();
  }
};

API_REFERENCE void* init_predictor(const char* prog_file,
                                   const char* param_file,
                                   const float fraction_of_gpu_memory,
                                   const bool use_gpu, const int device) {
  return new Predictor(prog_file, param_file, fraction_of_gpu_memory, use_gpu,
                       device);
}

API_REFERENCE void predict(void* handle, float* input, const int channel,
                           const int height, const int width, void* output,
                           int& output_length, int batch_size) {
  assert(handle != nullptr);
  ((Predictor*)handle)
      ->predict(input, channel, height, width, output, output_length,
                batch_size);
}

API_REFERENCE void predict_file(void* handle, const char* bmp_name,
                                void* output, int& output_length) {
  assert(handle != nullptr);
  Record record;
  if (ImageProcess::preprocess_image(record, bmp_name)) {
    ((Predictor*)handle)
        ->predict(record.data, C, H, W, output, output_length, 1);
  }
}

API_REFERENCE void destory_predictor(void* handle) {
  if (handle != nullptr) {
    delete (Predictor*)handle;
    handle = nullptr;
  }
}

API_REFERENCE void save_image(const char* filename, const void* output,
                              const int output_length) {
  ImageProcess::bmp_save(filename, (int64_t*)output, output_length);
}
