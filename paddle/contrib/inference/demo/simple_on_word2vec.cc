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
#include "paddle/contrib/inference/paddle_inference_api.h"

namespace paddle {
namespace demo {

DEFINE_string(dirname, "", "Directory of the inference model.");

void PrepareInputs(std::vector<PaddleTensor>* slots) {
  const int num_slots = 4;
  const int data_size = 6;
  slots->clear();
  // word id type is int64_t
  std::unique_ptr<std::vector<int64_t>> data(
      new std::vector<int64_t>(data_size));

  // Set scratch data
  for (int i = 0; i < data_size; i++) {
    data->at(i) = i;
  }

  // For simplicity, we set all the slots with the same data.
  for (int i = 0; i < num_slots; i++) {
    PaddleBuf buf{.data = data.get(), .length = data_size};
    slots->emplace_back(PaddleTensor{.data = buf,
                                     .dtype = PaddleDType::INT64,
                                     .shape = std::vector<int>({6, 1})});
  }
}

void Main() {
  std::vector<PaddleTensor> slots;
  PrepareInputs(&slots);

  NativeConfig config{.use_gpu = true,
                      .fraction_of_gpu_memory = 0.3,
                      .device = 0,
                      .model_dir = FLAGS_dirname + "word2vec.inference.model"};
  auto predictor =
      CreatePaddlePredictor<NativeConfig, PaddleEngineKind::kNative>(config);

  std::vector<PaddleTensor> outputs;

  CHECK(predictor->Run(slots, &outputs));
}

TEST(demo, word2vec) { Main(); }

}  // namespace demo
}  // namespace paddle
