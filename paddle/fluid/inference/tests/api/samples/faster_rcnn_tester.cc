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

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "common.h"

namespace paddle {
namespace inference {

void SetInputs(std::vector<paddle::PaddleTensor> &input_tensors) {
  //
  // input image
  //
  paddle::PaddleTensor image_tensor;

  image_tensor.name = "image";
  image_tensor.dtype = paddle::PaddleDType::FLOAT32;
  SetupTensor<float>(image_tensor, {1, 3, 1333, 1333}, static_cast<float>(0),
                     static_cast<float>(255));

  //
  // input im_info
  //
  paddle::PaddleTensor im_info_tensor;

  im_info_tensor.name = "im_info";
  im_info_tensor.dtype = paddle::PaddleDType::FLOAT32;
  SetupTensor<float>(im_info_tensor, {1, 3}, static_cast<float>(0),
                     static_cast<float>(1));

  // input_tensors
  input_tensors.push_back(image_tensor);
  input_tensors.push_back(im_info_tensor);
}

void profile(std::string model_dir, bool use_analysis, bool use_tensorrt) {
  std::vector<paddle::PaddleTensor> inputs;
  SetInputs(inputs);

  std::vector<std::vector<PaddleTensor>> inputs_all;
  inputs_all.push_back(inputs);

  std::vector<paddle::PaddleTensor> outputs;
  if (use_analysis || use_tensorrt) {
    contrib::AnalysisConfig config(true);
    SetConfig<contrib::AnalysisConfig>(&config, model_dir, true, use_tensorrt,
                                       FLAGS_batch_size);
    TestPrediction(reinterpret_cast<PaddlePredictor::Config *>(&config),
                   inputs_all, &outputs, FLAGS_num_threads, true);
  } else {
    NativeConfig config;
    SetConfig<NativeConfig>(&config, model_dir, true, false);
    TestPrediction(reinterpret_cast<PaddlePredictor::Config *>(&config),
                   inputs_all, &outputs, FLAGS_num_threads, false);
  }
}

TEST(video, profile) {
  std::string model_dir = FLAGS_infer_model;
  profile(model_dir, /* use_analysis */ true, FLAGS_use_tensorrt);
}

}  // namespace inference
}  // namespace paddle
