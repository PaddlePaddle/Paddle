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

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "common.h"

namespace paddle {
namespace inference {

void SetInputs(std::vector<paddle::PaddleTensor> &input_tensors,
               std::string &data_dir) {
  //
  // input rgb
  //
  paddle::PaddleTensor rgb_tensor;

  rgb_tensor.name = "rgb";
  rgb_tensor.dtype = paddle::PaddleDType::FLOAT32;
  rgb_tensor.lod = {{0, 200}};
  SetupTensor<float>(data_dir + "/rgb.txt", rgb_tensor, {200, 2048});

  //
  // input audio
  //
  paddle::PaddleTensor audio_tensor;

  audio_tensor.name = "audio";
  audio_tensor.dtype = paddle::PaddleDType::FLOAT32;
  audio_tensor.lod = {{0, 200}};
  SetupTensor<float>(data_dir + "/audio.txt", audio_tensor, {200, 2048});

  //
  // input title
  //
  paddle::PaddleTensor title_tensor;

  title_tensor.name = "title";
  title_tensor.dtype = paddle::PaddleDType::INT64;
  title_tensor.lod = {{0, 20}};
  SetupTensor<int64_t>(data_dir + "/title.txt", title_tensor, {20, 1});

  // input_tensors
  input_tensors.push_back(rgb_tensor);
  input_tensors.push_back(audio_tensor);
  input_tensors.push_back(title_tensor);
}

void profile(std::string model_dir, bool use_analysis, bool use_tensorrt) {
  std::vector<paddle::PaddleTensor> inputs;
  SetInputs(inputs, FLAGS_infer_data);

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
