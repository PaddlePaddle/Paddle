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

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "gflags/gflags.h"
#include "paddle/contrib/inference/paddle_inference_api_impl.h"
#include "paddle/fluid/inference/tests/test_helper.h"

DEFINE_string(dirname, "", "Directory of the inference model.");

namespace paddle {

PaddleTensor LodTensorToPaddleTensor(framework::LoDTensor* t) {
  PaddleTensor pt;
  pt.data.data = t->data<void>();

  if (t->type() == typeid(int64_t)) {
    pt.data.length = t->numel() * sizeof(int64_t);
    pt.dtype = PaddleDType::INT64;
  } else if (t->type() == typeid(float)) {
    pt.data.length = t->numel() * sizeof(float);
    pt.dtype = PaddleDType::FLOAT32;
  } else {
    LOG(FATAL) << "unsupported type.";
  }
  pt.shape = framework::vectorize2int(t->dims());
  return pt;
}

TEST(paddle_inference_api_impl, word2vec) {
  VisConfig config;
  config.model_dir = FLAGS_dirname + "word2vec.inference.model";
  LOG(INFO) << "dirname  " << config.model_dir;
  config.fraction_of_gpu_memory = 0.15;
  config.device = 0;
  config.share_variables = true;

  std::unique_ptr<PaddlePredictorImpl> predictor =
      CreatePaddlePredictorImpl(config);

  framework::LoDTensor first_word, second_word, third_word, fourth_word;
  framework::LoD lod{{0, 1}};
  int64_t dict_size = 2073;  // The size of dictionary

  SetupLoDTensor(&first_word, lod, static_cast<int64_t>(0), dict_size - 1);
  SetupLoDTensor(&second_word, lod, static_cast<int64_t>(0), dict_size - 1);
  SetupLoDTensor(&third_word, lod, static_cast<int64_t>(0), dict_size - 1);
  SetupLoDTensor(&fourth_word, lod, static_cast<int64_t>(0), dict_size - 1);

  std::vector<PaddleTensor> cpu_feeds;
  cpu_feeds.push_back(LodTensorToPaddleTensor(&first_word));
  cpu_feeds.push_back(LodTensorToPaddleTensor(&second_word));
  cpu_feeds.push_back(LodTensorToPaddleTensor(&third_word));
  cpu_feeds.push_back(LodTensorToPaddleTensor(&fourth_word));

  std::vector<PaddleTensor> outputs;
  ASSERT_TRUE(predictor->Run(cpu_feeds, &outputs));
  ASSERT_EQ(outputs.size(), 1);
  for (size_t i = 0; i < outputs.size(); ++i) {
    size_t len = outputs[i].data.length;
    float* data = static_cast<float*>(outputs[i].data.data);
    for (size_t j = 0; j < len / sizeof(float); ++j) {
      ASSERT_LT(data[j], 1.0);
      ASSERT_GT(data[j], -1.0);
    }
    free(outputs[i].data.data);
  }
}

}  // namespace paddle
