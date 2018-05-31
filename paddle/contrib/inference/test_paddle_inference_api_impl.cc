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

NativeConfig GetConfig() {
  NativeConfig config;
  config.model_dir = FLAGS_dirname + "word2vec.inference.model";
  LOG(INFO) << "dirname  " << config.model_dir;
  config.fraction_of_gpu_memory = 0.15;
  config.use_gpu = true;
  config.device = 0;
  config.share_variables = true;
  return config;
}

TEST(paddle_inference_api_impl, word2vec) {
  NativeConfig config = GetConfig();
  auto predictor = CreatePaddlePredictor<NativeConfig>(config);

  framework::LoDTensor first_word, second_word, third_word, fourth_word;
  framework::LoD lod{{0, 1}};
  int64_t dict_size = 2073;  // The size of dictionary

  SetupLoDTensor(&first_word, lod, static_cast<int64_t>(0), dict_size - 1);
  SetupLoDTensor(&second_word, lod, static_cast<int64_t>(0), dict_size - 1);
  SetupLoDTensor(&third_word, lod, static_cast<int64_t>(0), dict_size - 1);
  SetupLoDTensor(&fourth_word, lod, static_cast<int64_t>(0), dict_size - 1);

  std::vector<PaddleTensor> paddle_tensor_feeds;
  paddle_tensor_feeds.push_back(LodTensorToPaddleTensor(&first_word));
  paddle_tensor_feeds.push_back(LodTensorToPaddleTensor(&second_word));
  paddle_tensor_feeds.push_back(LodTensorToPaddleTensor(&third_word));
  paddle_tensor_feeds.push_back(LodTensorToPaddleTensor(&fourth_word));

  std::vector<PaddleTensor> outputs;
  ASSERT_TRUE(predictor->Run(paddle_tensor_feeds, &outputs));
  ASSERT_EQ(outputs.size(), 1UL);
  size_t len = outputs[0].data.length;
  float* data = static_cast<float*>(outputs[0].data.data);
  for (int j = 0; j < len / sizeof(float); ++j) {
    ASSERT_LT(data[j], 1.0);
    ASSERT_GT(data[j], -1.0);
  }

  std::vector<paddle::framework::LoDTensor*> cpu_feeds;
  cpu_feeds.push_back(&first_word);
  cpu_feeds.push_back(&second_word);
  cpu_feeds.push_back(&third_word);
  cpu_feeds.push_back(&fourth_word);

  framework::LoDTensor output1;
  std::vector<paddle::framework::LoDTensor*> cpu_fetchs1;
  cpu_fetchs1.push_back(&output1);

  TestInference<platform::CPUPlace>(config.model_dir, cpu_feeds, cpu_fetchs1);

  float* lod_data = output1.data<float>();
  for (size_t i = 0; i < output1.numel(); ++i) {
    EXPECT_LT(lod_data[i] - data[i], 1e-3);
    EXPECT_GT(lod_data[i] - data[i], -1e-3);
  }

  free(outputs[0].data.data);
}

TEST(paddle_inference_api_impl, image_classification) {
  int batch_size = 2;
  bool use_mkldnn = false;
  bool repeat = false;
  NativeConfig config = GetConfig();
  config.model_dir =
      FLAGS_dirname + "image_classification_resnet.inference.model";

  const bool is_combined = false;
  std::vector<std::vector<int64_t>> feed_target_shapes =
      GetFeedTargetShapes(config.model_dir, is_combined);

  framework::LoDTensor input;
  // Use normilized image pixels as input data,
  // which should be in the range [0.0, 1.0].
  feed_target_shapes[0][0] = batch_size;
  framework::DDim input_dims = framework::make_ddim(feed_target_shapes[0]);
  SetupTensor<float>(
      &input, input_dims, static_cast<float>(0), static_cast<float>(1));
  std::vector<framework::LoDTensor*> cpu_feeds;
  cpu_feeds.push_back(&input);

  framework::LoDTensor output1;
  std::vector<framework::LoDTensor*> cpu_fetchs1;
  cpu_fetchs1.push_back(&output1);

  TestInference<platform::CPUPlace, false, true>(config.model_dir,
                                                 cpu_feeds,
                                                 cpu_fetchs1,
                                                 repeat,
                                                 is_combined,
                                                 use_mkldnn);

  auto predictor = CreatePaddlePredictor(config);
  std::vector<PaddleTensor> paddle_tensor_feeds;
  paddle_tensor_feeds.push_back(LodTensorToPaddleTensor(&input));

  std::vector<PaddleTensor> outputs;
  ASSERT_TRUE(predictor->Run(paddle_tensor_feeds, &outputs));
  ASSERT_EQ(outputs.size(), 1UL);
  size_t len = outputs[0].data.length;
  float* data = static_cast<float*>(outputs[0].data.data);
  float* lod_data = output1.data<float>();
  for (size_t j = 0; j < len / sizeof(float); ++j) {
    EXPECT_NEAR(lod_data[j], data[j], 1e-3);
  }
  free(data);
}

}  // namespace paddle
