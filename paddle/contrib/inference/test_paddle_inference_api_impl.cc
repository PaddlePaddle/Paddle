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

#include <thread>

#include "gflags/gflags.h"
#include "paddle/contrib/inference/paddle_inference_api_impl.h"
#include "paddle/fluid/inference/tests/test_helper.h"

DEFINE_string(dirname, "", "Directory of the inference model.");

namespace paddle {

PaddleTensor LodTensorToPaddleTensor(framework::LoDTensor* t) {
  PaddleTensor pt;

  if (t->type() == typeid(int64_t)) {
    pt.data.Reset(t->data<void>(), t->numel() * sizeof(int64_t));
    pt.dtype = PaddleDType::INT64;
  } else if (t->type() == typeid(float)) {
    pt.data.Reset(t->data<void>(), t->numel() * sizeof(float));
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
#ifdef PADDLE_WITH_CUDA
  config.use_gpu = true;
#else
  config.use_gpu = false;
#endif
  config.device = 0;
  return config;
}

void MainWord2Vec(bool use_gpu) {
  NativeConfig config = GetConfig();
  auto predictor = CreatePaddlePredictor<NativeConfig>(config);
  config.use_gpu = use_gpu;

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
  size_t len = outputs[0].data.length();
  float* data = static_cast<float*>(outputs[0].data.data());
  for (size_t j = 0; j < len / sizeof(float); ++j) {
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
  for (int i = 0; i < output1.numel(); ++i) {
    EXPECT_LT(lod_data[i] - data[i], 1e-3);
    EXPECT_GT(lod_data[i] - data[i], -1e-3);
  }
}

void MainImageClassification(bool use_gpu) {
  int batch_size = 2;
  bool repeat = false;
  NativeConfig config = GetConfig();
  config.use_gpu = use_gpu;
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

  TestInference<platform::CPUPlace, false, true>(
      config.model_dir, cpu_feeds, cpu_fetchs1, repeat, is_combined);

  auto predictor = CreatePaddlePredictor(config);
  std::vector<PaddleTensor> paddle_tensor_feeds;
  paddle_tensor_feeds.push_back(LodTensorToPaddleTensor(&input));

  std::vector<PaddleTensor> outputs;
  ASSERT_TRUE(predictor->Run(paddle_tensor_feeds, &outputs));
  ASSERT_EQ(outputs.size(), 1UL);
  size_t len = outputs[0].data.length();
  float* data = static_cast<float*>(outputs[0].data.data());
  float* lod_data = output1.data<float>();
  for (size_t j = 0; j < len / sizeof(float); ++j) {
    EXPECT_NEAR(lod_data[j], data[j], 1e-3);
  }
}

void MainThreadsWord2Vec(bool use_gpu) {
  NativeConfig config = GetConfig();
  config.use_gpu = use_gpu;
  auto main_predictor = CreatePaddlePredictor<NativeConfig>(config);

  // prepare inputs data and reference results
  constexpr int num_jobs = 3;
  std::vector<std::vector<framework::LoDTensor>> jobs(num_jobs);
  std::vector<std::vector<PaddleTensor>> paddle_tensor_feeds(num_jobs);
  std::vector<framework::LoDTensor> refs(num_jobs);
  for (size_t i = 0; i < jobs.size(); ++i) {
    // each job has 4 words
    jobs[i].resize(4);
    for (size_t j = 0; j < 4; ++j) {
      framework::LoD lod{{0, 1}};
      int64_t dict_size = 2073;  // The size of dictionary
      SetupLoDTensor(&jobs[i][j], lod, static_cast<int64_t>(0), dict_size - 1);
      paddle_tensor_feeds[i].push_back(LodTensorToPaddleTensor(&jobs[i][j]));
    }

    // get reference result of each job
    std::vector<paddle::framework::LoDTensor*> ref_feeds;
    std::vector<paddle::framework::LoDTensor*> ref_fetches(1, &refs[i]);
    for (auto& word : jobs[i]) {
      ref_feeds.push_back(&word);
    }
    TestInference<platform::CPUPlace>(config.model_dir, ref_feeds, ref_fetches);
  }

  // create threads and each thread run 1 job
  std::vector<std::thread> threads;
  for (int tid = 0; tid < num_jobs; ++tid) {
    threads.emplace_back([&, tid]() {
      auto predictor = main_predictor->Clone();
      auto& local_inputs = paddle_tensor_feeds[tid];
      std::vector<PaddleTensor> local_outputs;
      ASSERT_TRUE(predictor->Run(local_inputs, &local_outputs));

      // check outputs range
      ASSERT_EQ(local_outputs.size(), 1UL);
      const size_t len = local_outputs[0].data.length();
      float* data = static_cast<float*>(local_outputs[0].data.data());
      for (size_t j = 0; j < len / sizeof(float); ++j) {
        ASSERT_LT(data[j], 1.0);
        ASSERT_GT(data[j], -1.0);
      }

      // check outputs correctness
      float* ref_data = refs[tid].data<float>();
      EXPECT_EQ(refs[tid].numel(), static_cast<int64_t>(len / sizeof(float)));
      for (int i = 0; i < refs[tid].numel(); ++i) {
        EXPECT_NEAR(ref_data[i], data[i], 1e-3);
      }
    });
  }
  for (int i = 0; i < num_jobs; ++i) {
    threads[i].join();
  }
}

void MainThreadsImageClassification(bool use_gpu) {
  constexpr int num_jobs = 4;  // each job run 1 batch
  constexpr int batch_size = 1;
  NativeConfig config = GetConfig();
  config.use_gpu = use_gpu;
  config.model_dir =
      FLAGS_dirname + "image_classification_resnet.inference.model";

  auto main_predictor = CreatePaddlePredictor<NativeConfig>(config);
  std::vector<framework::LoDTensor> jobs(num_jobs);
  std::vector<std::vector<PaddleTensor>> paddle_tensor_feeds(num_jobs);
  std::vector<framework::LoDTensor> refs(num_jobs);
  for (size_t i = 0; i < jobs.size(); ++i) {
    // prepare inputs
    std::vector<std::vector<int64_t>> feed_target_shapes =
        GetFeedTargetShapes(config.model_dir, /*is_combined*/ false);
    feed_target_shapes[0][0] = batch_size;
    framework::DDim input_dims = framework::make_ddim(feed_target_shapes[0]);
    SetupTensor<float>(&jobs[i], input_dims, 0.f, 1.f);
    paddle_tensor_feeds[i].push_back(LodTensorToPaddleTensor(&jobs[i]));

    // get reference result of each job
    std::vector<framework::LoDTensor*> ref_feeds(1, &jobs[i]);
    std::vector<framework::LoDTensor*> ref_fetches(1, &refs[i]);
    TestInference<platform::CPUPlace>(config.model_dir, ref_feeds, ref_fetches);
  }

  // create threads and each thread run 1 job
  std::vector<std::thread> threads;
  for (int tid = 0; tid < num_jobs; ++tid) {
    threads.emplace_back([&, tid]() {
      auto predictor = main_predictor->Clone();
      auto& local_inputs = paddle_tensor_feeds[tid];
      std::vector<PaddleTensor> local_outputs;
      ASSERT_TRUE(predictor->Run(local_inputs, &local_outputs));

      // check outputs correctness
      ASSERT_EQ(local_outputs.size(), 1UL);
      const size_t len = local_outputs[0].data.length();
      float* data = static_cast<float*>(local_outputs[0].data.data());
      float* ref_data = refs[tid].data<float>();
      EXPECT_EQ(refs[tid].numel(), len / sizeof(float));
      for (int i = 0; i < refs[tid].numel(); ++i) {
        EXPECT_NEAR(ref_data[i], data[i], 1e-3);
      }
    });
  }
  for (int i = 0; i < num_jobs; ++i) {
    threads[i].join();
  }
}

TEST(inference_api_native, word2vec_cpu) { MainWord2Vec(false /*use_gpu*/); }
TEST(inference_api_native, word2vec_cpu_threads) {
  MainThreadsWord2Vec(false /*use_gpu*/);
}
TEST(inference_api_native, image_classification_cpu) {
  MainThreadsImageClassification(false /*use_gpu*/);
}
TEST(inference_api_native, image_classification_cpu_threads) {
  MainThreadsImageClassification(false /*use_gpu*/);
}

#ifdef PADDLE_WITH_CUDA
TEST(inference_api_native, word2vec_gpu) { MainWord2Vec(true /*use_gpu*/); }
TEST(inference_api_native, word2vec_gpu_threads) {
  MainThreadsWord2Vec(true /*use_gpu*/);
}
TEST(inference_api_native, image_classification_gpu) {
  MainThreadsImageClassification(true /*use_gpu*/);
}
TEST(inference_api_native, image_classification_gpu_threads) {
  MainThreadsImageClassification(true /*use_gpu*/);
}

#endif

}  // namespace paddle
