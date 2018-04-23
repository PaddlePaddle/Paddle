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

#include "gflags/gflags.h"
#include "gtest/gtest.h"
#include "paddle/fluid/inference/tests/test_helper.h"

DEFINE_string(dirname, "", "Directory of the inference model.");

TEST(inference, recommender_system) {
  if (FLAGS_dirname.empty()) {
    LOG(FATAL) << "Usage: ./example --dirname=path/to/your/model";
  }

  LOG(INFO) << "FLAGS_dirname: " << FLAGS_dirname << std::endl;
  std::string dirname = FLAGS_dirname;

  // 0. Call `paddle::framework::InitDevices()` initialize all the devices
  // In unittests, this is done in paddle/testing/paddle_gtest_main.cc

  int64_t batch_size = 1;

  paddle::framework::LoDTensor user_id, gender_id, age_id, job_id, movie_id,
      category_id, movie_title;

  // Use the first data from paddle.dataset.movielens.test() as input
  std::vector<int64_t> user_id_data = {1};
  SetupTensor<int64_t>(&user_id, {batch_size, 1}, user_id_data);

  std::vector<int64_t> gender_id_data = {1};
  SetupTensor<int64_t>(&gender_id, {batch_size, 1}, gender_id_data);

  std::vector<int64_t> age_id_data = {0};
  SetupTensor<int64_t>(&age_id, {batch_size, 1}, age_id_data);

  std::vector<int64_t> job_id_data = {10};
  SetupTensor<int64_t>(&job_id, {batch_size, 1}, job_id_data);

  std::vector<int64_t> movie_id_data = {783};
  SetupTensor<int64_t>(&movie_id, {batch_size, 1}, movie_id_data);

  std::vector<int64_t> category_id_data = {10, 8, 9};
  SetupLoDTensor<int64_t>(&category_id, {3, 1}, {{0, 3}}, category_id_data);

  std::vector<int64_t> movie_title_data = {1069, 4140, 2923, 710, 988};
  SetupLoDTensor<int64_t>(&movie_title, {5, 1}, {{0, 5}}, movie_title_data);

  std::vector<paddle::framework::LoDTensor*> cpu_feeds;
  cpu_feeds.push_back(&user_id);
  cpu_feeds.push_back(&gender_id);
  cpu_feeds.push_back(&age_id);
  cpu_feeds.push_back(&job_id);
  cpu_feeds.push_back(&movie_id);
  cpu_feeds.push_back(&category_id);
  cpu_feeds.push_back(&movie_title);

  paddle::framework::LoDTensor output1;
  std::vector<paddle::framework::LoDTensor*> cpu_fetchs1;
  cpu_fetchs1.push_back(&output1);

  // Run inference on CPU
  TestInference<paddle::platform::CPUPlace>(dirname, cpu_feeds, cpu_fetchs1);
  LOG(INFO) << output1.dims();

#ifdef PADDLE_WITH_CUDA
  paddle::framework::LoDTensor output2;
  std::vector<paddle::framework::LoDTensor*> cpu_fetchs2;
  cpu_fetchs2.push_back(&output2);

  // Run inference on CUDA GPU
  TestInference<paddle::platform::CUDAPlace>(dirname, cpu_feeds, cpu_fetchs2);
  LOG(INFO) << output2.dims();

  CheckError<float>(output1, output2);
#endif
}
