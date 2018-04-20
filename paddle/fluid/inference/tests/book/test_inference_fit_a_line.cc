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
#include "paddle/fluid/inference/tests/test_multi_thread_helper.h"

DEFINE_string(dirname, "", "Directory of the inference model.");

TEST(inference, fit_a_line) {
  if (FLAGS_dirname.empty()) {
    LOG(FATAL) << "Usage: ./example --dirname=path/to/your/model";
  }

  LOG(INFO) << "FLAGS_dirname: " << FLAGS_dirname << std::endl;
  std::string dirname = FLAGS_dirname;

  // 0. Call `paddle::framework::InitDevices()` initialize all the devices
  // In unittests, this is done in paddle/testing/paddle_gtest_main.cc

  for (int num_threads : {1, 2}) {
    std::vector<std::vector<paddle::framework::LoDTensor*>> cpu_feeds;
    cpu_feeds.resize(num_threads);
    for (int i = 0; i < num_threads; ++i) {
      auto* input = new paddle::framework::LoDTensor();
      // The second dim of the input tensor should be 13
      // The input data should be >= 0
      int64_t batch_size = 10;
      SetupTensor<float>(input, {batch_size, 13}, static_cast<float>(0),
                         static_cast<float>(10));
      cpu_feeds[i].push_back(input);
    }

    std::vector<std::vector<paddle::framework::LoDTensor*>> cpu_fetchs1;
    cpu_fetchs1.resize(num_threads);
    for (int i = 0; i < num_threads; ++i) {
      auto* output = new paddle::framework::LoDTensor();
      cpu_fetchs1[i].push_back(output);
    }

    // Run inference on CPU
    LOG(INFO) << "--- CPU Runs (num_threads: " << num_threads << "): ---";
    if (num_threads == 1) {
      TestInference<paddle::platform::CPUPlace>(dirname, cpu_feeds[0],
                                                cpu_fetchs1[0]);
    } else {
      TestMultiThreadInference<paddle::platform::CPUPlace>(
          dirname, cpu_feeds, cpu_fetchs1, num_threads);
    }

#ifdef PADDLE_WITH_CUDA
    std::vector<std::vector<paddle::framework::LoDTensor*>> cpu_fetchs2;
    cpu_fetchs2.resize(num_threads);
    for (int i = 0; i < num_threads; ++i) {
      auto* output = new paddle::framework::LoDTensor();
      cpu_fetchs2[i].push_back(output);
    }

    // Run inference on CUDA GPU
    LOG(INFO) << "--- GPU Runs (num_threads: " << num_threads << "): ---";
    if (num_threads == 1) {
      TestInference<paddle::platform::CUDAPlace>(dirname, cpu_feeds[0],
                                                 cpu_fetchs2[0]);
    } else {
      TestMultiThreadInference<paddle::platform::CUDAPlace>(
          dirname, cpu_feeds, cpu_fetchs2, num_threads);
    }

    for (int i = 0; i < num_threads; ++i) {
      CheckError<float>(*cpu_fetchs1[i][0], *cpu_fetchs2[i][0]);
      delete cpu_fetchs2[i][0];
    }
#endif

    for (int i = 0; i < num_threads; ++i) {
      delete cpu_feeds[i][0];
      delete cpu_fetchs1[i][0];
    }
  }  // num_threads-loop
}
