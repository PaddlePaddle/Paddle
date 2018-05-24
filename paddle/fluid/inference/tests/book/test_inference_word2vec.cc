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

#include <sys/time.h>
#include <time.h>
#include <thread>

#include "gflags/gflags.h"
#include "gtest/gtest.h"
#include "paddle/fluid/inference/tests/test_helper.h"

DEFINE_string(dirname, "", "Directory of the inference model.");

inline uint64_t PosixInNsec() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return 1000 * (static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec);
}

TEST(inference, word2vec) {
  if (FLAGS_dirname.empty()) {
    LOG(FATAL) << "Usage: ./example --dirname=path/to/your/model";
  }

  LOG(INFO) << "FLAGS_dirname: " << FLAGS_dirname << std::endl;
  std::string dirname = FLAGS_dirname;

  // 0. Call `paddle::framework::InitDevices()` initialize all the devices
  // In unittests, this is done in paddle/testing/paddle_gtest_main.cc

  paddle::framework::LoDTensor first_word, second_word, third_word, fourth_word;
  paddle::framework::LoD lod{{0, 1}};
  int64_t dict_size = 2073;  // The size of dictionary

  SetupLoDTensor(&first_word, lod, static_cast<int64_t>(0), dict_size - 1);
  SetupLoDTensor(&second_word, lod, static_cast<int64_t>(0), dict_size - 1);
  SetupLoDTensor(&third_word, lod, static_cast<int64_t>(0), dict_size - 1);
  SetupLoDTensor(&fourth_word, lod, static_cast<int64_t>(0), dict_size - 1);

  std::vector<paddle::framework::LoDTensor*> cpu_feeds;
  cpu_feeds.push_back(&first_word);
  cpu_feeds.push_back(&second_word);
  cpu_feeds.push_back(&third_word);
  cpu_feeds.push_back(&fourth_word);

  int total_work = 1000;
  int num_threads = 10;
  int work_per_thread = total_work / num_threads;
  // Run inference on CPU
  std::vector<std::unique_ptr<std::thread>> infer_threads;
  for (int i = 0; i < num_threads; ++i) {
    infer_threads.emplace_back(new std::thread([&, i]() {
      for (int j = 0; j < work_per_thread; ++j) {
        /*
        paddle::framework::LoDTensor output1;
        std::vector<paddle::framework::LoDTensor*> cpu_fetchs1;
        cpu_fetchs1.push_back(&output1);

        TestInference<paddle::platform::CPUPlace>(
            dirname, cpu_feeds, cpu_fetchs1);*/
#ifdef PADDLE_WITH_CUDA
        paddle::framework::LoDTensor output2;
        std::vector<paddle::framework::LoDTensor*> cpu_fetchs2;
        cpu_fetchs2.push_back(&output2);

        // Run inference on CUDA GPU
        TestInference<paddle::platform::CUDAPlace>(
            dirname, cpu_feeds, cpu_fetchs2);

        // CheckError<float>(output1, output2);
#endif
        fprintf(stderr, "working on %dth in thread %d\n", j, i);
      }
    }));
  }
  uint64_t start_ns = PosixInNsec();
  for (int i = 0; i < num_threads; ++i) {
    infer_threads[i]->join();
  }
  uint64_t stop_ns = PosixInNsec();
  fprintf(stderr, "total time: %lu\n", stop_ns - start_ns);
}
