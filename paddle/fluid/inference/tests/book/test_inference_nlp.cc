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
#include "gflags/gflags.h"
#include "gtest/gtest.h"
#include "paddle/fluid/inference/tests/test_helper.h"

DEFINE_string(dirname, "", "Directory of the inference model.");

TEST(inference, understand_sentiment) {
  if (FLAGS_dirname.empty()) {
    LOG(FATAL) << "Usage: ./example --dirname=path/to/your/model";
  }

  LOG(INFO) << "FLAGS_dirname: " << FLAGS_dirname << std::endl;
  std::string dirname = FLAGS_dirname;

  // 0. Call `paddle::framework::InitDevices()` initialize all the devices
  // In unittests, this is done in paddle/testing/paddle_gtest_main.cc
  paddle::framework::LoDTensor words;
  /*
    paddle::framework::LoD lod{{0, 83}};
    int64_t word_dict_len = 198392;
    SetupLoDTensor(&words, lod, static_cast<int64_t>(0),
                   static_cast<int64_t>(word_dict_len - 1));
   */
  std::vector<int64_t> srcdata{
      784,    784,   1550,   6463,   56,     75693, 6189,  784,    784,  1550,
      198391, 6463,  42468,  4376,   10251,  10760, 6189,  297,    396,  6463,
      6463,   1550,  198391, 6463,   22564,  1612,  291,   68,     164,  784,
      784,    1550,  198391, 6463,   13659,  3362,  42468, 6189,   2209, 198391,
      6463,   2209,  2209,   198391, 6463,   2209,  1062,  3029,   1831, 3029,
      1065,   2281,  100,    11216,  1110,   56,    10869, 9811,   100,  198391,
      6463,   100,   9280,   100,    288,    40031, 1680,  1335,   100,  1550,
      9280,   7265,  244,    1550,   198391, 6463,  1550,  198391, 6463, 42468,
      4376,   10251, 10760};
  paddle::framework::LoD lod{{0, srcdata.size()}};
  words.set_lod(lod);
  int64_t* pdata = words.mutable_data<int64_t>(
      {static_cast<int64_t>(srcdata.size()), 1}, paddle::platform::CPUPlace());
  memcpy(pdata, srcdata.data(), words.numel() * sizeof(int64_t));

  LOG(INFO) << "number of input size:" << words.numel();
  std::vector<paddle::framework::LoDTensor*> cpu_feeds;
  cpu_feeds.push_back(&words);

  paddle::framework::LoDTensor output1;
  std::vector<paddle::framework::LoDTensor*> cpu_fetchs1;
  cpu_fetchs1.push_back(&output1);

  int repeat = 100;
  // Run inference on CPU
  TestInference<paddle::platform::CPUPlace, true, true>(dirname, cpu_feeds,
                                                        cpu_fetchs1, repeat);
  LOG(INFO) << output1.lod();
  LOG(INFO) << output1.dims();

#ifdef PADDLE_WITH_CUDA
  paddle::framework::LoDTensor output2;
  std::vector<paddle::framework::LoDTensor*> cpu_fetchs2;
  cpu_fetchs2.push_back(&output2);

  // Run inference on CUDA GPU
  TestInference<paddle::platform::CUDAPlace>(dirname, cpu_feeds, cpu_fetchs2);
  LOG(INFO) << output2.lod();
  LOG(INFO) << output2.dims();

  CheckError<float>(output1, output2);
#endif
}
