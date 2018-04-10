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
DEFINE_int32(batch_size, 1, "Batch size of input data");
DEFINE_int32(repeat, 1, "Running the inference program repeat times");

TEST(inference, recognize_digits) {
  if (FLAGS_dirname.empty() || FLAGS_batch_size < 1 || FLAGS_repeat < 1) {
    LOG(FATAL) << "Usage: ./example --dirname=path/to/your/model "
                  "--batch_size=1 --repeat=1";
  }

  LOG(INFO) << "FLAGS_dirname: " << FLAGS_dirname << std::endl;
  std::string dirname = FLAGS_dirname;

  // 0. Call `paddle::framework::InitDevices()` initialize all the devices
  // In unittests, this is done in paddle/testing/paddle_gtest_main.cc

  paddle::framework::LoDTensor input;
  // Use normilized image pixels as input data,
  // which should be in the range [-1.0, 1.0].
  SetupTensor<float>(&input, {FLAGS_batch_size, 1, 28, 28},
                     static_cast<float>(-1), static_cast<float>(1));
  std::vector<paddle::framework::LoDTensor*> cpu_feeds;
  cpu_feeds.push_back(&input);

  for (auto is_combined : {false, true}) {
    paddle::framework::LoDTensor output1;
    std::vector<paddle::framework::LoDTensor*> cpu_fetchs1;
    cpu_fetchs1.push_back(&output1);

    // Run inference on CPU
    LOG(INFO) << "--- CPU Runs: is_combined=" << is_combined << " ---";
    TestInference<paddle::platform::CPUPlace>(dirname, cpu_feeds, cpu_fetchs1,
                                              FLAGS_repeat, is_combined);
    LOG(INFO) << output1.dims();

#ifdef PADDLE_WITH_CUDA
    paddle::framework::LoDTensor output2;
    std::vector<paddle::framework::LoDTensor*> cpu_fetchs2;
    cpu_fetchs2.push_back(&output2);

    // Run inference on CUDA GPU
    LOG(INFO) << "--- GPU Runs: is_combined=" << is_combined << " ---";
    TestInference<paddle::platform::CUDAPlace>(dirname, cpu_feeds, cpu_fetchs2,
                                               FLAGS_repeat, is_combined);
    LOG(INFO) << output2.dims();

    CheckError<float>(output1, output2);
#endif
  }
}
