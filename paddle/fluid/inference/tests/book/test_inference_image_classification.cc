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

DEFINE_string(data_set, "cifar10", "Data set to test");
DEFINE_string(dirname, "", "Directory of the inference model.");
DEFINE_string(fp16_dirname, "", "Directory of the float16 inference model.");
DEFINE_int32(batch_size, 1, "Batch size of input data");
DEFINE_int32(repeat, 1, "Running the inference program repeat times");
DEFINE_bool(skip_cpu, false, "Skip the cpu test");

TEST(inference, image_classification) {
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
  // which should be in the range [0.0, 1.0].
  if (FLAGS_data_set == "cifar10") {
    SetupTensor<float>(&input, {FLAGS_batch_size, 3, 32, 32},
                       static_cast<float>(0), static_cast<float>(1));
  } else if (FLAGS_data_set == "imagenet") {
    SetupTensor<float>(&input, {FLAGS_batch_size, 3, 224, 224},
                       static_cast<float>(0), static_cast<float>(1));
  } else {
    LOG(FATAL) << "Only cifar10 or imagenet is supported.";
  }

  std::vector<paddle::framework::LoDTensor*> cpu_feeds;
  cpu_feeds.push_back(&input);

  paddle::framework::LoDTensor output1;
  if (!FLAGS_skip_cpu) {
    std::vector<paddle::framework::LoDTensor*> cpu_fetchs1;
    cpu_fetchs1.push_back(&output1);

    // Run inference on CPU
    LOG(INFO) << "--- CPU Runs: ---";
    LOG(INFO) << "Batch size is " << FLAGS_batch_size;
    TestInference<paddle::platform::CPUPlace, false, true>(
        dirname, cpu_feeds, cpu_fetchs1, FLAGS_repeat);
    LOG(INFO) << output1.dims();
  }

#ifdef PADDLE_WITH_CUDA
  paddle::framework::LoDTensor output2;
  std::vector<paddle::framework::LoDTensor*> cpu_fetchs2;
  cpu_fetchs2.push_back(&output2);

  // Run inference on CUDA GPU
  LOG(INFO) << "--- GPU Runs: ---";
  LOG(INFO) << "Batch size is " << FLAGS_batch_size;
  TestInference<paddle::platform::CUDAPlace, false, true>(
      dirname, cpu_feeds, cpu_fetchs2, FLAGS_repeat);
  LOG(INFO) << output2.dims();

  if (!FLAGS_skip_cpu) {
    CheckError<float>(output1, output2);
  }

  // float16 inference requires cuda GPUs with >= 5.3 compute capability
  if (paddle::platform::GetCUDAComputeCapability(0) >= 53) {
    paddle::framework::LoDTensor output3;
    std::vector<paddle::framework::LoDTensor*> cpu_fetchs3;
    cpu_fetchs3.push_back(&output3);

    LOG(INFO) << "--- GPU Runs in float16 mode: ---";
    LOG(INFO) << "Batch size is " << FLAGS_batch_size;
    if (FLAGS_fp16_dirname.empty()) {
      FLAGS_fp16_dirname = dirname;
      FLAGS_fp16_dirname.replace(FLAGS_fp16_dirname.find("book/"),
                                 std::string("book/").size(), "book/float16_");
    }

    TestInference<paddle::platform::CUDAPlace, false, true>(
        FLAGS_fp16_dirname, cpu_feeds, cpu_fetchs3, FLAGS_repeat);

    CheckError<float>(output2, output3);
  }
#endif
}
