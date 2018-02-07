/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>
#include <time.h>
#include <sstream>
#include "gflags/gflags.h"
#include "test_helper.h"

DEFINE_string(dirname, "", "Directory of the inference model.");

template <typename Place, typename T, bool IsCombined>
void TestInferenceWithCombine(
    const std::string& dirname,
    const std::vector<paddle::framework::LoDTensor*>& cpu_feeds,
    std::vector<paddle::framework::LoDTensor*>& cpu_fetchs) {
  // 1. Define place, executor and scope
  auto place = Place();
  auto executor = paddle::framework::Executor(place);
  auto* scope = new paddle::framework::Scope();
  std::unique_ptr<paddle::framework::ProgramDesc> inference_program;

  LOG(INFO) << "Going to call load()" << std::endl;
  // 2. Initialize the inference_program and load all parameters from file
  if (IsCombined) {
    // Hard-coding the names for combined params case
    std::string prog_filename = "__model_combined__";
    std::string param_filename = "__params_combined__";
    inference_program = paddle::inference::Load(executor,
                                                *scope,
                                                dirname + "/" + prog_filename,
                                                dirname + "/" + param_filename);
  } else {
    inference_program = paddle::inference::Load(executor, *scope, dirname);
  }

  // 3. Get the feed_target_names and fetch_target_names
  const std::vector<std::string>& feed_target_names =
      inference_program->GetFeedTargetNames();
  const std::vector<std::string>& fetch_target_names =
      inference_program->GetFetchTargetNames();

  // 4. Prepare inputs: set up maps for feed targets
  std::map<std::string, const paddle::framework::LoDTensor*> feed_targets;
  for (size_t i = 0; i < feed_target_names.size(); ++i) {
    // Please make sure that cpu_feeds[i] is right for feed_target_names[i]
    feed_targets[feed_target_names[i]] = cpu_feeds[i];
  }

  // 5. Define Tensor to get the outputs: set up maps for fetch targets
  std::map<std::string, paddle::framework::LoDTensor*> fetch_targets;
  for (size_t i = 0; i < fetch_target_names.size(); ++i) {
    fetch_targets[fetch_target_names[i]] = cpu_fetchs[i];
  }

  // 6. Run the inference program
  executor.Run(*inference_program, scope, feed_targets, fetch_targets);

  delete scope;
}

TEST(inference, recognize_digits) {
  if (FLAGS_dirname.empty()) {
    LOG(FATAL) << "Usage: ./example --dirname=path/to/your/model";
  }

  LOG(INFO) << "FLAGS_dirname: " << FLAGS_dirname << std::endl;
  std::string dirname = FLAGS_dirname;

  // 0. Call `paddle::framework::InitDevices()` initialize all the devices
  // In unittests, this is done in paddle/testing/paddle_gtest_main.cc

  paddle::framework::LoDTensor input;
  // Use normilized image pixels as input data,
  // which should be in the range [-1.0, 1.0].
  SetupTensor<float>(
      input, {1, 28, 28}, static_cast<float>(-1), static_cast<float>(1));
  std::vector<paddle::framework::LoDTensor*> cpu_feeds;
  cpu_feeds.push_back(&input);

  paddle::framework::LoDTensor output1;
  std::vector<paddle::framework::LoDTensor*> cpu_fetchs1;
  cpu_fetchs1.push_back(&output1);

  // Run inference on CPU
  TestInferenceWithCombine<paddle::platform::CPUPlace, float, false>(
      dirname, cpu_feeds, cpu_fetchs1);
  LOG(INFO) << output1.dims();

#ifdef PADDLE_WITH_CUDA
  paddle::framework::LoDTensor output2;
  std::vector<paddle::framework::LoDTensor*> cpu_fetchs2;
  cpu_fetchs2.push_back(&output2);

  // Run inference on CUDA GPU
  TestInferenceWithCombine<paddle::platform::CUDAPlace, float, false>(
      dirname, cpu_feeds, cpu_fetchs2);
  LOG(INFO) << output2.dims();

  CheckError<float>(output1, output2);
#endif
}

TEST(inference, recognize_digits_combine) {
  if (FLAGS_dirname.empty()) {
    LOG(FATAL) << "Usage: ./example --dirname=path/to/your/model";
  }

  LOG(INFO) << "FLAGS_dirname: " << FLAGS_dirname << std::endl;
  std::string dirname = FLAGS_dirname;

  // 0. Call `paddle::framework::InitDevices()` initialize all the devices
  // In unittests, this is done in paddle/testing/paddle_gtest_main.cc

  paddle::framework::LoDTensor input;
  // Use normilized image pixels as input data,
  // which should be in the range [-1.0, 1.0].
  SetupTensor<float>(
      input, {1, 28, 28}, static_cast<float>(-1), static_cast<float>(1));
  std::vector<paddle::framework::LoDTensor*> cpu_feeds;
  cpu_feeds.push_back(&input);

  paddle::framework::LoDTensor output1;
  std::vector<paddle::framework::LoDTensor*> cpu_fetchs1;
  cpu_fetchs1.push_back(&output1);

  // Run inference on CPU
  TestInferenceWithCombine<paddle::platform::CPUPlace, float, true>(
      dirname, cpu_feeds, cpu_fetchs1);
  LOG(INFO) << output1.dims();

#ifdef PADDLE_WITH_CUDA
  paddle::framework::LoDTensor output2;
  std::vector<paddle::framework::LoDTensor*> cpu_fetchs2;
  cpu_fetchs2.push_back(&output2);

  // Run inference on CUDA GPU
  TestInferenceWithCombine<paddle::platform::CUDAPlace, float, true>(
      dirname, cpu_feeds, cpu_fetchs2);
  LOG(INFO) << output2.dims();

  CheckError<float>(output1, output2);
#endif
}
