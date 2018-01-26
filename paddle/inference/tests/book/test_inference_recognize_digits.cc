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
#include "paddle/framework/init.h"
#include "paddle/inference/inference.h"

DEFINE_string(dirname, "", "Directory of the inference model.");

template <typename Place, typename T>
void TestInference(const std::string& dirname,
                   const std::vector<paddle::framework::LoDTensor*>& cpu_feeds,
                   std::vector<paddle::framework::LoDTensor*>& cpu_fetchs) {
  // 1. Define place, executor and scope
  auto place = Place();
  auto executor = paddle::framework::Executor(place);
  auto* scope = new paddle::framework::Scope();

  // 2. Initialize the inference_program and load all parameters from file
  paddle::InferenceEngine* engine = new paddle::InferenceEngine();
  paddle::framework::ProgramDesc* inference_program =
      engine->LoadInferenceModel(executor, scope, dirname);

  // 3. Get the feed_var_names and fetch_var_names
  const std::vector<std::string>& feed_target_names = engine->GetFeedVarNames();
  const std::vector<std::string>& fetch_target_names =
      engine->GetFetchVarNames();

  // 4. Prepare inputs
  std::map<std::string, const paddle::framework::LoDTensor*> feed_targets;
  for (size_t i = 0; i < feed_target_names.size(); ++i) {
    // Please make sure that cpu_feeds[i] is right for feed_target_names[i]
    feed_targets[feed_target_names[i]] = cpu_feeds[i];
  }

  // 5. Define Tensor to get the outputs
  std::map<std::string, paddle::framework::LoDTensor*> fetch_targets;
  for (size_t i = 0; i < fetch_target_names.size(); ++i) {
    fetch_targets[fetch_target_names[i]] = cpu_fetchs[i];
  }

  // 6. Run the inference program
  executor.Run(*inference_program, scope, feed_targets, fetch_targets);

  delete scope;
  delete engine;
}

TEST(inference, recognize_digits) {
  if (FLAGS_dirname.empty()) {
    LOG(FATAL) << "Usage: ./example --dirname=path/to/your/model";
  }

  LOG(INFO) << "FLAGS_dirname: " << FLAGS_dirname << std::endl;

  // 0. Initialize all the devices
  paddle::framework::InitDevices();

  paddle::framework::LoDTensor input;
  srand(time(0));
  float* input_ptr =
      input.mutable_data<float>({1, 28, 28}, paddle::platform::CPUPlace());
  for (int i = 0; i < 784; ++i) {
    input_ptr[i] = rand() / (static_cast<float>(RAND_MAX));
  }
  std::vector<paddle::framework::LoDTensor*> cpu_feeds;
  cpu_feeds.push_back(&input);

  paddle::framework::LoDTensor output1;
  std::vector<paddle::framework::LoDTensor*> cpu_fetchs1;
  cpu_fetchs1.push_back(&output1);

  // Run inference on CPU
  TestInference<paddle::platform::CPUPlace, float>(
      FLAGS_dirname, cpu_feeds, cpu_fetchs1);
  LOG(INFO) << output1.dims();

#ifdef PADDLE_WITH_CUDA
  paddle::framework::LoDTensor output2;
  std::vector<paddle::framework::LoDTensor*> cpu_fetchs2;
  cpu_fetchs2.push_back(&output2);

  // Run inference on CUDA GPU
  TestInference<paddle::platform::CUDAPlace, float>(
      FLAGS_dirname, cpu_feeds, cpu_fetchs2);
  LOG(INFO) << output2.dims();

  EXPECT_EQ(output1.dims(), output2.dims());
  EXPECT_EQ(output1.numel(), output2.numel());

  float err = 1E-3;
  int count = 0;
  for (int64_t i = 0; i < output1.numel(); ++i) {
    if (fabs(output1.data<float>()[i] - output2.data<float>()[i]) > err) {
      count++;
    }
  }
  EXPECT_EQ(count, 0) << "There are " << count << " different elements.";
#endif
}

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, false);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
