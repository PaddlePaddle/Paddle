/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

TEST(recognize_digits, CPU) {
  if (FLAGS_dirname.empty()) {
    LOG(FATAL) << "Usage: ./example --dirname=path/to/your/model";
  }

  std::cout << "FLAGS_dirname: " << FLAGS_dirname << std::endl;
  std::string dirname = FLAGS_dirname;

  // 0. Initialize all the devices
  paddle::framework::InitDevices();

  // 1. Define place, executor and scope
  auto place = paddle::platform::CPUPlace();
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
  paddle::framework::LoDTensor input;
  srand(time(0));
  float* input_ptr =
      input.mutable_data<float>({1, 28, 28}, paddle::platform::CPUPlace());
  for (int i = 0; i < 784; ++i) {
    input_ptr[i] = rand() / (static_cast<float>(RAND_MAX));
  }
  feed_targets[feed_target_names[0]] = &input;

  // 5. Define Tensor to get the outputs
  std::map<std::string, paddle::framework::LoDTensor*> fetch_targets;
  paddle::framework::LoDTensor output;
  fetch_targets[fetch_target_names[0]] = &output;

  // 6. Run the inference program
  executor.Run(*inference_program, scope, feed_targets, fetch_targets);

  // 7. Use the output as your expect.
  LOG(INFO) << output.dims();
  std::stringstream ss;
  ss << "result:";
  float* output_ptr = output.data<float>();
  for (int j = 0; j < output.numel(); ++j) {
    ss << " " << output_ptr[j];
  }
  LOG(INFO) << ss.str();

  delete scope;
  delete engine;
}

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, false);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
