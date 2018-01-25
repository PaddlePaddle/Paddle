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
#include "paddle/inference/inference.h"

DEFINE_string(dirname, "", "Directory of the inference model.");

TEST(inference, recognize_digits) {
  if (FLAGS_dirname.empty()) {
    LOG(FATAL) << "Usage: ./example --dirname=path/to/your/model";
  }

  std::cout << "FLAGS_dirname: " << FLAGS_dirname << std::endl;
  std::string dirname = FLAGS_dirname;

  paddle::InferenceEngine* engine = new paddle::InferenceEngine();
  engine->LoadInferenceModel(dirname);

  paddle::framework::LoDTensor input;
  srand(time(0));
  float* input_ptr =
      input.mutable_data<float>({1, 784}, paddle::platform::CPUPlace());
  for (int i = 0; i < 784; ++i) {
    input_ptr[i] = rand() / (static_cast<float>(RAND_MAX));
  }

  std::vector<paddle::framework::LoDTensor> feeds;
  feeds.push_back(input);
  std::vector<paddle::framework::LoDTensor> fetchs;
  engine->Execute(feeds, fetchs);

  for (size_t i = 0; i < fetchs.size(); ++i) {
    LOG(INFO) << fetchs[i].dims();
    std::stringstream ss;
    ss << "result:";
    float* output_ptr = fetchs[i].data<float>();
    for (int j = 0; j < fetchs[i].numel(); ++j) {
      ss << " " << output_ptr[j];
    }
    LOG(INFO) << ss.str();
  }

  delete engine;
}

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, false);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
