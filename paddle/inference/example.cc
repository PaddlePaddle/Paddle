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

#include <time.h>
#include <iostream>
#include "gflags/gflags.h"
#include "paddle/framework/init.h"
#include "paddle/framework/lod_tensor.h"
#include "paddle/inference/io.h"

DEFINE_string(dirname, "", "Directory of the inference model.");

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_dirname.empty()) {
    // Example:
    //   ./example --dirname=recognize_digits_mlp.inference.model
    std::cout << "Usage: ./example --dirname=path/to/your/model" << std::endl;
    exit(1);
  }

  // 1. Define place, executor, scope
  auto place = paddle::platform::CPUPlace();
  paddle::framework::InitDevices();
  auto* executor = new paddle::framework::Executor(place);
  auto* scope = new paddle::framework::Scope();

  std::cout << "FLAGS_dirname: " << FLAGS_dirname << std::endl;
  std::string dirname = FLAGS_dirname;

  // 2. Initialize the inference program
  auto inference_program = paddle::inference::Load(*executor, *scope, dirname);

  // 3. Optional: perform optimization on the inference_program

  // 4. Get the feed_target_names and fetch_target_names
  const std::vector<std::string>& feed_target_names =
      inference_program->GetFeedTargetNames();
  const std::vector<std::string>& fetch_target_names =
      inference_program->GetFetchTargetNames();

  // 5. Generate input
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

  // Set up maps for feed and fetch targets
  std::map<std::string, const paddle::framework::LoDTensor*> feed_targets;
  std::map<std::string, paddle::framework::LoDTensor*> fetch_targets;

  // set_feed_variable
  for (size_t i = 0; i < feed_target_names.size(); ++i) {
    feed_targets[feed_target_names[i]] = &feeds[i];
  }

  // get_fetch_variable
  fetchs.resize(fetch_target_names.size());
  for (size_t i = 0; i < fetch_target_names.size(); ++i) {
    fetch_targets[fetch_target_names[i]] = &fetchs[i];
  }

  // Run the inference program
  executor->Run(*inference_program, scope, feed_targets, fetch_targets);

  // Get outputs
  for (size_t i = 0; i < fetchs.size(); ++i) {
    auto dims_i = fetchs[i].dims();
    std::cout << "dims_i:";
    for (int j = 0; j < dims_i.size(); ++j) {
      std::cout << " " << dims_i[j];
    }
    std::cout << std::endl;
    std::cout << "result:";
    float* output_ptr = fetchs[i].data<float>();
    for (int j = 0; j < paddle::framework::product(dims_i); ++j) {
      std::cout << " " << output_ptr[j];
    }
    std::cout << std::endl;
  }

  delete scope;
  delete executor;

  return 0;
}
