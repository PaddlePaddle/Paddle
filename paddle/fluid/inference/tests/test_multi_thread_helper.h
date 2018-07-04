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

#pragma once

#include <map>
#include <string>
#include <thread>  // NOLINT
#include <vector>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/inference/io.h"

void ThreadedRunInference(
    const std::unique_ptr<paddle::framework::ProgramDesc>& inference_program,
    paddle::framework::Executor* executor, paddle::framework::Scope* scope,
    const int thread_id,
    const std::vector<paddle::framework::LoDTensor*>& cpu_feeds,
    const std::vector<paddle::framework::LoDTensor*>& cpu_fetchs) {
  auto copy_program = std::unique_ptr<paddle::framework::ProgramDesc>(
      new paddle::framework::ProgramDesc(*inference_program));

  std::string feed_holder_name = "feed_" + paddle::string::to_string(thread_id);
  std::string fetch_holder_name =
      "fetch_" + paddle::string::to_string(thread_id);
  copy_program->SetFeedHolderName(feed_holder_name);
  copy_program->SetFetchHolderName(fetch_holder_name);

  // 3. Get the feed_target_names and fetch_target_names
  const std::vector<std::string>& feed_target_names =
      copy_program->GetFeedTargetNames();
  const std::vector<std::string>& fetch_target_names =
      copy_program->GetFetchTargetNames();

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
  executor->Run(*copy_program, scope, feed_targets, fetch_targets, true,
                feed_holder_name, fetch_holder_name);
}

template <typename Place>
void TestMultiThreadInference(
    const std::string& dirname,
    const std::vector<std::vector<paddle::framework::LoDTensor*>>& cpu_feeds,
    const std::vector<std::vector<paddle::framework::LoDTensor*>>& cpu_fetchs,
    const int num_threads) {
  // 1. Define place, executor, scope
  auto place = Place();
  auto executor = paddle::framework::Executor(place);
  auto* scope = new paddle::framework::Scope();

  // 2. Initialize the inference_program and load parameters
  std::unique_ptr<paddle::framework::ProgramDesc> inference_program =
      paddle::inference::Load(executor, *scope, dirname);

  std::vector<std::thread*> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.push_back(new std::thread(
        ThreadedRunInference, std::ref(inference_program), &executor, scope, i,
        std::ref(cpu_feeds[i]), std::ref(cpu_fetchs[i])));
  }
  for (int i = 0; i < num_threads; ++i) {
    threads[i]->join();
    delete threads[i];
  }

  delete scope;
}
