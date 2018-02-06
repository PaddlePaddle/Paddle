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

#include "paddle/framework/lod_tensor.h"
#include "paddle/inference/io.h"

template <typename T>
void SetupTensor(paddle::framework::LoDTensor& input,
                 paddle::framework::DDim dims,
                 T lower,
                 T upper) {
  srand(time(0));
  float* input_ptr = input.mutable_data<T>(dims, paddle::platform::CPUPlace());
  for (int i = 0; i < input.numel(); ++i) {
    input_ptr[i] =
        (static_cast<T>(rand()) / static_cast<T>(RAND_MAX)) * (upper - lower) +
        lower;
  }
}

template <typename T>
void CheckError(paddle::framework::LoDTensor& output1,
                paddle::framework::LoDTensor& output2) {
  // Check lod information
  EXPECT_EQ(output1.lod(), output2.lod());

  EXPECT_EQ(output1.dims(), output2.dims());
  EXPECT_EQ(output1.numel(), output2.numel());

  T err = static_cast<T>(0);
  if (typeid(T) == typeid(float)) {
    err = 1E-3;
  } else if (typeid(T) == typeid(double)) {
    err = 1E-6;
  } else {
    err = 0;
  }

  size_t count = 0;
  for (int64_t i = 0; i < output1.numel(); ++i) {
    if (fabs(output1.data<T>()[i] - output2.data<T>()[i]) > err) {
      count++;
    }
  }
  EXPECT_EQ(count, 0) << "There are " << count << " different elements.";
}

template <typename Place, typename T>
void TestInference(const std::string& dirname,
                   const std::vector<paddle::framework::LoDTensor*>& cpu_feeds,
                   std::vector<paddle::framework::LoDTensor*>& cpu_fetchs) {
  // 1. Define place, executor and scope
  auto place = Place();
  auto executor = paddle::framework::Executor(place);
  auto* scope = new paddle::framework::Scope();

  // 2. Initialize the inference_program and load all parameters from file
  auto inference_program = paddle::inference::Load(executor, *scope, dirname);

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
