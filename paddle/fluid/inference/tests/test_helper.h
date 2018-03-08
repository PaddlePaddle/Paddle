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

#include <time.h>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/inference/io.h"
#include "paddle/fluid/platform/profiler.h"

template <typename T>
void SetupTensor(paddle::framework::LoDTensor& input,
                 paddle::framework::DDim dims,
                 T lower,
                 T upper) {
  srand(time(0));
  T* input_ptr = input.mutable_data<T>(dims, paddle::platform::CPUPlace());
  for (int i = 0; i < input.numel(); ++i) {
    input_ptr[i] =
        (static_cast<T>(rand()) / static_cast<T>(RAND_MAX)) * (upper - lower) +
        lower;
  }
}

template <typename T>
void SetupTensor(paddle::framework::LoDTensor& input,
                 paddle::framework::DDim dims,
                 std::vector<T>& data) {
  CHECK_EQ(paddle::framework::product(dims), static_cast<int64_t>(data.size()));
  T* input_ptr = input.mutable_data<T>(dims, paddle::platform::CPUPlace());
  memcpy(input_ptr, data.data(), input.numel() * sizeof(T));
}

template <typename T>
void SetupLoDTensor(paddle::framework::LoDTensor& input,
                    paddle::framework::LoD& lod,
                    T lower,
                    T upper) {
  input.set_lod(lod);
  int dim = lod[0][lod[0].size() - 1];
  SetupTensor<T>(input, {dim, 1}, lower, upper);
}

template <typename T>
void SetupLoDTensor(paddle::framework::LoDTensor& input,
                    paddle::framework::DDim dims,
                    paddle::framework::LoD lod,
                    std::vector<T>& data) {
  const size_t level = lod.size() - 1;
  CHECK_EQ(dims[0], static_cast<int64_t>((lod[level]).back()));
  input.set_lod(lod);
  SetupTensor<T>(input, dims, data);
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
  EXPECT_EQ(count, 0U) << "There are " << count << " different elements.";
}

template <typename Place>
void TestInference(const std::string& dirname,
                   const std::vector<paddle::framework::LoDTensor*>& cpu_feeds,
                   std::vector<paddle::framework::LoDTensor*>& cpu_fetchs,
                   const int repeat = 1,
                   const bool is_combined = false) {
  // 1. Define place, executor, scope
  auto place = Place();
  auto executor = paddle::framework::Executor(place);
  auto* scope = new paddle::framework::Scope();

  // Profile the performance
  paddle::platform::ProfilerState state;
  if (paddle::platform::is_cpu_place(place)) {
    state = paddle::platform::ProfilerState::kCPU;
  } else {
#ifdef PADDLE_WITH_CUDA
    state = paddle::platform::ProfilerState::kCUDA;
    // The default device_id of paddle::platform::CUDAPlace is 0.
    // Users can get the device_id using:
    //   int device_id = place.GetDeviceId();
    paddle::platform::SetDeviceId(0);
#else
    PADDLE_THROW("'CUDAPlace' is not supported in CPU only device.");
#endif
  }

  // Enable the profiler
  paddle::platform::EnableProfiler(state);

  // 2. Initialize the inference_program and load parameters
  std::unique_ptr<paddle::framework::ProgramDesc> inference_program;
  {
    paddle::platform::RecordEvent record_event(
        "init_program",
        paddle::platform::DeviceContextPool::Instance().Get(place));

    if (is_combined) {
      // All parameters are saved in a single file.
      // Hard-coding the file names of program and parameters in unittest.
      // The file names should be consistent with that used in Python API
      //  `fluid.io.save_inference_model`.
      std::string prog_filename = "__model_combined__";
      std::string param_filename = "__params_combined__";
      inference_program =
          paddle::inference::Load(executor,
                                  *scope,
                                  dirname + "/" + prog_filename,
                                  dirname + "/" + param_filename);
    } else {
      // Parameters are saved in separate files sited in the specified
      // `dirname`.
      inference_program = paddle::inference::Load(executor, *scope, dirname);
    }
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
  {
    // Run repeat times to profile the performance
    for (int i = 0; i < repeat; ++i) {
      paddle::platform::RecordEvent record_event(
          "run_inference",
          paddle::platform::DeviceContextPool::Instance().Get(place));

      executor.Run(*inference_program, scope, feed_targets, fetch_targets);
    }
  }

  // Disable the profiler and print the timing information
  paddle::platform::DisableProfiler(paddle::platform::EventSortingKey::kDefault,
                                    "profiler.txt");
  paddle::platform::ResetProfiler();

  delete scope;
}
