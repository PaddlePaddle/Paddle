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

#include <sys/time.h>
#include <time.h>
#include <fstream>
#include <thread>  // NOLINT
#include "gflags/gflags.h"
#include "gtest/gtest.h"
#include "paddle/fluid/inference/tests/test_helper.h"
#ifdef PADDLE_WITH_MKLML
#include <mkl_service.h>
#include <omp.h>
#endif

DEFINE_string(model_path, "", "Directory of the inference model.");
DEFINE_string(data_file, "", "File of input index data.");
DEFINE_int32(repeat, 100, "Running the inference program repeat times");
DEFINE_bool(prepare_vars, true, "Prepare variables before executor");
DEFINE_int32(num_threads, 1, "Number of threads should be used");
DECLARE_bool(use_mkldnn);

inline double GetCurrentMs() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+3 * time.tv_sec + 1e-3 * time.tv_usec;
}

// This function just give dummy data for recognize_digits model.
size_t DummyData(std::vector<paddle::framework::LoDTensor>* out) {
  paddle::framework::LoDTensor input;
  SetupTensor<float>(&input, {1, 1, 28, 28}, -1.f, 1.f);
  out->emplace_back(input);
  return 1;
}

// Load the input word index data from file and save into LodTensor.
// Return the size of words.
size_t LoadData(std::vector<paddle::framework::LoDTensor>* out,
                const std::string& filename) {
  if (filename.empty()) {
    return DummyData(out);
  }

  size_t sz = 0;
  std::fstream fin(filename);
  std::string line;
  out->clear();
  while (getline(fin, line)) {
    std::istringstream iss(line);
    std::vector<int64_t> ids;
    std::string field;
    while (getline(iss, field, ' ')) {
      ids.push_back(stoi(field));
    }
    if (ids.size() >= 1024) {
      // Synced with NLP guys, they will ignore input larger then 1024
      continue;
    }

    paddle::framework::LoDTensor words;
    paddle::framework::LoD lod{{0, ids.size()}};
    words.set_lod(lod);
    int64_t* pdata = words.mutable_data<int64_t>(
        {static_cast<int64_t>(ids.size()), 1}, paddle::platform::CPUPlace());
    memcpy(pdata, ids.data(), words.numel() * sizeof(int64_t));
    out->emplace_back(words);
    sz += ids.size();
  }
  return sz;
}

// Split input data samples into small pieces jobs as balanced as possible,
// according to the number of threads.
void SplitData(
    const std::vector<paddle::framework::LoDTensor>& datasets,
    std::vector<std::vector<const paddle::framework::LoDTensor*>>* jobs,
    const int num_threads) {
  size_t s = 0;
  jobs->resize(num_threads);
  while (s < datasets.size()) {
    for (auto it = jobs->begin(); it != jobs->end(); it++) {
      it->emplace_back(&datasets[s]);
      s++;
      if (s >= datasets.size()) {
        break;
      }
    }
  }
}

void ThreadRunInfer(
    const int tid, paddle::framework::Scope* scope,
    const std::vector<std::vector<const paddle::framework::LoDTensor*>>& jobs) {
  // maybe framework:ProgramDesc is not thread-safe
  paddle::platform::CPUPlace place;
  paddle::framework::Executor executor(place);
  auto& sub_scope = scope->NewScope();
  auto inference_program =
      paddle::inference::Load(&executor, scope, FLAGS_model_path);

  auto ctx = executor.Prepare(*inference_program, /*block_id*/ 0);
  executor.CreateVariables(*inference_program, &sub_scope, /*block_id*/ 0);

  const std::vector<std::string>& feed_target_names =
      inference_program->GetFeedTargetNames();
  const std::vector<std::string>& fetch_target_names =
      inference_program->GetFetchTargetNames();

  PADDLE_ENFORCE_EQ(fetch_target_names.size(), 1UL);
  std::map<std::string, paddle::framework::LoDTensor*> fetch_targets;
  paddle::framework::LoDTensor outtensor;
  fetch_targets[fetch_target_names[0]] = &outtensor;

  std::map<std::string, const paddle::framework::LoDTensor*> feed_targets;
  PADDLE_ENFORCE_EQ(feed_target_names.size(), 1UL);

  auto& inputs = jobs[tid];
  auto start_ms = GetCurrentMs();
  for (size_t i = 0; i < inputs.size(); ++i) {
    feed_targets[feed_target_names[0]] = inputs[i];
    executor.RunPreparedContext(ctx.get(), &sub_scope, &feed_targets,
                                &fetch_targets, false /*create_local_scope*/);
  }
  auto stop_ms = GetCurrentMs();
  scope->DeleteScope(&sub_scope);
  LOG(INFO) << "Tid: " << tid << ", process " << inputs.size()
            << " samples, avg time per sample: "
            << (stop_ms - start_ms) / inputs.size() << " ms";
}

TEST(inference, nlp) {
  if (FLAGS_model_path.empty()) {
    LOG(FATAL) << "Usage: ./example --model_path=path/to/your/model";
  }
  if (FLAGS_data_file.empty()) {
    LOG(WARNING) << "No data file provided, will use dummy data!"
                 << "Note: if you use nlp model, please provide data file.";
  }
  LOG(INFO) << "Model Path: " << FLAGS_model_path;
  LOG(INFO) << "Data File: " << FLAGS_data_file;

  std::vector<paddle::framework::LoDTensor> datasets;
  size_t num_total_words = LoadData(&datasets, FLAGS_data_file);
  LOG(INFO) << "Number of samples (seq_len<1024): " << datasets.size();
  LOG(INFO) << "Total number of words: " << num_total_words;

  // 0. Call `paddle::framework::InitDevices()` initialize all the devices
  std::unique_ptr<paddle::framework::Scope> scope(
      new paddle::framework::Scope());

#ifdef PADDLE_WITH_MKLML
  // only use 1 thread number per std::thread
  omp_set_dynamic(0);
  omp_set_num_threads(1);
  mkl_set_num_threads(1);
#endif

  double start_ms = 0, stop_ms = 0;
  if (FLAGS_num_threads > 1) {
    std::vector<std::vector<const paddle::framework::LoDTensor*>> jobs;
    SplitData(datasets, &jobs, FLAGS_num_threads);
    std::vector<std::unique_ptr<std::thread>> threads;
    start_ms = GetCurrentMs();
    for (int i = 0; i < FLAGS_num_threads; ++i) {
      threads.emplace_back(
          new std::thread(ThreadRunInfer, i, scope.get(), std::ref(jobs)));
    }
    for (int i = 0; i < FLAGS_num_threads; ++i) {
      threads[i]->join();
    }
    stop_ms = GetCurrentMs();
  } else {
    // 1. Define place, executor, scope
    paddle::platform::CPUPlace place;
    paddle::framework::Executor executor(place);

    // 2. Initialize the inference_program and load parameters
    std::unique_ptr<paddle::framework::ProgramDesc> inference_program;
    inference_program = InitProgram(&executor, scope.get(), FLAGS_model_path,
                                    /*model combined*/ false);
    // always prepare context
    std::unique_ptr<paddle::framework::ExecutorPrepareContext> ctx;
    ctx = executor.Prepare(*inference_program, 0);
    if (FLAGS_prepare_vars) {
      executor.CreateVariables(*inference_program, scope.get(), 0);
    }
    // preapre fetch
    const std::vector<std::string>& fetch_target_names =
        inference_program->GetFetchTargetNames();
    PADDLE_ENFORCE_EQ(fetch_target_names.size(), 1UL);
    std::map<std::string, paddle::framework::LoDTensor*> fetch_targets;
    paddle::framework::LoDTensor outtensor;
    fetch_targets[fetch_target_names[0]] = &outtensor;

    // prepare feed
    const std::vector<std::string>& feed_target_names =
        inference_program->GetFeedTargetNames();
    PADDLE_ENFORCE_EQ(feed_target_names.size(), 1UL);
    std::map<std::string, const paddle::framework::LoDTensor*> feed_targets;

    // feed data and run
    start_ms = GetCurrentMs();
    for (size_t i = 0; i < datasets.size(); ++i) {
      feed_targets[feed_target_names[0]] = &(datasets[i]);
      executor.RunPreparedContext(ctx.get(), scope.get(), &feed_targets,
                                  &fetch_targets, !FLAGS_prepare_vars);
    }
    stop_ms = GetCurrentMs();
    LOG(INFO) << "Tid: 0, process " << datasets.size()
              << " samples, avg time per sample: "
              << (stop_ms - start_ms) / datasets.size() << " ms";
  }
  LOG(INFO) << "Total inference time with " << FLAGS_num_threads
            << " threads : " << (stop_ms - start_ms) / 1000.0
            << " sec, QPS: " << datasets.size() / ((stop_ms - start_ms) / 1000);
}
