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
#include <iostream>
#include <sstream>
#include <string>
#include <thread>  // NOLINT
#include <vector>
#include "gflags/gflags.h"
#include "gtest/gtest.h"
#include "paddle/fluid/inference/tests/test_helper.h"

DEFINE_string(dirname, "", "Directory of the inference model.");
DEFINE_int32(repeat, 100, "Running the inference program repeat times");
DEFINE_bool(use_mkldnn, false, "Use MKLDNN to run inference");
DEFINE_bool(prepare_vars, true, "Prepare variables before executor");
DEFINE_bool(prepare_context, true, "Prepare Context before executor");

DEFINE_int32(num_threads, 1, "Number of threads should be used");

inline double get_current_ms() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+3 * time.tv_sec + 1e-3 * time.tv_usec;
}

// return size of total words
size_t read_datasets(std::vector<paddle::framework::LoDTensor>* out,
                     const std::string& filename) {
  using namespace std;  // NOLINT
  size_t sz = 0;
  fstream fin(filename);
  string line;
  out->clear();
  while (getline(fin, line)) {
    istringstream iss(line);
    vector<int64_t> ids;
    string field;
    while (getline(iss, field, ' ')) {
      ids.push_back(stoi(field));
    }
    if (ids.size() >= 1024 || out->size() >= 100) {
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

void test_multi_threads() {
  /*
    size_t jobs_per_thread = std::min(inputdatas.size() / FLAGS_num_threads,
    inputdatas.size());
    std::vector<size_t> workers(FLAGS_num_threads, jobs_per_thread);
    workers[FLAGS_num_threads - 1] += inputdatas.size() % FLAGS_num_threads;

    std::vector<std::unique_ptr<std::thread>> infer_threads;

    for (size_t i = 0; i < workers.size(); ++i) {
      infer_threads.emplace_back(new std::thread([&, i]() {
        size_t start = i * jobs_per_thread;
        for (size_t j = start; j < start + workers[i]; ++j ) {
          // 0. Call `paddle::framework::InitDevices()` initialize all the
    devices
          // In unittests, this is done in paddle/testing/paddle_gtest_main.cc
          paddle::framework::LoDTensor words;
          auto& srcdata = inputdatas[j];
          paddle::framework::LoD lod{{0, srcdata.size()}};
          words.set_lod(lod);
          int64_t* pdata = words.mutable_data<int64_t>(
              {static_cast<int64_t>(srcdata.size()), 1},
              paddle::platform::CPUPlace());
          memcpy(pdata, srcdata.data(), words.numel() * sizeof(int64_t));

          LOG(INFO) << "thread id: " << i << ", words size:" << words.numel();
          std::vector<paddle::framework::LoDTensor*> cpu_feeds;
          cpu_feeds.push_back(&words);

          paddle::framework::LoDTensor output1;
          std::vector<paddle::framework::LoDTensor*> cpu_fetchs1;
          cpu_fetchs1.push_back(&output1);

          // Run inference on CPU
          if (FLAGS_prepare_vars) {
            if (FLAGS_prepare_context) {
              TestInference<paddle::platform::CPUPlace, false, true>(
                  dirname, cpu_feeds, cpu_fetchs1, FLAGS_repeat, model_combined,
                  FLAGS_use_mkldnn);
            } else {
              TestInference<paddle::platform::CPUPlace, false, false>(
                  dirname, cpu_feeds, cpu_fetchs1, FLAGS_repeat, model_combined,
                  FLAGS_use_mkldnn);
            }
          } else {
            if (FLAGS_prepare_context) {
              TestInference<paddle::platform::CPUPlace, true, true>(
                  dirname, cpu_feeds, cpu_fetchs1, FLAGS_repeat, model_combined,
                  FLAGS_use_mkldnn);
            } else {
              TestInference<paddle::platform::CPUPlace, true, false>(
                  dirname, cpu_feeds, cpu_fetchs1, FLAGS_repeat, model_combined,
                  FLAGS_use_mkldnn);
            }
          }
          //LOG(INFO) << output1.lod();
          //LOG(INFO) << output1.dims();
        }
      }));
    }
    auto start_ms = get_current_ms();
    for (int i = 0; i < FLAGS_num_threads; ++i) {
      infer_threads[i]->join();
    }
    auto stop_ms = get_current_ms();
    LOG(INFO) << "total: " << stop_ms - start_ms << " ms";*/
}

TEST(inference, nlp) {
  if (FLAGS_dirname.empty()) {
    LOG(FATAL) << "Usage: ./example --dirname=path/to/your/model";
  }
  LOG(INFO) << "FLAGS_dirname: " << FLAGS_dirname << std::endl;
  std::string dirname = FLAGS_dirname;

  std::vector<paddle::framework::LoDTensor> datasets;
  size_t num_total_words =
      read_datasets(&datasets, "/home/tangjian/paddle-tj/out.ids.txt");
  LOG(INFO) << "Number of dataset samples(seq len<1024): " << datasets.size();
  LOG(INFO) << "Total number of words: " << num_total_words;

  const bool model_combined = false;

  // 0. Call `paddle::framework::InitDevices()` initialize all the devices
  // 1. Define place, executor, scope
  auto place = paddle::platform::CPUPlace();
  auto executor = paddle::framework::Executor(place);
  auto* scope = new paddle::framework::Scope();

  // 2. Initialize the inference_program and load parameters
  std::unique_ptr<paddle::framework::ProgramDesc> inference_program;
  inference_program = InitProgram(&executor, scope, dirname, model_combined);
  if (FLAGS_use_mkldnn) {
    EnableMKLDNN(inference_program);
  }

  if (FLAGS_num_threads > 1) {
    test_multi_threads();
  } else {
    if (FLAGS_prepare_vars) {
      executor.CreateVariables(*inference_program, scope, 0);
    }
    // always prepare context and burning first time
    std::unique_ptr<paddle::framework::ExecutorPrepareContext> ctx;
    ctx = executor.Prepare(*inference_program, 0);

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

    // for data and run
    auto start_ms = get_current_ms();
    for (size_t i = 0; i < datasets.size(); ++i) {
      feed_targets[feed_target_names[0]] = &(datasets[i]);
      executor.RunPreparedContext(ctx.get(), scope, &feed_targets,
                                  &fetch_targets, !FLAGS_prepare_vars);
    }
    auto stop_ms = get_current_ms();
    LOG(INFO) << "Total infer time: " << (stop_ms - start_ms) / 1000.0 / 60
              << " min, avg time per seq: "
              << (stop_ms - start_ms) / datasets.size() << " ms";
    {  // just for test
      auto* scope = new paddle::framework::Scope();
      paddle::framework::LoDTensor outtensor;
      TestInference<paddle::platform::CPUPlace, false, true>(
          dirname, {&(datasets[0])}, {&outtensor}, FLAGS_repeat, model_combined,
          false);
      delete scope;
    }
  }
  delete scope;
}
