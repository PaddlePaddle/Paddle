// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <gtest/gtest.h>
#include <string>
#include <thread>  // NOLINT
#include <vector>
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/inference/analysis/analyzer.h"
#include "paddle/fluid/inference/analysis/ut_helper.h"
#include "paddle/fluid/inference/api/analysis_predictor.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_inference_pass.h"
#include "paddle/fluid/platform/profiler.h"

DEFINE_string(infer_model, "", "model path");
DEFINE_string(infer_data, "", "data file");
DEFINE_int32(batch_size, 1, "batch size.");
DEFINE_int32(repeat, 1, "Running the inference program repeat times.");
DEFINE_bool(test_all_data, false, "Test the all dataset in data file.");
DEFINE_int32(num_threads, 1, "Running the inference program in multi-threads.");
DEFINE_bool(use_analysis, true,
            "Running the inference program in analysis mode.");

namespace paddle {
namespace inference {

using contrib::AnalysisConfig;

void CompareResult(const std::vector<PaddleTensor> &outputs,
                   const std::vector<PaddleTensor> &ref_outputs) {
  EXPECT_GT(outputs.size(), 0UL);
  EXPECT_EQ(outputs.size(), ref_outputs.size());
  for (size_t i = 0; i < outputs.size(); i++) {
    auto &out = outputs[i];
    auto &ref_out = ref_outputs[i];
    size_t size = VecReduceToInt(out.shape);
    size_t ref_size = VecReduceToInt(ref_out.shape);
    EXPECT_GT(size, 0);
    EXPECT_EQ(size, ref_size);
    EXPECT_EQ(out.dtype, ref_out.dtype);
    switch (out.dtype) {
      case PaddleDType::INT64: {
        int64_t *pdata = static_cast<int64_t *>(out.data.data());
        int64_t *pdata_ref = static_cast<int64_t *>(ref_out.data.data());
        for (size_t j = 0; j < size; ++j) {
          EXPECT_EQ(pdata_ref[j], pdata[j]);
        }
        break;
      }
      case PaddleDType::FLOAT32: {
        float *pdata = static_cast<float *>(out.data.data());
        float *pdata_ref = static_cast<float *>(ref_out.data.data());
        for (size_t j = 0; j < size; ++j) {
          EXPECT_NEAR(pdata_ref[j], pdata[j], 1e-3);
        }
        break;
      }
    }
  }
}

std::unique_ptr<PaddlePredictor> CreateTestPredictor(
    const AnalysisConfig &config, bool use_analysis = true) {
  if (use_analysis) {
    return CreatePaddlePredictor<contrib::AnalysisConfig,
                                 PaddleEngineKind::kAnalysis>(config);
  } else {
    return CreatePaddlePredictor<NativeConfig, PaddleEngineKind::kNative>(
        config);
  }
}

size_t GetSize(const PaddleTensor &out) { return VecReduceToInt(out.shape); }

std::unordered_map<std::string, int> GetFuseStatis(AnalysisConfig config,
                                                   int *num_ops) {
  auto predictor = CreateTestPredictor(config);
  AnalysisPredictor *analysis_predictor =
      dynamic_cast<AnalysisPredictor *>(predictor.get());
  auto &fuse_statis = analysis_predictor->analysis_argument()
                          .Get<std::unordered_map<std::string, int>>(
                              framework::ir::kFuseStatisAttr);
  for (auto &item : fuse_statis) {
    LOG(INFO) << "fused " << item.first << " " << item.second;
  }
  int num = 0;
  for (auto &node :
       analysis_predictor->analysis_argument().main_dfg->nodes.nodes()) {
    if (node->IsFunction()) {
      ++num;
    }
  }
  *num_ops = num;
  return fuse_statis;
}

void TestOneThreadPrediction(
    const AnalysisConfig &config,
    const std::vector<std::vector<PaddleTensor>> &inputs,
    std::vector<PaddleTensor> *outputs, bool use_analysis = true) {
  int batch_size = FLAGS_batch_size;
  int num_times = FLAGS_repeat;
  auto predictor = CreateTestPredictor(config, use_analysis);
  Timer timer;
  timer.tic();
  for (int i = 0; i < num_times; i++) {
    for (size_t j = 0; j < inputs.size(); j++) {
      predictor->Run(inputs[j], outputs);
    }
  }
  PrintTime(batch_size, num_times, 1, 0, timer.toc() / num_times,
            inputs.size());
}

void TestMultiThreadPrediction(
    const AnalysisConfig &config,
    const std::vector<std::vector<PaddleTensor>> &inputs,
    std::vector<PaddleTensor> *outputs, int num_threads,
    bool use_analysis = true) {
  int batch_size = FLAGS_batch_size;
  int num_times = FLAGS_repeat;
  std::vector<std::thread> threads;
  std::vector<std::unique_ptr<PaddlePredictor>> predictors;
  // TODO(yanchunwei): Bug here, the analyzer phase can't be parallelled
  // because AttentionLSTM's hard code nodeid will be damanged.
  for (int tid = 0; tid < num_threads; ++tid) {
    predictors.emplace_back(CreateTestPredictor(config, use_analysis));
  }
  for (int tid = 0; tid < num_threads; ++tid) {
    threads.emplace_back([&, tid]() {
      // Each thread should have local inputs and outputs.
      // The inputs of each thread are all the same.
      std::vector<std::vector<PaddleTensor>> inputs_tid = inputs;
      std::vector<PaddleTensor> outputs_tid;
      Timer timer;
      timer.tic();
      for (int i = 0; i < num_times; i++) {
        for (size_t j = 0; j < inputs_tid.size(); j++) {
          predictors[tid]->Run(inputs_tid[j], &outputs_tid);
        }
      }
      PrintTime(batch_size, num_times, num_threads, tid,
                timer.toc() / num_times, inputs_tid.size());
    });
  }
  for (int i = 0; i < num_threads; ++i) {
    threads[i].join();
  }
}

void TestPrediction(const AnalysisConfig &config,
                    const std::vector<std::vector<PaddleTensor>> &inputs,
                    std::vector<PaddleTensor> *outputs, int num_threads,
                    bool use_analysis = FLAGS_use_analysis) {
  LOG(INFO) << "use_analysis: " << use_analysis;
  if (num_threads == 1) {
    TestOneThreadPrediction(config, inputs, outputs, use_analysis);
  } else {
    TestMultiThreadPrediction(config, inputs, outputs, num_threads,
                              use_analysis);
  }
}

void CompareNativeAndAnalysis(
    const AnalysisConfig &config,
    const std::vector<std::vector<PaddleTensor>> &inputs) {
  std::vector<PaddleTensor> native_outputs, analysis_outputs;
  TestOneThreadPrediction(config, inputs, &native_outputs, false);
  TestOneThreadPrediction(config, inputs, &analysis_outputs, true);
  CompareResult(analysis_outputs, native_outputs);
}

}  // namespace inference
}  // namespace paddle
