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

#include <gtest/gtest.h>

#include <fstream>
#include <iostream>

#include "paddle/fluid/inference/tests/api/tester_helper.h"

namespace paddle {
namespace inference {
namespace analysis {

struct Record {
  std::vector<float> data;
  std::vector<int32_t> shape;
};

Record ProcessALine(const std::string &line) {
  VLOG(3) << "process a line";
  std::vector<std::string> columns;
  split(line, '\t', &columns);
  CHECK_EQ(columns.size(), 2UL)
      << "data format error, should be <data>\t<shape>";

  Record record;
  std::vector<std::string> data_strs;
  split(columns[0], ' ', &data_strs);
  for (auto &d : data_strs) {
    record.data.push_back(std::stof(d));
  }

  std::vector<std::string> shape_strs;
  split(columns[1], ' ', &shape_strs);
  for (auto &s : shape_strs) {
    record.shape.push_back(std::stoi(s));
  }
  VLOG(3) << "data size " << record.data.size();
  VLOG(3) << "data shape size " << record.shape.size();
  return record;
}

void SetConfig(AnalysisConfig *cfg) {
  cfg->SetModel(FLAGS_infer_model + "/__model__",
                FLAGS_infer_model + "/__params__");
  cfg->DisableGpu();
  cfg->SwitchIrDebug();
  cfg->SwitchSpecifyInputNames(false);
}

void SetInput(std::vector<std::vector<PaddleTensor>> *inputs) {
  PADDLE_ENFORCE_EQ(
      FLAGS_test_all_data,
      0,
      paddle::platform::errors::Fatal("Only have single batch of data."));
  std::string line;
  std::ifstream file(FLAGS_infer_data);
  std::getline(file, line);
  auto record = ProcessALine(line);

  PaddleTensor input;
  input.shape = record.shape;
  input.dtype = PaddleDType::FLOAT32;
  size_t input_size = record.data.size() * sizeof(float);
  input.data.Resize(input_size);
  memcpy(input.data.data(), record.data.data(), input_size);
  std::vector<PaddleTensor> input_slots;
  input_slots.assign({input});
  (*inputs).emplace_back(input_slots);
}

// Easy for profiling independently.
//  ocr, mobilenet and se_resnext50
void profile(bool use_mkldnn = false) {
  AnalysisConfig cfg;
  SetConfig(&cfg);
  if (use_mkldnn) {
    cfg.EnableMKLDNN();
  }
  // cfg.pass_builder()->TurnOnDebug();
  std::vector<std::vector<PaddleTensor>> outputs;

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  TestPrediction(reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
                 input_slots_all,
                 &outputs,
                 FLAGS_num_threads);
  if (FLAGS_num_threads == 1 && !FLAGS_test_all_data) {
    std::string line;
    std::ifstream file(FLAGS_refer_result);
    std::getline(file, line);
    auto refer = ProcessALine(line);
    file.close();

    PADDLE_ENFORCE_GT(outputs.size(),
                      0,
                      paddle::platform::errors::Fatal(
                          "The size of output should be greater than 0."));
    auto &output = outputs.back().front();
    size_t numel = output.data.length() / PaddleDtypeSize(output.dtype);
    CHECK_EQ(numel, refer.data.size());
    for (size_t i = 0; i < numel; ++i) {
      EXPECT_NEAR(
          static_cast<float *>(output.data.data())[i], refer.data[i], 1e-5);
    }
  }
}

TEST(Analyzer_vis, profile) { profile(); }

#ifdef PADDLE_WITH_MKLDNN
TEST(Analyzer_vis, profile_mkldnn) { profile(true /* use_mkldnn */); }
#endif

// Check the fuse status
TEST(Analyzer_vis, fuse_statis) {
  AnalysisConfig cfg;
  SetConfig(&cfg);
  int num_ops;
  auto predictor = CreatePaddlePredictor<AnalysisConfig>(cfg);
  GetFuseStatis(predictor.get(), &num_ops);
}

// Compare result of NativeConfig and AnalysisConfig
void compare(bool use_mkldnn = false) {
  AnalysisConfig cfg;
  SetConfig(&cfg);
  if (use_mkldnn) {
    cfg.EnableMKLDNN();
  }

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  CompareNativeAndAnalysis(
      reinterpret_cast<const PaddlePredictor::Config *>(&cfg), input_slots_all);
}

TEST(Analyzer_vis, compare) { compare(); }
#ifdef PADDLE_WITH_MKLDNN
TEST(Analyzer_vis, compare_mkldnn) { compare(true /* use_mkldnn */); }
#endif

// Compare Deterministic result
TEST(Analyzer_vis, compare_determine) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  CompareDeterministic(reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
                       input_slots_all);
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
