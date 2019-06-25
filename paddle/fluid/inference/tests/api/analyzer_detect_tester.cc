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
DEFINE_string(infer_shape, "", "data shape file");

namespace paddle {
namespace inference {
namespace analysis {

struct Record {
  std::vector<float> data;
  std::vector<int32_t> shape;
};

Record ProcessALine(const std::string &line, const std::string &shape_line) {
  VLOG(3) << "process a line";
  std::vector<std::string> columns;

  Record record;
  std::vector<std::string> data_strs;
  split(line, ' ', &data_strs);
  for (auto &d : data_strs) {
    record.data.push_back(std::stof(d));
  }

  std::vector<std::string> shape_strs;
  split(shape_line, ' ', &shape_strs);
  for (auto &s : shape_strs) {
    record.shape.push_back(std::stoi(s));
  }
  // VLOG(3) << "data size " << record.data.size();
  // VLOG(3) << "data shape size " << record.shape.size();
  VLOG(2) << "data shape size " << record.shape[3];
  return record;
}

void SetConfig(AnalysisConfig *cfg) {
  cfg->SetModel(FLAGS_infer_model + "/model", FLAGS_infer_model + "/params");
  cfg->DisableGpu();
  cfg->SwitchIrDebug();
  cfg->SwitchSpecifyInputNames(false);
  cfg->SetCpuMathLibraryNumThreads(FLAGS_paddle_num_threads);
}

void SetInput(std::vector<std::vector<PaddleTensor>> *inputs) {
  std::string line;
  std::ifstream file(FLAGS_infer_data);
  std::string shape_line;
  std::ifstream infer_file(FLAGS_infer_shape);

  int iteration = FLAGS_test_all_data ? 1000 : 1;
  for (int k = 0; k < iteration; k++) {
    std::getline(file, line);
    std::getline(infer_file, shape_line);
    auto record = ProcessALine(line, shape_line);

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
}

// Easy for profiling independently.
//  ocr, mobilenet and se_resnext50
void profile(bool use_mkldnn = false) {
  AnalysisConfig cfg;
  SetConfig(&cfg);
  if (use_mkldnn) {
    cfg.EnableMKLDNN();
    cfg.pass_builder()->AppendPass("fc_mkldnn_pass");
  }
  // cfg.pass_builder()->TurnOnDebug();
  std::vector<std::vector<PaddleTensor>> outputs;

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  TestPrediction(reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
                 input_slots_all, &outputs, FLAGS_num_threads);
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
    cfg.pass_builder()->AppendPass("fc_mkldnn_pass");
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
