/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <fstream>
#include <iostream>

#include "test/cpp/inference/api/tester_helper.h"

namespace paddle {
namespace inference {
namespace analysis {

struct Record {
  std::vector<float> data;
  std::vector<int32_t> shape;
  Record() : data(), shape() {}
};

Record ProcessALine(const std::string &line) {
  std::vector<std::string> columns;
  split(line, '\t', &columns);
  PADDLE_ENFORCE_EQ(columns.size(),
                    2UL,
                    common::errors::InvalidArgument(
                        "Data format is invalid, should be <data>\t<shape>"));

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

  return record;
}

void SetInput(std::vector<std::vector<PaddleTensor>> *inputs) {
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

void SetConfig(AnalysisConfig *cfg, bool use_mkldnn = false) {
  cfg->SetModel(FLAGS_infer_model + "/inference.pdmodel",
                FLAGS_infer_model + "/inference.pdiparams");

  if (use_mkldnn) {
    cfg->EnableMKLDNN();
    cfg->SwitchIrOptim();
  }
}

// Compare results of NativeConfig and AnalysisConfig
void compare(bool use_mkldnn = false) {
  AnalysisConfig cfg;
  SetConfig(&cfg, use_mkldnn);

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  CompareNativeAndAnalysis(
      reinterpret_cast<const PaddlePredictor::Config *>(&cfg), input_slots_all);
}

TEST(Analyzer_vit_ocr, compare) { compare(); }

#ifdef PADDLE_WITH_DNNL
TEST(Analyzer_vit_ocr, compare_mkldnn) { compare(true /* use_mkldnn */); }
#endif

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
