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

/*
 * Use the native and analysis fluid engine to inference the demo.
 * ocr, mobilenet and se_resnext50
 */
void TestVisualPrediction(bool use_mkldnn) {
  std::unique_ptr<PaddlePredictor> predictor;
  AnalysisConfig cfg;
  cfg.param_file = FLAGS_infer_model + "/__params__";
  cfg.prog_file = FLAGS_infer_model + "/__model__";
  cfg.use_gpu = false;
  cfg._use_mkldnn = use_mkldnn;
  cfg.device = 0;
  cfg.enable_ir_optim = true;
  cfg.ir_passes.push_back("fc_gru_fuse_pass");
  predictor =
      CreatePaddlePredictor<AnalysisConfig, PaddleEngineKind::kAnalysis>(cfg);

  // Only have single batch of data.
  std::string line;
  std::ifstream file(FLAGS_infer_data);
  std::getline(file, line);
  auto record = ProcessALine(line);
  file.close();

  // Inference.
  PaddleTensor input;
  input.shape = record.shape;
  input.data =
      PaddleBuf(record.data.data(), record.data.size() * sizeof(float));
  input.dtype = PaddleDType::FLOAT32;

  std::vector<PaddleTensor> outputs_slots;
  Timer timer;
  timer.tic();
  for (int i = 0; i < FLAGS_repeat; i++) {
    predictor->Run({input}, &outputs_slots);
  }
  PrintTime(/*batch size*/ 1, FLAGS_repeat, /*num threads*/ 1, /*thread id*/ 0,
            timer.toc() / FLAGS_repeat);

  VLOG(3) << "output.size " << outputs_slots.size();

  // run native as reference
  auto ref_predictor =
      CreatePaddlePredictor<NativeConfig, PaddleEngineKind::kNative>(cfg);
  std::vector<PaddleTensor> ref_outputs_slots;
  ref_predictor->Run({input}, &ref_outputs_slots);
  CompareResult(outputs_slots, ref_outputs_slots);
  // print what are fused
  AnalysisPredictor *analysis_predictor =
      dynamic_cast<AnalysisPredictor *>(predictor.get());
  auto &fuse_statis = analysis_predictor->analysis_argument()
                          .Get<std::unordered_map<std::string, int>>(
                              framework::ir::kFuseStatisAttr);
  for (auto &item : fuse_statis) {
    LOG(INFO) << "fused " << item.first << " " << item.second;
  }
  int num_ops = 0;
  for (auto &node :
       analysis_predictor->analysis_argument().main_dfg->nodes.nodes()) {
    if (node->IsFunction()) {
      ++num_ops;
    }
  }
  LOG(INFO) << "has num ops: " << num_ops;
}

TEST(Analyzer_vis, analysis) { TestVisualPrediction(/*use_mkldnn*/ false); }
#ifdef PADDLE_WITH_MKLDNN
TEST(Analyzer_vis, analysis_mkldnn) {
  TestVisualPrediction(/*use_mkldnn*/ true);
}
#endif

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
