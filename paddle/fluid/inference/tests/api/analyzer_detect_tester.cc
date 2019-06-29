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

#include <gperftools/heap-checker.h>
#include <gtest/gtest.h>
#include <fstream>
#include <iostream>
#include "paddle/fluid/inference/tests/api/tester_helper.h"
DEFINE_string(infer_shape, "", "data shape file");
DEFINE_int32(sample, 1, "number of sample");

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
  // LOG(INFO) << "data shape size " << record.shape[3];
  return record;
}

void SetConfig(AnalysisConfig *cfg) {
  cfg->SetModel(FLAGS_infer_model + "/model", FLAGS_infer_model + "/params");
  cfg->DisableGpu();
  cfg->SwitchIrDebug();
  cfg->SwitchSpecifyInputNames(false);
  cfg->SetCpuMathLibraryNumThreads(FLAGS_paddle_num_threads);
}

void SetInput(std::vector<std::vector<PaddleTensor>> *inputs,
              const std::string &line, const std::string &shape_line) {
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

// Easy for profiling independently.
//  ocr, mobilenet and se_resnext50
void profile(bool use_mkldnn = false) {
  AnalysisConfig cfg;
  SetConfig(&cfg);
  if (use_mkldnn) {
    cfg.EnableMKLDNN(10);
  }
  // cfg.pass_builder()->TurnOnDebug();
  std::vector<std::vector<PaddleTensor>> outputs;
  std::vector<std::vector<PaddleTensor>> input_slots_all;

  Timer run_timer;
  double elapsed_time = 0;

  int iterations = FLAGS_sample;
  int num_times = FLAGS_repeat;
  auto predictor = CreatePaddlePredictor<AnalysisConfig>(cfg);
  outputs.resize(iterations);

  std::vector<std::thread> threads;

  for (int j = 0; j < num_times; j++) {
    std::ifstream file(FLAGS_infer_data);
    std::ifstream infer_file(FLAGS_infer_shape);
    std::string line;
    std::string shape_line;

    for (int i = 0; i < iterations; i++) {
      threads.emplace_back([&, i]() {
        std::getline(file, line);
        std::getline(infer_file, shape_line);
        SetInput(&input_slots_all, line, shape_line);

        run_timer.tic();
        predictor->Run(input_slots_all[0], &outputs[0], FLAGS_batch_size);
        elapsed_time += run_timer.toc();
      });
      threads[0].join();
      threads.clear();
      if (i % 100 == 0) LOG(INFO) << i << " samples";
      std::vector<std::vector<PaddleTensor>>().swap(input_slots_all);
    }

    file.close();
    infer_file.close();
  }

  auto batch_latency = elapsed_time / (iterations * num_times);
  PrintTime(FLAGS_batch_size, num_times, FLAGS_num_threads, 0, batch_latency,
            iterations, VarType::FP32);
}

TEST(Analyzer_vis, profile) { profile(); }

#ifdef PADDLE_WITH_MKLDNN
TEST(Analyzer_vis, profile_mkldnn) { profile(true /* use_mkldnn */); }
#endif

}  // namespace analysis
}  // namespace inference
}  // namespace paddle

// following lines are used for pprof
int main(int argc, char **argv) {
  HeapLeakChecker heap_checker("test_foo");
  FLAGS_infer_model = "third_party/inference_demo/face_model/densebox";
  FLAGS_infer_data = "third_party/inference_demo/face_model/detect_input.txt";
  FLAGS_infer_shape = "third_party/inference_demo/face_model/shape.txt";
  FLAGS_paddle_num_threads = 4;
  FLAGS_repeat = 1;
  FLAGS_batch_size = 1;
  FLAGS_sample = 10;
  paddle::inference::analysis::profile(true);
  std::cout << heap_checker.NoLeaks() << std::endl;
}
