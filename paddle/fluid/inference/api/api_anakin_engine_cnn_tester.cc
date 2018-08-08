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

#include <gflags/gflags.h>
#include <glog/logging.h>  // use glog instead of PADDLE_ENFORCE to avoid importing other paddle header files.
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>  //NOLINT
#include "anakin/utils/logger/logger.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "sys/time.h"
#include "utils.h"

namespace paddle {

DEFINE_string(modelfile, "", "Directory of the inference model and data.");
DEFINE_string(data, "", "Directory of the inference model and data.");
struct Record {
  std::vector<float> data;
};

static void split(const std::string& str, char sep,
                  std::vector<std::string>* pieces) {
  pieces->clear();
  if (str.empty()) {
    return;
  }
  size_t pos = 0;
  size_t next = str.find(sep, pos);
  while (next != std::string::npos) {
    pieces->push_back(str.substr(pos, next - pos));
    pos = next + 1;
    next = str.find(sep, pos);
  }
  if (!str.substr(pos).empty()) {
    pieces->push_back(str.substr(pos));
  }
}

Record ProcessALine(const std::string& line) {
  Record record;
  std::vector<std::string> data_strs;
  split(line, ',', &data_strs);
  for (auto& d : data_strs) {
    record.data.push_back(std::stof(d));
  }
  return record;
}

int Main(int max_batch) {
  AnakinConfig config;
  config.model_file = FLAGS_modelfile;
  config.device = 0;
  config.max_batch_size = max_batch;
  config.target_type = AnakinConfig::TargetType::X86;
  std::string line;
  std::ifstream file(FLAGS_data);
  std::vector<PaddleTensor> inputs;
  std::vector<std::vector<int>> shapes({{4, 1, 1},
                                        {1, 50, 12},
                                        {1, 50, 19},
                                        {1, 50, 1},
                                        {4, 50, 1},
                                        {1, 50, 1},
                                        {5, 50, 1},
                                        {7, 50, 1},
                                        {3, 50, 1}});

  int id = 0;
  int ids[] = {8, 0, 1, 2, 3, 4, 5, 6, 7};
  for (auto& shape : shapes) {
    std::getline(file, line);
    auto record = ProcessALine(line);
    shape.insert(shape.begin(), max_batch);
    PaddleTensor feature;
    feature.name = "input_" + std::to_string(ids[id++]);
    feature.shape = shape;
    feature.data = PaddleBuf(
        record.data.data(),
        sizeof(float) * std::accumulate(shape.begin(), shape.end(), 1,
                                        [](int a, int b) { return a * b; }));
    feature.dtype = PaddleDType::FLOAT32;
    inputs.emplace_back(std::move(feature));
    CHECK_EQ(inputs.back().shape.size(), 4UL);
  }
  auto predictor =
      CreatePaddlePredictor<AnakinConfig, PaddleEngineKind::kAnakin>(config);

  struct timeval cur_time;
  gettimeofday(&cur_time, NULL);
  long t = cur_time.tv_sec * 1000000 + cur_time.tv_usec;

  PaddleTensor tensor_out;
  tensor_out.name = "outnet_con1.tmp_1_gout";
  tensor_out.shape = std::vector<int>({});
  tensor_out.data = PaddleBuf();
  tensor_out.dtype = PaddleDType::FLOAT32;
  std::vector<PaddleTensor> outputs(1, tensor_out);

  for (int i = 0; i < 100000; i++) {
    CHECK(predictor->Run(inputs, &outputs));
  }

  LOG(INFO) << "output.size: " << outputs.size();
  for (auto& tensor : outputs) {
    LOG(INFO) << "output.length: " << tensor.data.length();
    std::stringstream ss;
    float* data = static_cast<float*>(tensor.data.data());
    for (int i = 0; i < std::min(100UL, tensor.data.length()); i++) {
      ss << data[i] << " ";
    }
    LOG(INFO) << "data: " << ss.str();
  }
  gettimeofday(&cur_time, NULL);
  long t2 = cur_time.tv_sec * 1000000 + cur_time.tv_usec;

  std::cout << "max_batch:" << max_batch << ", time:" << (t2 - t) / 1000 << "ms"
            << std::endl;
}
}

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  logger::init(argv[0]);

  paddle::Main(100);
  return 0;
}
