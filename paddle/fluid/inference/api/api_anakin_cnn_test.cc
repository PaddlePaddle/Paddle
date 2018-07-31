#include <gflags/gflags.h>
#include <glog/logging.h>  // use glog instead of PADDLE_ENFORCE to avoid importing other paddle header files.
#include <gtest/gtest.h>
#include <fstream>
#include <iostream>
#include <thread>  //NOLINT
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
  std::string line;
  std::ifstream file(FLAGS_data);
  std::vector<PaddleTensor> inputs;
  std::vector<std::vector<int>> shapes({{4},
                                        {1, 50, 12},
                                        {1, 50, 19},
                                        {1, 50, 1},
                                        {4, 50, 1},
                                        {1, 50, 1},
                                        {5, 50, 1},
                                        {7, 50, 1},
                                        {3, 50, 1}});
  for (auto& shape : shapes) {
    std::getline(file, line);
    auto record = ProcessALine(line);
    shape.insert(shape.begin(), max_batch);
    PaddleTensor feature{
        .name = "",
        .shape = shape,
        .data =
            PaddleBuf(record.data.data(),
                      sizeof(float) *
                          std::accumulate(shape.begin(), shape.end(), 1,
                                          [](int a, int b) { return a * b; })),
        .dtype = PaddleDType::FLOAT32};
    inputs.emplace_back(std::move(feature));
  }
  auto predictor =
      CreatePaddlePredictor<AnakinConfig, PaddleEngineKind::kAnakin>(config);
  std::vector<PaddleTensor> outputs;

  struct timeval cur_time;
  gettimeofday(&cur_time, NULL);
  long t = cur_time.tv_sec * 1000000 + cur_time.tv_usec;
  for (int i = 0; i < 10000; i++) CHECK(predictor->Run(inputs, &outputs));
  gettimeofday(&cur_time, NULL);
  long t2 = cur_time.tv_sec * 1000000 + cur_time.tv_usec;
  std::cout << "10000 iteration, max_batch:" << max_batch
            << ", time:" << (t2 - t) / 1000 << "ms" << std::endl;
}

TEST(Anakin, cnn) { Main(1); }
}