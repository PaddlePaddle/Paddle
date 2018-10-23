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

#include <cassert>
#include <chrono>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <string>

#include "paddle/fluid/inference/api/paddle_inference_api.h"

namespace paddle {

std::string DIRNAME = "./infer_model";
std::string DATA = "./test-image.txt"; 
const int C = 3; // image channel
const int H = 449; // image height
const int W = 581; // image width
// 数据格式
// "<space splitted floats as data>\t<space splitted ints as shape"
// 1. 存储为float32格式。
// 2. 必须减去均值。 CHW三个通道为 mean = 112.15, 109.41, 185.42

struct Record
{
  std::vector<float> data;
  std::vector<int32_t> shape;
};

NativeConfig GetConfig() {
  NativeConfig config;
  config.prog_file=DIRNAME + "/__model__";
  config.param_file=DIRNAME + "/__params__";
  config.fraction_of_gpu_memory = 0.0;
  config.use_gpu = true;
  config.device = 0;
  return config;
}

using Time = decltype(std::chrono::high_resolution_clock::now());

Time time() { return std::chrono::high_resolution_clock::now(); };

double time_diff(Time t1, Time t2) {
  typedef std::chrono::microseconds ms;
  auto diff = t2 - t1;
  ms counter = std::chrono::duration_cast<ms>(diff);
  return counter.count() / 1000.0;
}

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
  std::vector<std::string> columns;
  split(line, '\t', &columns);

  Record record;
  std::vector<std::string> data_strs;
  split(columns[0], ' ', &data_strs);
  for (auto& d : data_strs) {
    record.data.push_back(std::stof(d));
  }

  std::vector<std::string> shape_strs;
  split(columns[1], ' ', &shape_strs);
  for (auto& s : shape_strs) {
    record.shape.push_back(std::stoi(s));
  }
  return record;
}

void test_naive(int batch_size){
  NativeConfig config = GetConfig();
  auto predictor = CreatePaddlePredictor<NativeConfig>(config);
  int height = H;
  int width = W;
  int channel = C;
  int num_sum = height * width * channel * batch_size;
  
  // 1. use fake data
  std::vector<float> data;
  for(int i = 0; i < num_sum; i++) {
    data.push_back(0.0);
  }
  
  PaddleTensor tensor;
  tensor.shape = std::vector<int>({batch_size, channel, height, width});
  tensor.data.Resize(sizeof(float) * batch_size * channel * height * width);
  std::copy(data.begin(), data.end(), static_cast<float*>(tensor.data.data()));
  tensor.dtype = PaddleDType::FLOAT32;

  // 2. read data from file
  // std::string line;
  // std::ifstream file(DATA);
  // std::getline(file, line);
  // auto record = ProcessALine(line);
  // file.close();
  // PaddleTensor tensor;
  // tensor.shape = record.shape;
  // tensor.data =
  //     PaddleBuf(record.data.data(), record.data.size() * sizeof(float));

  std::vector<PaddleTensor> paddle_tensor_feeds(1, tensor);
  PaddleTensor tensor_out;

  std::vector<PaddleTensor> outputs(1, tensor_out);

  predictor->Run(paddle_tensor_feeds, &outputs, batch_size);
  auto time1 = time(); 
  
  for(size_t i = 0; i < 2; i++) {
    std::cout << "Pass " << i << "predict";
    predictor->Run(paddle_tensor_feeds, &outputs, batch_size);
  } 

  auto time2 = time(); 
  std::ofstream ofresult("naive_test_result.txt", std::ios::app);

  std::cout <<"batch: " << batch_size << " predict cost: " << time_diff(time1, time2) / 100.0 << "ms" << std::endl;
  std::cout << outputs.size() << std::endl;

}
}  // namespace paddle

int main(int argc, char** argv) {
  paddle::test_naive(1 << 0);
  return 0;
}