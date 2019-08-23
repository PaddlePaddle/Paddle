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

#include <gflags/gflags.h>
#include <time.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <thread>  // NOLINT
#include <vector>
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "utils/logger/logger.h"

DEFINE_string(model, "", "Directory of the inference model.");
DEFINE_string(datapath, "", "Path of the dataset.");
DEFINE_int32(batch_size, 1, "batch size.");
DEFINE_int32(repeat, 1, "Running the inference program repeat times.");

class Data {
 public:
  Data(std::string file_name, int batch_size)
      : _batch_size(batch_size), _total_length(0) {
    _file.open(file_name);
    _file.seekg(_file.end);
    _total_length = _file.tellg();
    _file.seekg(_file.beg);
  }
  void get_batch_data(std::vector<std::vector<float>>& fea,         // NOLINT
                      std::vector<std::vector<float>>& week_fea,    // NOLINT
                      std::vector<std::vector<float>>& time_fea,    // NOLINT
                      std::vector<long unsigned int>& seq_offset);  // NOLINT

 private:
  std::fstream _file;
  int _total_length;
  int _batch_size;
};

void Data::get_batch_data(
    std::vector<std::vector<float>>& fea,          // NOLINT
    std::vector<std::vector<float>>& week_fea,     // NOLINT
    std::vector<std::vector<float>>& time_fea,     // NOLINT
    std::vector<long unsigned int>& seq_offset) {  // NOLINT
  int seq_num = 0;
  long unsigned int cum = 0;  // NOLINT

  char buf[10000];
  seq_offset.clear();
  seq_offset.push_back(0);
  fea.clear();
  week_fea.clear();
  time_fea.clear();
  while (_file.getline(buf, 10000)) {
    std::vector<std::string> data_vec;
    paddle::inference::split(buf, ':', &data_vec);

    std::vector<std::string> seq;
    paddle::inference::split(data_vec[0], '|', &seq);

    for (auto link : seq) {
      std::vector<float> vec;
      paddle::inference::split_to_float(link, ',', &vec);
      fea.push_back(vec);
    }

    std::vector<float> vec_w;
    paddle::inference::split_to_float(data_vec[2], ',', &vec_w);
    week_fea.push_back(vec_w);

    std::vector<float> vec_t;
    paddle::inference::split_to_float(data_vec[1], ',', &vec_t);
    time_fea.push_back(vec_t);

    cum += seq.size();
    seq_offset.push_back(cum);

    seq_num++;
    if (seq_num >= _batch_size) {
      break;
    }
  }
}

namespace paddle {

contrib::AnakinConfig GetConfig() {
  contrib::AnakinConfig config;
  // using AnakinConfig::X86 if you need to use cpu to do inference
  config.target_type = contrib::AnakinConfig::X86;
  config.model_file = FLAGS_model;
  config.device_id = 0;
  config.init_batch_size = 1000;  // the max number of token
  return config;
}

void set_tensor(std::string name, std::vector<int> shape,
                std::vector<PaddleTensor>& vec) {  // NOLINT
  int sum = 1;
  std::for_each(shape.begin(), shape.end(), [&](int n) { sum *= n; });
  float* data = new float[sum];
  PaddleTensor tensor;
  tensor.name = name;
  tensor.shape = shape;
  tensor.data = PaddleBuf(data, sum);
  tensor.dtype = PaddleDType::FLOAT32;
  vec.push_back(tensor);
}

void single_test() {
  auto config = GetConfig();
  auto predictor =
      CreatePaddlePredictor<contrib::AnakinConfig, PaddleEngineKind::kAnakin>(
          config);

  int max_batch_size = 1000;
  std::string feature_file = FLAGS_datapath;
  Data map_data(feature_file, FLAGS_batch_size);
  std::vector<std::vector<float>> fea;
  std::vector<std::vector<float>> week_fea;
  std::vector<std::vector<float>> time_fea;
  std::vector<long unsigned int> seq_offset;  // NOLINT

  paddle::PaddleTensor tensor_0, tensor_1, tensor_2;
  tensor_0.name = "input_0";
  tensor_1.name = "input_4";
  tensor_2.name = "input_5";

  PaddleTensor tensor_out;
  tensor_out.name = "final_output.tmp_1_gout";
  tensor_out.shape = std::vector<int>({});
  tensor_out.data = PaddleBuf();
  tensor_out.dtype = PaddleDType::FLOAT32;

  std::vector<PaddleTensor> inputs;
  std::vector<PaddleTensor> outputs(1, tensor_out);

  int data_0_dim = 38;
  int data_1_dim = 10;
  int data_2_dim = 10;
  float data_0[max_batch_size * data_0_dim];  // NOLINT
  float data_1[max_batch_size * data_1_dim];  // NOLINT
  float data_2[max_batch_size * data_2_dim];  // NOLINT

  int count = 0;
  while (true) {
    if (count++ > 0) break;  // only run the first batch in ci.
    seq_offset.clear();
    map_data.get_batch_data(fea, week_fea, time_fea, seq_offset);
    if (seq_offset.size() <= 1) {
      LOG(FATAL) << "seq_offset.size() <= 1, exit.";
      break;
    }

    std::vector<std::vector<long unsigned int>> seq_offset_vec;  // NOLINT
    seq_offset_vec.push_back(seq_offset);
    tensor_0.lod = seq_offset_vec;

    int p_shape_0[] = {(int)fea.size(), 1, 1, data_0_dim};       // NOLINT
    int p_shape_1[] = {(int)week_fea.size(), data_1_dim, 1, 1};  // NOLINT
    int p_shape_2[] = {(int)time_fea.size(), data_2_dim, 1, 1};  // NOLINT

    std::vector<int> shape_0(p_shape_0, p_shape_0 + 4);
    std::vector<int> shape_1(p_shape_1, p_shape_1 + 4);
    std::vector<int> shape_2(p_shape_2, p_shape_2 + 4);

    tensor_0.shape = shape_0;
    tensor_1.shape = shape_1;
    tensor_2.shape = shape_2;

    for (int i = 0; i < fea.size(); i++) {
      memcpy(data_0 + i * data_0_dim, &fea[i][0], sizeof(float) * data_0_dim);
    }
    for (int i = 0; i < week_fea.size(); i++) {
      memcpy(data_1 + i * data_1_dim, &week_fea[i][0],
             sizeof(float) * data_1_dim);
    }
    for (int i = 0; i < time_fea.size(); i++) {
      memcpy(data_2 + i * data_2_dim, &time_fea[i][0],
             sizeof(float) * data_2_dim);
    }

    tensor_0.data =
        paddle::PaddleBuf(data_0, fea.size() * sizeof(float) * data_0_dim);
    tensor_1.data =
        paddle::PaddleBuf(data_1, week_fea.size() * sizeof(float) * data_1_dim);
    tensor_2.data =
        paddle::PaddleBuf(data_2, time_fea.size() * sizeof(float) * data_2_dim);

    tensor_0.dtype = paddle::PaddleDType::FLOAT32;
    tensor_1.dtype = paddle::PaddleDType::FLOAT32;
    tensor_2.dtype = paddle::PaddleDType::FLOAT32;

    inputs.clear();
    inputs.push_back(tensor_1);
    inputs.push_back(tensor_2);
    inputs.push_back(tensor_0);

    paddle::inference::Timer timer;
    timer.tic();
    for (int i = 0; i < FLAGS_repeat; i++) predictor->Run(inputs, &outputs);

    paddle::inference::PrintTime(FLAGS_batch_size, FLAGS_repeat, 1, 0,
                                 timer.toc() / FLAGS_repeat);
    LOG(INFO) << "sequence_length = " << seq_offset[seq_offset.size() - 1];

    float* data_o = static_cast<float*>(outputs[0].data.data());
    VLOG(3) << "outputs[0].data.length() = " << outputs[0].data.length();
    for (size_t j = 0; j < outputs[0].data.length(); ++j) {
      VLOG(3) << "output[" << j << "]: " << data_o[j];
    }
  }
}
}  // namespace paddle

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  logger::init(argv[0]);

  paddle::single_test();
  /* multi-threads
  std::vector<std::thread> threads;
  int num = 1;
  for (int i = 0; i < num; i++) {
    LOG(INFO) << " thread id : " << i;
    threads.emplace_back(paddle::single_test);
  }
  for (int i = 0; i < num; i++) {
    threads[i].join();
  }
  threads.clear();
  */

  return 0;
}
