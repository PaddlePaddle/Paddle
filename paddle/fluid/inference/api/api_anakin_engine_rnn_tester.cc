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
#include <algorithm>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>
#include "net.h"
#include "paddle_inference_api.h"

DEFINE_string(modeldir, "", "Directory of the inference model.");
DEFINE_string(datapath, "", "Path of the dataset.");
DEFINE_int32(batch_size, 1, "batch size.");

std::vector<std::string> string_split(std::string in_str,
                                      std::string delimiter) {
  std::vector<std::string> seq;
  int found = in_str.find(delimiter);
  int pre_found = -1;
  while (found != std::string::npos) {
    if (pre_found == -1) {
      seq.push_back(in_str.substr(0, found));
    } else {
      seq.push_back(in_str.substr(pre_found + delimiter.length(),
                                  found - delimiter.length() - pre_found));
    }
    pre_found = found;
    found = in_str.find(delimiter, pre_found + delimiter.length());
  }
  seq.push_back(
      in_str.substr(pre_found + 1, in_str.length() - (pre_found + 1)));
  return seq;
}
std::vector<std::string> string_split(std::string in_str,
                                      std::vector<std::string>& delimiter) {
  std::vector<std::string> in;
  std::vector<std::string> out;
  out.push_back(in_str);
  for (auto del : delimiter) {
    in = out;
    out.clear();
    for (auto s : in) {
      auto out_s = string_split(s, del);
      for (auto o : out_s) {
        out.push_back(o);
      }
    }
  }
  return out;
}

class Data {
 public:
  Data(std::string file_name, int batch_size)
      : _batch_size(batch_size), _total_length(0) {
    _file.open(file_name);
    _file.seekg(_file.end);
    _total_length = _file.tellg();
    _file.seekg(_file.beg);
  }
  void get_batch_data(std::vector<std::vector<float>>& fea,
                      std::vector<std::vector<float>>& week_fea,
                      std::vector<std::vector<float>>& time_fea,
                      std::vector<long unsigned int>& seq_offset);

 private:
  std::fstream _file;
  int _total_length;
  int _batch_size;
};

void Data::get_batch_data(std::vector<std::vector<float>>& fea,
                          std::vector<std::vector<float>>& week_fea,
                          std::vector<std::vector<float>>& time_fea,
                          std::vector<long unsigned int>& seq_offset) {
  int seq_num = 0;
  long unsigned int cum = 0;

  char buf[10000];
  seq_offset.clear();
  seq_offset.push_back(0);
  fea.clear();
  week_fea.clear();
  time_fea.clear();
  while (_file.getline(buf, 10000)) {
    std::string s = buf;
    std::vector<std::string> deli_vec = {":"};
    std::vector<std::string> data_vec = string_split(s, deli_vec);

    std::vector<std::string> seq;
    seq = string_split(data_vec[0], {"|"});

    for (auto link : seq) {
      std::vector<std::string> data = string_split(link, ",");
      std::vector<float> vec;
      for (int i = 0; i < data.size(); i++) {
        vec.push_back(atof(data[i].c_str()));
      }
      fea.push_back(vec);
    }
    std::vector<std::string> week_data;
    std::vector<std::string> time_data;

    week_data = string_split(data_vec[2], ",");
    std::vector<float> vec_w;
    for (int i = 0; i < week_data.size(); i++) {
      vec_w.push_back(atof(week_data[i].c_str()));
    }
    week_fea.push_back(vec_w);

    time_data = string_split(data_vec[1], ",");
    std::vector<float> vec_t;
    for (int i = 0; i < time_data.size(); i++) {
      vec_t.push_back(atof(time_data[i].c_str()));
    }
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

AnakinConfig GetConfig() {
  AnakinConfig config;
  // using AnakinConfig::X86 if you need to use cpu to do inference
  config.target_type = AnakinConfig::X86;
  config.model_file = FLAGS_modeldir;
  config.device = 0;
  config.max_batch_size = 128;
  return config;
}

void set_tensor(std::string name, std::vector<int> shape,
                std::vector<PaddleTensor>& vec) {
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
  AnakinConfig config = GetConfig();
  auto predictor =
      CreatePaddlePredictor<AnakinConfig, PaddleEngineKind::kAnakin>(config);

  int max_batch_size = 1000;
  std::string feature_file = FLAGS_datapath;
  Data map_data(feature_file, FLAGS_batch_size);
  std::vector<std::vector<float>> fea;
  std::vector<std::vector<float>> week_fea;
  std::vector<std::vector<float>> time_fea;
  std::vector<long unsigned int> seq_offset;

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

  float data_0[max_batch_size * 38];
  float data_1[max_batch_size * 10];
  float data_2[max_batch_size * 10];

  int count = 0;
  while (true) {
    if (count++ > 5) break;
    seq_offset.clear();
    map_data.get_batch_data(fea, week_fea, time_fea, seq_offset);
    if (seq_offset.size() <= 1) {
      LOG(FATAL) << "seq_offset.size() <= 1, exit.";
      break;
    }

    std::vector<std::vector<long unsigned int>> seq_offset_vec;
    seq_offset_vec.push_back(seq_offset);
    tensor_0.lod = seq_offset_vec;

    int p_shape_0[] = {(int)fea.size(), 1, 1, 38};
    int p_shape_1[] = {(int)week_fea.size(), 10, 1, 1};
    int p_shape_2[] = {(int)time_fea.size(), 10, 1, 1};

    std::vector<int> shape_0(p_shape_0, p_shape_0 + 4);
    std::vector<int> shape_1(p_shape_1, p_shape_1 + 4);
    std::vector<int> shape_2(p_shape_2, p_shape_2 + 4);

    tensor_0.shape = shape_0;
    tensor_1.shape = shape_1;
    tensor_2.shape = shape_2;

    for (int i = 0; i < fea.size(); i++) {
      memcpy(data_0 + i * 38, &fea[i][0], sizeof(float) * 38);
    }
    for (int i = 0; i < week_fea.size(); i++) {
      memcpy(data_1 + i * 10, &week_fea[i][0], sizeof(float) * 10);
    }
    for (int i = 0; i < time_fea.size(); i++) {
      memcpy(data_2 + i * 10, &time_fea[i][0], sizeof(float) * 10);
    }

    tensor_0.data = paddle::PaddleBuf(data_0, fea.size() * sizeof(float) * 38);
    tensor_1.data =
        paddle::PaddleBuf(data_1, week_fea.size() * sizeof(float) * 10);
    tensor_2.data =
        paddle::PaddleBuf(data_2, time_fea.size() * sizeof(float) * 10);

    tensor_0.dtype = paddle::PaddleDType::FLOAT32;
    tensor_1.dtype = paddle::PaddleDType::FLOAT32;
    tensor_2.dtype = paddle::PaddleDType::FLOAT32;

    inputs.clear();
    inputs.push_back(tensor_1);
    inputs.push_back(tensor_2);
    inputs.push_back(tensor_0);

    predictor->Run(inputs, &outputs);

    float* data_o = static_cast<float*>(outputs[0].data.data());
    LOG(INFO) << "outputs[0].data.length() = " << outputs[0].data.length();
    for (size_t j = 0; j < outputs[0].data.length(); ++j) {
      LOG(INFO) << "output[" << j << "]: " << data_o[j];
    }
  }
}
}

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  logger::init(argv[0]);

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

  return 0;
}
