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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"

#define BUFFER_SIZE 10000

DEFINE_string(model, "", "Directory of the inference model.");
DEFINE_string(datapath, "", "Path of the dataset.");
DEFINE_int32(batch_size, 1, "batch size.");

class Data {
 public:
  Data(std::string file_name, int batch_size)
      : _batch_size(batch_size), _total_length(0), _inputs_size(6) {
    _file.open(file_name);
    _file.seekg(_file.end);
    _total_length = _file.tellg();
    _file.seekg(_file.beg);
  }
  void get_batch_data(std::vector<std::vector<float>>* inputs,
                      std::vector<std::vector<size_t>>* seq_offsets);

 private:
  std::fstream _file;
  int _batch_size;
  int _total_length;
  size_t _inputs_size;
};

void Data::get_batch_data(std::vector<std::vector<float>>* inputs,
                          std::vector<std::vector<size_t>>* seq_offsets) {
  char buf[BUFFER_SIZE];

  inputs->clear();
  seq_offsets->clear();
  inputs->resize(_inputs_size);
  seq_offsets->resize(_inputs_size);
  for (auto& offset : *seq_offsets) {
    offset.push_back(0);
  }

  while (_file.getline(buf, BUFFER_SIZE)) {
    static int seq_num = 0;
    std::vector<std::string> line;
    paddle::inference::split(buf, ';', &line);
    PADDLE_ENFORCE(line.size() == _inputs_size + 1);
    for (size_t i = 1; i < line.size(); i++) {
      std::vector<float> float_v;
      paddle::inference::split_to_float(line[i], ' ', &float_v);
      if (float_v.size() == 0) {
        float_v.push_back(-1);
      }
      (*inputs)[i - 1].insert((*inputs)[i - 1].end(), float_v.begin(),
                              float_v.end());
      (*seq_offsets)[i - 1].push_back((*seq_offsets)[i - 1][seq_num] +
                                      float_v.size());
    }
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
  config.target_type = contrib::AnakinConfig::NVGPU;
  config.model_file = FLAGS_model;
  config.device = 0;
  config.max_batch_size = 1000;  // the max number of token
  return config;
}

void single_test() {
  auto config = GetConfig();
  auto predictor =
      CreatePaddlePredictor<contrib::AnakinConfig, PaddleEngineKind::kAnakin>(
          config);

  std::string feature_file = FLAGS_datapath;
  Data map_data(feature_file, FLAGS_batch_size);

  std::vector<std::vector<float>> inputs;
  std::vector<std::vector<size_t>> seq_offsets;

  const std::vector<std::string> input_names{
      "q_basic_input",    "q_bigram0_input", "pt_basic_input",
      "pt_bigram0_input", "pa_basic_input",  "pa_bigram0_input"};
  std::vector<paddle::PaddleTensor> input_tensors;
  std::vector<paddle::PaddleTensor> output_tensors;
  for (auto& name : input_names) {
    paddle::PaddleTensor tensor;
    tensor.name = name;
    tensor.dtype = PaddleDType::FLOAT32;
    input_tensors.push_back(tensor);
  }

  PaddleTensor tensor_out;
  tensor_out.name = "qps_out";
  tensor_out.shape = std::vector<int>({});
  tensor_out.data = PaddleBuf();
  tensor_out.dtype = PaddleDType::FLOAT32;
  output_tensors.push_back(tensor_out);

  seq_offsets.clear();
  map_data.get_batch_data(&inputs, &seq_offsets);
  for (size_t i = 0; i < input_tensors.size(); i++) {
    input_tensors[i].data =
        paddle::PaddleBuf(&inputs[i][0], inputs[i].size() * sizeof(float));
    input_tensors[i].lod = std::vector<std::vector<size_t>>({seq_offsets[i]});
    input_tensors[i].shape =
        std::vector<int>({static_cast<int>(inputs[i].size()), 1, 1, 1});
  }

  ASSERT_TRUE(predictor->Run(input_tensors, &output_tensors));

  const float* data_o = static_cast<float*>(output_tensors[0].data.data());
  LOG(INFO) << "outputs[0].data.size() = "
            << output_tensors[0].data.length() / sizeof(float);
  for (size_t j = 0; j < output_tensors[0].data.length() / sizeof(float); ++j) {
    LOG(INFO) << "output[" << j << "]: " << data_o[j];
  }
}
}  // namespace paddle

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  paddle::single_test();
  return 0;
}
