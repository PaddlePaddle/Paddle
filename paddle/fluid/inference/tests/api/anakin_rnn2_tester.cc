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

#define BUFFER_SIZE (10000)

DEFINE_string(model, "", "Directory of the inference model.");
DEFINE_string(datapath, "", "Path of the dataset.");
DEFINE_int32(batch_size, 1, "Batch size per execution.");
DEFINE_int32(repeats, 1, "Number of iterations.");
DEFINE_int32(
    start_line, 0,
    "The starting line of the text file read (this line will be read).");
DEFINE_int32(end_line, SIZE_MAX,
             "The ending line of the text file read (this line will be read).");
DEFINE_int32(max_batch_size, 4000,
             "Max batch size for Anakin memory allocation.");

class Data {
 public:
  Data(std::string file_name, size_t batch_size, size_t start, size_t end)
      : _batch_size(batch_size), _total_length(0), _inputs_size(6) {
    _file.open(file_name);
    _file.seekg(_file.end);
    _total_length = _file.tellg();
    _file.seekg(_file.beg);
    read_file_to_vec(start, end);
    reset_current_line();
  }
  void reset_current_line();
  void read_file_to_vec(const size_t start, const size_t end);
  int get_next_batches(std::vector<std::vector<float>>* inputs,
                       std::vector<std::vector<size_t>>* seq_offsets);

 private:
  std::fstream _file;
  int _batch_size;
  size_t _total_length;
  size_t _inputs_size;
  std::vector<std::string> _lines;
  size_t _current_line;
};

void Data::read_file_to_vec(const size_t start, const size_t end) {
  std::string line;
  size_t count = 0;
  _lines.clear();
  while (std::getline(_file, line)) {
    if (count >= start && count <= end) {
      _lines.push_back(line);
    }
    count++;
  }
}

void Data::reset_current_line() { _current_line = 0; }

int Data::get_next_batches(std::vector<std::vector<float>>* data,
                           std::vector<std::vector<size_t>>* offsets) {
  PADDLE_ENFORCE(!_lines.empty());
  data->clear();
  offsets->clear();
  data->resize(_inputs_size);
  offsets->resize(_inputs_size);
  for (auto& offset : *offsets) {
    offset.push_back(0);
  }

  int seq_num = -1;
  int pre_query_index = -1;
  while (_current_line < _lines.size()) {
    int cur_query_index = -1;
    std::vector<std::string> line;
    paddle::inference::split(_lines[_current_line], ';', &line);
    PADDLE_ENFORCE(line.size() == _inputs_size + 1);
    for (size_t i = 0; i < line.size(); i++) {
      std::vector<float> float_v;
      paddle::inference::split_to_float(line[i], ' ', &float_v);
      if (i == 0) {
        cur_query_index = float_v[0];
        if (pre_query_index != -1 && cur_query_index != pre_query_index) {
          return seq_num;
        }
        seq_num++;
        _current_line++;
      } else {
        if (float_v.size() == 0) {
          float_v.push_back(-1);
        }
        (*data)[i - 1].insert((*data)[i - 1].end(), float_v.begin(),
                              float_v.end());
        (*offsets)[i - 1].push_back((*offsets)[i - 1][seq_num] +
                                    float_v.size());
      }
    }
    if (seq_num + 1 >= _batch_size) {
      return seq_num;
    } else {
      pre_query_index = cur_query_index;
    }
  }
  return seq_num;
}

namespace paddle {

contrib::AnakinConfig GetConfig() {
  contrib::AnakinConfig config;
  // using AnakinConfig::X86 if you need to use cpu to do inference
  config.target_type = contrib::AnakinConfig::NVGPU;
  config.model_file = FLAGS_model;
  config.device = 0;
  config.max_batch_size = FLAGS_max_batch_size;
  return config;
}

void single_test() {
  auto config = GetConfig();
  auto predictor =
      CreatePaddlePredictor<contrib::AnakinConfig, PaddleEngineKind::kAnakin>(
          config);

  std::string feature_file = FLAGS_datapath;
  Data data(feature_file, FLAGS_batch_size, FLAGS_start_line, FLAGS_end_line);

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

  paddle::inference::Timer timer;
  for (int i = 0; i < FLAGS_repeats; i++) {
    data.reset_current_line();
    while (data.get_next_batches(&inputs, &seq_offsets) >= 0) {
      for (size_t j = 0; j < input_tensors.size(); j++) {
        input_tensors[j].data =
            paddle::PaddleBuf(&inputs[j][0], inputs[j].size() * sizeof(float));
        input_tensors[j].lod =
            std::vector<std::vector<size_t>>({seq_offsets[j]});
        input_tensors[j].shape =
            std::vector<int>({static_cast<int>(inputs[j].size()), 1, 1, 1});
      }
      timer.tic();
      ASSERT_TRUE(predictor->Run(input_tensors, &output_tensors));
    }
  }
  paddle::inference::PrintTime(FLAGS_batch_size, FLAGS_repeats, 1, 0,
                               timer.toc() / FLAGS_repeats);
}
}  // namespace paddle

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  paddle::single_test();
  return 0;
}
