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
#include <cmath>
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"

#define BUFFER_SIZE (10000)
#define COMPARE_OUTPUTS (1)
#define PRINT_INPUTS (0)

DEFINE_string(model, "", "Directory of the inference model.");
DEFINE_string(datapath, "", "Path of the dataset.");
DEFINE_string(truthpath, "", "Path of the dataset.");
DEFINE_int32(batch_size, 1, "Batch size per execution.");
DEFINE_int32(repeats, 1, "Number of iterations.");
DEFINE_int32(
    start_line, 0,
    "The starting line of the text file read (this line will be read).");
DEFINE_int32(end_line, 1000000,
             "The ending line of the text file read (this line will be read).");
DEFINE_int32(init_batch_size, 40,
             "Max batch size for Anakin memory allocation.");
DEFINE_int32(threads_num, 2, "Threads num for Anakin.");

class Data {
 public:
  Data(std::string file_name, size_t batch_size, size_t start = 0,
       size_t end = 1000000)
      : _batch_size(batch_size), _total_length(0), _inputs_size(6) {
    _file.open(file_name);
    _file.seekg(_file.end);
    _total_length = _file.tellg();
    _file.seekg(_file.beg);
    read_file_to_vec(start, end);
    reset_current_line();
  }
  void reset_current_line();
  const std::vector<std::string>& get_lines();
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

const std::vector<std::string>& Data::get_lines() { return _lines; }

void Data::reset_current_line() { _current_line = 0; }

int Data::get_next_batches(std::vector<std::vector<float>>* data,
                           std::vector<std::vector<size_t>>* offsets) {
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

  std::map<std::string, std::vector<int>> init_inputs_shape;
  init_inputs_shape["q_basic"] = std::vector<int>({1000, 1, 1, 1});
  init_inputs_shape["q_bigram0"] = std::vector<int>({1000, 1, 1, 1});
  init_inputs_shape["pt_basic"] = std::vector<int>({2000, 1, 1, 1});
  init_inputs_shape["pa_basic"] = std::vector<int>({4000, 1, 1, 1});
  init_inputs_shape["pa_bigram0"] = std::vector<int>({4000, 1, 1, 1});
  init_inputs_shape["pt_bigram0"] = std::vector<int>({2000, 1, 1, 1});

  // using AnakinConfig::X86 if you need to use cpu to do inference
  config.target_type = contrib::AnakinConfig::NVGPU;
  config.model_file = FLAGS_model;
  config.device_id = 0;
  config.init_batch_size = FLAGS_init_batch_size;
  config.init_inputs_shape = init_inputs_shape;
  config.re_allocable = false;
  return config;
}

void single_test(PaddlePredictor* predictor_master) {
  auto predictor = predictor_master->Clone();

  Data data(FLAGS_datapath, FLAGS_batch_size, FLAGS_start_line, FLAGS_end_line);

  std::vector<std::vector<float>> inputs;
  std::vector<std::vector<size_t>> seq_offsets;
  std::vector<float> compare_outputs;

  const std::vector<std::string> input_names{"q_basic",  "q_bigram0",
                                             "pt_basic", "pt_bigram0",
                                             "pa_basic", "pa_bigram0"};
  std::vector<PaddleTensor> input_tensors;
  std::vector<PaddleTensor> output_tensors;
  for (auto& name : input_names) {
    PaddleTensor tensor;
    tensor.name = name;
    tensor.dtype = PaddleDType::FLOAT32;
    input_tensors.push_back(tensor);
  }

  PaddleTensor tensor_out;
  tensor_out.name = "save_infer_model/scale_0";
  tensor_out.shape = std::vector<int>({});
  tensor_out.data = PaddleBuf();
  tensor_out.dtype = PaddleDType::FLOAT32;
  output_tensors.push_back(tensor_out);

  inference::Timer timer;
  for (int i = 0; i < FLAGS_repeats; i++) {
    data.reset_current_line();
    size_t count = 0;
    float time_sum = 0;
    while (data.get_next_batches(&inputs, &seq_offsets) >= 0) {
#if PRINT_INPUTS
      for (size_t i = 0; i < inputs.size(); i++) {
        LOG(INFO) << "data " << i;
        for (size_t j = 0; j < inputs[i].size(); j++) {
          LOG(INFO) << j << ": " << inputs[i][j];
        }
        for (auto j : seq_offsets[i]) {
          LOG(INFO) << "offsets: " << i << ": " << j;
        }
      }
#endif
      for (size_t j = 0; j < input_tensors.size(); j++) {
        input_tensors[j].data =
            PaddleBuf(&inputs[j][0], inputs[j].size() * sizeof(float));
        input_tensors[j].lod =
            std::vector<std::vector<size_t>>({seq_offsets[j]});
        input_tensors[j].shape =
            std::vector<int>({static_cast<int>(inputs[j].size()), 1, 1, 1});
      }
      timer.tic();
      predictor->Run(input_tensors, &output_tensors);
      float time = timer.toc();
#if COMPARE_OUTPUTS
      float* data_o = static_cast<float*>(output_tensors[0].data.data());
      LOG(INFO) << "outputs[0].data.size() = "
                << output_tensors[0].data.length() / sizeof(float);
      size_t sum = 1;
      for_each(output_tensors[0].shape.begin(), output_tensors[0].shape.end(),
               [&](int n) { sum *= n; });
      for (size_t j = 0; j < sum; ++j) {
        LOG(INFO) << "output[" << j << "]: " << data_o[j];
        compare_outputs.push_back(data_o[j]);
      }
#endif
      LOG(INFO) << "Single Time: " << time;
      count++;
      if (count > 10) {
        time_sum += timer.toc();
      }
    }
    inference::PrintTime(FLAGS_batch_size, FLAGS_repeats, 1, 0,
                         time_sum / (count - 10));
#if COMPARE_OUTPUTS
    Data data(FLAGS_truthpath, 1);
    const std::vector<std::string> truth_vals = data.get_lines();
    for (size_t j = 0; j < truth_vals.size(); j++) {
      float truth = std::atof(truth_vals[j].c_str());
      float compa = compare_outputs[j];
      float diff = std::abs(truth - compa);
      LOG(INFO) << "[DIFF " << j << " ] " << diff;
      if (diff > 0.0001) {
        LOG(FATAL) << "The result is wrong!";
      }
    }
    LOG(INFO) << "The result is correct!";
#endif
  }
}
}  // namespace paddle

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  std::vector<std::thread> threads;

  auto config = paddle::GetConfig();
  config.data_stream_id = 0;
  config.compute_stream_id = 0;
  std::unique_ptr<paddle::PaddlePredictor> predictor_master =
      paddle::CreatePaddlePredictor<paddle::contrib::AnakinConfig,
                                    paddle::PaddleEngineKind::kAnakin>(config);

  for (int i = 0; i < FLAGS_threads_num; i++) {
    threads.push_back(std::thread(paddle::single_test, predictor_master.get()));
  }
  for (auto& t : threads) {
    t.join();
  }
  return 0;
}
