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
#include <glog/raw_logging.h>
#include <gtest/gtest.h>
#include <pthread.h>
#include <atomic>
#include <fstream>
#include <iostream>
#include <numeric>
#include <numeric>
#include <thread>

#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/api/paddle_inference_helper.h"

DEFINE_string(modeldir, "", "Directory of the inference model.");
DEFINE_string(datapath, "", "Path of the dataset.");
DEFINE_int32(batch_size, 1, "batch size");
DEFINE_int32(thread_batch_size, 1, "batch size for multi-thread");
DEFINE_int32(num_threads, 1, "thread number");

using namespace paddle;

float random(float low, float high) {
  static std::random_device rd;
  static std::mt19937 mt(rd());
  std::uniform_real_distribution<double> dist(low, high);
  return dist(mt);
}

void split(const std::string &str, char sep, std::vector<std::string> *pieces) {
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

void split_to_float(const std::string &str, char sep, std::vector<float> *fs) {
  std::vector<std::string> pieces;
  split(str, sep, &pieces);
  std::transform(pieces.begin(), pieces.end(), std::back_inserter(*fs),
                 [](const std::string &v) { return std::stof(v); });
}

template <typename T>
std::string to_string(const std::vector<T> &vec) {
  std::stringstream ss;
  for (const auto &c : vec) {
    ss << c << " ";
  }
  return ss.str();
}

template <>
std::string to_string<std::vector<float>>(
    const std::vector<std::vector<float>> &vec) {
  std::stringstream ss;
  for (const auto &piece : vec) {
    ss << to_string(piece) << "\n";
  }
  return ss.str();
}

template <>
std::string to_string<std::vector<std::vector<float>>>(
    const std::vector<std::vector<std::vector<float>>> &vec) {
  std::stringstream ss;
  for (const auto &line : vec) {
    for (const auto &rcd : line) {
      ss << to_string(rcd) << ";\t";
    }
    ss << '\n';
  }
  return ss.str();
}

// clang-format off
void TensorAssignData(PaddleTensor *tensor, const std::vector<std::vector<float>> &data) {
  // Assign buffer
  int dim = std::accumulate(tensor->shape.begin(), tensor->shape.end(), 1, [](int a, int b) { return a * b; });
  tensor->data.Resize(sizeof(float) * dim);
  int c = 0;
  for (const auto &f : data) {
    for (float v : f) { static_cast<float *>(tensor->data.data())[c++] = v; }
  }
}
// clang-format on

std::string DescribeTensor(const PaddleTensor &tensor) {
  std::stringstream os;
  os << "Tensor [" << tensor.name << "]\n";
  os << " - type: ";
  switch (tensor.dtype) {
    case PaddleDType::FLOAT32:
      os << "float32";
      break;
    case PaddleDType::INT64:
      os << "int64";
      break;
    default:
      os << "unset";
  }
  os << '\n';

  os << " - shape: " << to_string(tensor.shape) << '\n';
  os << " - lod: ";
  for (auto &l : tensor.lod) {
    os << to_string(l) << "; ";
  }
  os << "\n";
  os << " - data: ";

  // clang-format off
  int dim = std::accumulate(tensor.shape.begin(),
                            tensor.shape.end(),
                            1,
                            [](int a, int b) { return a * b; });  // clang-format on
  for (size_t i = 0; i < dim; i++) {
    os << static_cast<float *>(tensor.data.data())[i] << " ";
  }
  os << '\n';
  return os.str();
}

struct DataRecord {
  std::vector<std::vector<std::vector<float>>> link_step_data_all;
  std::vector<std::vector<float>> week_data_all, minute_data_all;
  std::vector<size_t> lod1, lod2, lod3;
  std::vector<std::vector<float>> rnn_link_data, rnn_week_datas,
      rnn_minute_datas;

  size_t batch_iter{0};
  size_t batch_size{1};

  DataRecord() = default;
  DataRecord(const std::string &path, int batch_size = 1)
      : batch_size(batch_size) {
    Load(path);
  }

  DataRecord NextBatch() {
    DataRecord data;
    size_t batch_end = batch_iter + batch_size;

    // NOTE skip the final batch, if no enough data is provided.
    if (batch_end <= link_step_data_all.size()) {
      data.link_step_data_all.assign(link_step_data_all.begin() + batch_iter,
                                     link_step_data_all.begin() + batch_end);
      data.week_data_all.assign(week_data_all.begin() + batch_iter,
                                week_data_all.begin() + batch_end);
      data.minute_data_all.assign(minute_data_all.begin() + batch_iter,
                                  minute_data_all.begin() + batch_end);

      // Prepare LoDs
      data.lod1.emplace_back(0);
      data.lod2.emplace_back(0);
      data.lod3.emplace_back(0);

      CHECK(!data.link_step_data_all.empty()) << "empty";
      CHECK(!data.week_data_all.empty());
      CHECK(!data.minute_data_all.empty());
      CHECK_EQ(data.link_step_data_all.size(), data.week_data_all.size());
      CHECK_EQ(data.minute_data_all.size(), data.link_step_data_all.size());

      for (size_t j = 0; j < data.link_step_data_all.size(); j++) {
        for (const auto &d : data.link_step_data_all[j]) {
          data.rnn_link_data.push_back(d);
        }
        data.rnn_week_datas.push_back(data.week_data_all[j]);
        data.rnn_minute_datas.push_back(data.minute_data_all[j]);
        // calculate lod
        data.lod1.push_back(data.lod1.back() +
                            data.link_step_data_all[j].size());
        data.lod3.push_back(data.lod3.back() + 1);
        for (size_t i = 1; i < data.link_step_data_all[j].size() + 1; i++) {
          data.lod2.push_back(data.lod2.back() +
                              data.link_step_data_all[j].size());
        }
      }
    }

    batch_iter += batch_size;
    return data;
  }

  void Load(const std::string &path) {
    std::ifstream file(FLAGS_datapath);
    std::string line;

    int num_lines = 0;
    while (std::getline(file, line)) {
      num_lines++;
      std::vector<std::string> data;
      split(line, ':', &data);

      std::vector<std::vector<float>> link_step_data;
      std::vector<std::string> link_datas;
      split(data[0], '|', &link_datas);

      for (auto &step_data : link_datas) {
        std::vector<float> tmp;
        split_to_float(step_data, ',', &tmp);
        link_step_data.emplace_back(tmp);
      }

      // load week data
      std::vector<float> week_data;
      split_to_float(data[2], ',', &week_data);

      // load minute data
      std::vector<float> minute_data;
      split_to_float(data[1], ',', &minute_data);

      link_step_data_all.emplace_back(std::move(link_step_data));
      week_data_all.emplace_back(std::move(week_data));
      minute_data_all.emplace_back(std::move(minute_data));
    }
  }
};

void PrepareInputs(std::vector<PaddleTensor> *input_slots, DataRecord *data,
                   int batch_size) {
  // DataRecord data(FLAGS_datapath, batch_size);

  PaddleTensor lod_attention_tensor, init_zero_tensor, lod_tensor_tensor,
      week_tensor, minute_tensor;
  lod_attention_tensor.name = "lod_attention";
  init_zero_tensor.name = "init_zero";
  lod_tensor_tensor.name = "lod_tensor";
  week_tensor.name = "week";
  minute_tensor.name = "minute";

  auto one_batch = data->NextBatch();

  // clang-format off
  std::vector<int> rnn_link_data_shape
      ({static_cast<int>(one_batch.rnn_link_data.size()), static_cast<int>(one_batch.rnn_link_data.front().size())});
  //LOG(INFO) << "set 1";
  lod_attention_tensor.shape.assign({1, 2});
  lod_attention_tensor.lod.assign({one_batch.lod1, one_batch.lod2});
  //LOG(INFO) << "set 1";
  init_zero_tensor.shape.assign({batch_size, 15});
  init_zero_tensor.lod.assign({one_batch.lod3});
  //LOG(INFO) << "set 1";
  lod_tensor_tensor.shape = rnn_link_data_shape;
  lod_tensor_tensor.lod.assign({one_batch.lod1});
  //LOG(INFO) << "set 1";
  week_tensor.shape.assign({(int) one_batch.rnn_week_datas.size(), (int) one_batch.rnn_week_datas.front().size()});
  week_tensor.lod.assign({one_batch.lod3});
  //LOG(INFO) << "set 1";
  minute_tensor.shape.assign({(int) one_batch.rnn_minute_datas.size(),
                              (int) one_batch.rnn_minute_datas.front().size()});
  minute_tensor.lod.assign({one_batch.lod3});

  // assign data
  TensorAssignData(&lod_attention_tensor, std::vector<std::vector<float>>({{0, 0}}));
  std::vector<float> tmp_zeros(batch_size * 15, 0.);
  TensorAssignData(&init_zero_tensor, {tmp_zeros});
  TensorAssignData(&lod_tensor_tensor, one_batch.rnn_link_data);
  TensorAssignData(&week_tensor, one_batch.rnn_week_datas);
  TensorAssignData(&minute_tensor, one_batch.rnn_minute_datas);
  // clang-format on

  input_slots->assign({lod_tensor_tensor, lod_attention_tensor,
                       init_zero_tensor, init_zero_tensor, week_tensor,
                       minute_tensor});

  for (auto &tensor : *input_slots) {
    tensor.dtype = PaddleDType::FLOAT32;
    // LOG(INFO) << DescribeTensor(tensor);
  }
}

void Main1(int batch_size) {
  NativeConfig config;
  config.prog_file = FLAGS_modeldir + "/__model__";
  config.param_file = FLAGS_modeldir + "/param";
  config.use_gpu = false;
  config.device = 0;

  auto predictor =
      CreatePaddlePredictor<NativeConfig, PaddleEngineKind::kNative>(config);

  std::vector<PaddleTensor> input_slots;
  DataRecord data(FLAGS_datapath, batch_size);

  // Run multiple time to cancel the memory malloc or initialization of the
  // first time.
  const int times = 10;
  double whole_time = 0.;
  PrepareInputs(&input_slots, &data, batch_size);
  for (int i = 0; i < times; i++) {
    std::vector<PaddleTensor> outputs;

    Timer timer;
    timer.tic();
    predictor->Run(input_slots, &outputs);
    whole_time += timer.toc();
  }
  LOG(INFO) << "time: " << whole_time / times;
}

void MainMultiThread(int batch_size) {
  std::vector<std::thread> threads;

  double whole_time{0.};

  NativeConfig config;
  config.prog_file = FLAGS_modeldir + "/__model__";
  config.param_file = FLAGS_modeldir + "/param";
  config.use_gpu = false;
  config.device = 0;

  DataRecord data(FLAGS_datapath, batch_size);
  std::vector<PaddleTensor> input_slots;
  PrepareInputs(&input_slots, &data, batch_size);

  for (int i = 0; i < FLAGS_num_threads; i++) {
    const int times = 10;

    threads.emplace_back([&, i] {
      auto predictor =
          CreatePaddlePredictor<NativeConfig, PaddleEngineKind::kNative>(
              config);

      std::vector<PaddleTensor> outputs;

      Timer timer;
      timer.tic();
      for (int t = 0; t < times; t++) {
        predictor->Run(input_slots, &outputs);
      }
      RAW_LOG_INFO("thread #%d: %f ms", i, timer.toc() / times);
      whole_time += timer.toc() / times;
    });

    // Set thread affine
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(i, &cpuset);
    int rc = pthread_setaffinity_np(threads[i].native_handle(),
                                    sizeof(cpu_set_t), &cpuset);
    CHECK_EQ(rc, 0) << "set thread " << i << " affine failed";
  }

  for (auto &t : threads) {
    t.join();
  }

  RAW_LOG_INFO("average time: %f", whole_time / FLAGS_num_threads);
}

TEST(ditu, main) {
  Main1(FLAGS_batch_size);
  MainMultiThread(FLAGS_batch_size);
}
