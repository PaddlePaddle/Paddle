// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <glog/logging.h>
#include <gperftools/heap-profiler.h>
#include <paddle/fluid/inference/api/paddle_inference_api.h>
#include <sys/sysinfo.h>
#include <sys/types.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

DEFINE_string(model_dir, "", "model directory");
DEFINE_string(data, "", "input data path");
DEFINE_int32(repeat, 1, "repeat");
DEFINE_bool(gtest_color, false, "");
DEFINE_string(gtest_filter,"" , "");

template <typename T>
void GetValueFromStream(std::stringstream *ss, T *t) {
  (*ss) >> (*t);
}

template <>
void GetValueFromStream<std::string>(std::stringstream *ss, std::string *t) {
  *t = ss->str();
}

// Split string to vector
template <typename T>
void Split(const std::string &line, char sep, std::vector<T> *v) {
  std::stringstream ss;
  T t;
  // int count = 0;
  for (auto c : line) {
    if (c != sep) {
      ss << c;
    } else {
      GetValueFromStream<T>(&ss, &t);
      v->push_back(std::move(t));
      ss.str({});
      ss.clear();
    }
    // if (++count == 128) break;
  }

  if (!ss.str().empty()) {
    GetValueFromStream<T>(&ss, &t);
    v->push_back(std::move(t));
    ss.str({});
    ss.clear();
  }
}

template <typename T>
constexpr paddle::PaddleDType GetPaddleDType();

template <>
constexpr paddle::PaddleDType GetPaddleDType<int64_t>() {
  return paddle::PaddleDType::INT64;
}

template <>
constexpr paddle::PaddleDType GetPaddleDType<float>() {
  return paddle::PaddleDType::FLOAT32;
}

// Parse tensor from string
template <typename T>
bool ParseTensor(const std::string &field, paddle::PaddleTensor *tensor) {
  std::vector<std::string> data;
  Split(field, ':', &data);
  if (data.size() < 2) return false;

  std::string shape_str = data[0];

  std::vector<int> shape;
  Split(shape_str, ' ', &shape);

  std::string mat_str = data[1];

  std::vector<T> mat;
  Split(mat_str, ' ', &mat);

  tensor->shape = shape;
  auto size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) *
      sizeof(T);
  tensor->data.Resize(size);
  std::copy(mat.begin(), mat.end(), static_cast<T *>(tensor->data.data()));
  tensor->dtype = GetPaddleDType<T>();

  return true;
}

// Parse input tensors from string
bool ParseLine(const std::string &line,
               std::vector<paddle::PaddleTensor> *tensors) {
  std::vector<std::string> fields;
  Split(line, ';', &fields);

  if (fields.size() < 5) return false;

  tensors->clear();
  tensors->reserve(5);

  int i = 0;
  // src_id
  paddle::PaddleTensor src_id;
  ParseTensor<int64_t>(fields[i++], &src_id);
  tensors->push_back(src_id);

  // pos_id
  paddle::PaddleTensor pos_id;
  ParseTensor<int64_t>(fields[i++], &pos_id);
  tensors->push_back(pos_id);

  // segment_id
  paddle::PaddleTensor segment_id;
  ParseTensor<int64_t>(fields[i++], &segment_id);
  tensors->push_back(segment_id);

  // self_attention_bias
  paddle::PaddleTensor self_attention_bias;
  ParseTensor<float>(fields[i++], &self_attention_bias);
  tensors->push_back(self_attention_bias);

  // next_segment_index
  paddle::PaddleTensor next_segment_index;
  ParseTensor<int64_t>(fields[i++], &next_segment_index);
  tensors->push_back(next_segment_index);

  return true;
}

// Print outputs to log
void PrintOutputs(const std::vector<paddle::PaddleTensor> &outputs) {
  LOG(INFO) << "example_id\tcontradiction\tentailment\tneutral";

  for (size_t i = 0; i < outputs.front().data.length() / sizeof(float);
       i += 3) {
    LOG(INFO) << (i / 3) << "\t"
              << static_cast<float *>(outputs.front().data.data())[i] << "\t"
              << static_cast<float *>(outputs.front().data.data())[i + 1]
              << "\t"
              << static_cast<float *>(outputs.front().data.data())[i + 2];
  }
}

bool LoadInputData(std::vector<std::vector<paddle::PaddleTensor>> *inputs) {
  if (FLAGS_data.empty()) {
    LOG(ERROR) << "please set input data path";
    return false;
  }

  std::ifstream fin(FLAGS_data);
  std::string line;

  int lineno = 0;
  while (std::getline(fin, line)) {
    std::vector<paddle::PaddleTensor> feed_data;
    if (!ParseLine(line, &feed_data)) {
      LOG(ERROR) << "Parse line[" << lineno << "] error!";
    } else {
      inputs->push_back(std::move(feed_data));
    }
  }

  return true;
}

// Bert inference demo
// Options:
//     --model_dir: bert model file directory
//     --data: data path
//     --repeat: repeat num
int main(int argc, char *argv[]) {
  google::InitGoogleLogging(*argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_model_dir.empty()) {
    LOG(ERROR) << "please set model dir";
    return -1;
  }

  // paddle::NativeConfig config;
  // config.model_dir = FLAGS_model_dir;
  // config.use_gpu = false;
  paddle::AnalysisConfig config;
  config.SetModel(FLAGS_model_dir);
  config.SetCpuMathLibraryNumThreads(0);

  auto predictor = CreatePaddlePredictor(config);

  std::vector<decltype(predictor)> predictors;

  const int num_threads = 1;
  for (int i = 0; i < num_threads; i++) {
    predictors.emplace_back(predictor->Clone());
  }

  auto run_infer = [&predictors](int thread_id) {
    std::ifstream fin(FLAGS_data);
    std::string line;
    std::vector<paddle::PaddleTensor> inputs;
    std::vector<paddle::PaddleTensor> outputs;
    // std::string thread_name = "predictor" + std::to_string(thread_id);
    auto thread_name = "predictor" + std::to_string(thread_id);

    //int num_line = 0;
    HeapProfilerStart("bert");
    for (int epoch = 0; epoch < 2; ++epoch) {
      LOG(INFO) << "Thread[" << thread_id << "], Epoch[" << epoch << "]";
      while (std::getline(fin, line)) {
        inputs.clear();
        outputs.clear();
        ParseLine(line, &inputs);
        LOG(INFO) << "working";
        predictors[thread_id]->Run(inputs, &outputs);
        //if (num_line++ % 1 == 0) {
          auto str_thread_id = std::to_string(thread_id);
          HeapProfilerDump(thread_name.c_str());
        //}
      }
    }
    HeapProfilerStop();
  };

  std::vector<std::thread> threads;
  run_infer(0);
  //for (int i = 0; i < num_threads; i++) {
    //threads.emplace_back(run_infer, i);
  //}

  //for (auto &th : threads) th.join();

  // HeapProfilerStop();

  return 0;
}
