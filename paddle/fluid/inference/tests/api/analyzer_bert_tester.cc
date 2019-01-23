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
#include <glog/logging.h>
#include <chrono>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include "paddle/fluid/inference/api/paddle_inference_api.h"

DEFINE_int32(repeat, 1, "repeat");

namespace paddle {
namespace inference {

using paddle::PaddleTensor;
using paddle::contrib::AnalysisConfig;

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
  for (auto c : line) {
    if (c != sep) {
      ss << c;
    } else {
      GetValueFromStream<T>(&ss, &t);
      v->push_back(std::move(t));
      ss.str({});
      ss.clear();
    }
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

  for (size_t i = 0; i < outputs.front().data.length(); i += 3) {
    LOG(INFO) << (i / 3) << "\t"
              << static_cast<float *>(outputs.front().data.data())[i] << "\t"
              << static_cast<float *>(outputs.front().data.data())[i + 1]
              << "\t"
              << static_cast<float *>(outputs.front().data.data())[i + 2];
  }
}

bool LoadInputData(std::vector<std::vector<paddle::PaddleTensor>> *inputs) {
  if (FLAGS_infer_data.empty()) {
    LOG(ERROR) << "please set input data path";
    return false;
  }

  std::ifstream fin(FLAGS_infer_data);
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

void SetConfig(contrib::AnalysisConfig *config) {
  config->SetModel(FLAGS_infer_model);
}

void profile(bool use_mkldnn = false) {
  contrib::AnalysisConfig config;
  SetConfig(&config);

  if (use_mkldnn) {
    config.EnableMKLDNN();
  }

  std::vector<PaddleTensor> outputs;
  std::vector<std::vector<PaddleTensor>> inputs;
  LoadInputData(&inputs);
  TestPrediction(reinterpret_cast<const PaddlePredictor::Config *>(&config),
                 inputs, &outputs, FLAGS_num_threads);
}

void compare(bool use_mkldnn = false) {
  AnalysisConfig config;
  SetConfig(&config);

  std::vector<std::vector<PaddleTensor>> inputs;
  LoadInputData(&inputs);
  CompareNativeAndAnalysis(
      reinterpret_cast<const PaddlePredictor::Config *>(&config), inputs);
}

TEST(Analyzer_bert, profile) { profile(); }
#ifdef PADDLE_WITH_MKLDNN
TEST(Analyzer_bert, profile_mkldnn) { profile(true); }
#endif
}  // namespace inference
}  // namespace paddle
