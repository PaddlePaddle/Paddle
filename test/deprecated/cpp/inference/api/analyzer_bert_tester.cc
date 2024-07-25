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

#include "paddle/common/errors.h"
#include "paddle/fluid/framework/transfer_scope_cache.h"
#include "paddle/phi/core/enforce.h"
#include "test/cpp/inference/api/tester_helper.h"

namespace paddle {
namespace inference {

using paddle::PaddleTensor;

void profile(bool use_mkldnn = false, bool use_bfloat16 = false);
std::vector<std::vector<paddle::PaddleTensor>> LoadInputData();
void CompareNativeAndAnalysisWrapper(bool use_mkldnn = false);
std::vector<paddle::PaddleTensor> ParseInputStreamToVector(
    const std::string &line);

AnalysisConfig SetConfig(bool use_mkldnn = false, bool use_bfloat16 = false);

template <typename T>
paddle::PaddleTensor ParseTensor(const std::string &field);

template <typename T>
std::vector<T> Split(const std::string &line, char separator);

template <typename T>
T GetValueFromStream(std::stringstream &ss);

template <>
std::string GetValueFromStream<std::string>(std::stringstream &ss);

TEST(Analyzer_bert, profile) {
#if !defined(_WIN32)
  setenv("NVIDIA_TF32_OVERRIDE", "0", 1);
#endif
  profile();
}

#ifdef PADDLE_WITH_DNNL
TEST(Analyzer_bert, profile_mkldnn) {
  auto use_mkldnn = true;
  profile(use_mkldnn);
}

TEST(Analyzer_bert, profile_mkldnn_bf16) {
  auto use_mkldnn = true;
  auto use_bfloat16 = true;
  profile(use_mkldnn, use_bfloat16);
}
#endif

// Check the fuse status
TEST(Analyzer_bert, fuse_statis) {
#if !defined(_WIN32)
  setenv("NVIDIA_TF32_OVERRIDE", "0", 1);
#endif
  auto cfg(SetConfig());
  int num_ops;
  auto predictor = CreatePaddlePredictor<AnalysisConfig>(cfg);
  auto fuse_statis = GetFuseStatis(
      static_cast<AnalysisPredictor *>(predictor.get()), &num_ops);
  LOG(INFO) << "num_ops: " << num_ops;
}

TEST(Analyzer_bert, compare) {
#if !defined(_WIN32)
  setenv("NVIDIA_TF32_OVERRIDE", "0", 1);
#endif
  CompareNativeAndAnalysisWrapper();
}
#ifdef PADDLE_WITH_DNNL
TEST(Analyzer_bert, compare_mkldnn) {
  auto use_mkldnn = true;
  CompareNativeAndAnalysisWrapper(use_mkldnn);
}
#endif

// Compare Deterministic result
TEST(Analyzer_bert, compare_determine) {
#if !defined(_WIN32)
  setenv("NVIDIA_TF32_OVERRIDE", "0", 1);
#endif
  auto cfg(SetConfig());

  auto inputs = LoadInputData();
  CompareDeterministic(reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
                       inputs);
}

TEST(Analyzer_bert, transfer_scope_cache) {
#if !defined(_WIN32)
  setenv("NVIDIA_TF32_OVERRIDE", "0", 1);
#endif
  auto config(SetConfig());

  std::vector<PaddleTensor> input, output;
  auto predictor = CreatePaddlePredictor<AnalysisConfig>(config);

  int threads_num = 10;
  std::vector<std::thread> threads;
  std::unordered_set<std::unordered_set<paddle::framework::Scope *> *>
      global_transfer_scope_cache;
  std::unordered_set<std::unordered_map<size_t, paddle::framework::Scope *> *>
      global_transfer_data_cache;

  std::ifstream fin(FLAGS_infer_data);
  std::string line;

  for (int i = 0; i < threads_num; i++) {
    threads.emplace_back([&]() {
      std::getline(fin, line);
      input = ParseInputStreamToVector(line);
      predictor->Run(input, &output, FLAGS_batch_size);
      global_transfer_scope_cache.insert(
          &paddle::framework::global_transfer_scope_cache());
      global_transfer_data_cache.insert(
          &paddle::framework::global_transfer_data_cache());
    });
    threads[0].join();
    threads.clear();
    std::vector<PaddleTensor>().swap(input);
  }
  // Since paddle::framework::global_transfer_scope_cache() and
  // paddle::framework::global_transfer_data_cache() are thread_local,
  // their pointer should be different among different thread id.
  PADDLE_ENFORCE_EQ(
      global_transfer_scope_cache.size(),
      threads_num,
      phi::errors::Fatal(
          "The size of scope cache is not equal to thread number."));
  PADDLE_ENFORCE_EQ(
      global_transfer_data_cache.size(),
      threads_num,
      phi::errors::Fatal(
          "The size of data cache is not equal to thread number."));
}

void profile(bool use_mkldnn, bool use_bfloat16) {
  auto config(SetConfig(use_mkldnn, use_bfloat16));
  std::vector<std::vector<PaddleTensor>> outputs;
  auto inputs = LoadInputData();
  TestPrediction(reinterpret_cast<const PaddlePredictor::Config *>(&config),
                 inputs,
                 &outputs,
                 FLAGS_num_threads);
}

std::vector<std::vector<paddle::PaddleTensor>> LoadInputData() {
  if (FLAGS_infer_data.empty()) {
    LOG(ERROR) << "please set input data path";
    PADDLE_THROW(phi::errors::NotFound("Missing input data path"));
  }

  std::ifstream fin(FLAGS_infer_data);
  std::string line;
  int sample = 0;

  std::vector<std::vector<paddle::PaddleTensor>> inputs;

  // The unit-test dataset only have 10 samples, each sample have 5 feeds.
  while (std::getline(fin, line)) {
    inputs.push_back(ParseInputStreamToVector(line));
    sample++;
    if (!FLAGS_test_all_data && sample == FLAGS_batch_size) break;
  }
  LOG(INFO) << "number of samples: " << sample;

  return inputs;
}

void CompareNativeAndAnalysisWrapper(bool use_mkldnn) {
  auto cfg(SetConfig(use_mkldnn));
  auto inputs = LoadInputData();
  CompareNativeAndAnalysis(
      reinterpret_cast<const PaddlePredictor::Config *>(&cfg), inputs);
}

std::vector<paddle::PaddleTensor> ParseInputStreamToVector(
    const std::string &line) {
  const auto fields = Split<std::string>(line, ';');

  if (fields.size() < 5) PADDLE_THROW(phi::errors::Fatal("Invalid input line"));

  std::vector<paddle::PaddleTensor> tensors;

  tensors.reserve(5);

  const std::size_t src_id = 0;
  const std::size_t pos_id = 1;
  const std::size_t segment_id = 2;
  const std::size_t self_attention_bias = 3;
  const std::size_t next_segment_index = 4;

  tensors.push_back(ParseTensor<int64_t>(fields[src_id]));
  tensors.push_back(ParseTensor<int64_t>(fields[pos_id]));
  tensors.push_back(ParseTensor<int64_t>(fields[segment_id]));
  tensors.push_back(ParseTensor<float>(fields[self_attention_bias]));
  tensors.push_back(ParseTensor<int64_t>(fields[next_segment_index]));

  return tensors;
}

AnalysisConfig SetConfig(bool use_mkldnn, bool use_bfloat16) {
  AnalysisConfig config;
  config.SetModel(FLAGS_infer_model);
  config.DisableFCPadding();

  if (use_mkldnn) {
    config.EnableMKLDNN();
  }

  if (use_bfloat16) config.EnableMkldnnBfloat16();

  return config;
}

template <typename T>
paddle::PaddleTensor ParseTensor(const std::string &field) {
  const auto data = Split<std::string>(field, ':');
  if (data.size() < 2) PADDLE_THROW(phi::errors::Fatal("Invalid data field"));

  std::string shape_str = data[0];
  const auto shape = Split<int>(shape_str, ' ');
  paddle::PaddleTensor tensor;
  tensor.shape = shape;
  auto size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) *
      sizeof(T);
  tensor.data.Resize(size);

  std::string mat_str = data[1];
  const auto mat = Split<T>(mat_str, ' ');
  std::copy(mat.cbegin(), mat.cend(), static_cast<T *>(tensor.data.data()));
  tensor.dtype = GetPaddleDType<T>();

  return tensor;
}

template <typename T>
std::vector<T> Split(const std::string &line, char separator) {
  std::vector<T> result;
  std::stringstream ss;
  for (auto c : line) {
    if (c != separator) {
      ss << c;
    } else {
      result.emplace_back(GetValueFromStream<T>(ss));
      ss.str({});
      ss.clear();
    }
  }

  auto ss_is_not_empty = !ss.str().empty();
  if (ss_is_not_empty) result.emplace_back(GetValueFromStream<T>(ss));

  return result;
}

template <typename T>
T GetValueFromStream(std::stringstream &ss) {
  T result;
  ss >> result;
  return result;
}

template <>
std::string GetValueFromStream<std::string>(std::stringstream &ss) {
  return ss.str();
}

}  // namespace inference
}  // namespace paddle
