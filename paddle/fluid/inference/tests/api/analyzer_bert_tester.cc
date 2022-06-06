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

#include "paddle/fluid/framework/transfer_scope_cache.h"
#include "paddle/fluid/inference/tests/api/tester_helper.h"

namespace paddle {
namespace inference {

using paddle::PaddleTensor;

template <typename T>
T GetValueFromStream(std::stringstream& ss) {
  T result;
  ss >> result;
  return result;
}

template <>
std::string GetValueFromStream<std::string>(std::stringstream& ss) {
  return ss.str();
}

// Split string to vector
template <typename T>
std::vector<T> Split(const std::string& line, char sep) {
  std::vector<T> result;
  std::stringstream ss;
  for (auto c : line) {
    if (c != sep) {
      ss << c;
    } else {
      result.emplace_back(GetValueFromStream<T>(ss));
      ss.str({});
      ss.clear();
    }
  }

  auto ss_is_not_empty = !ss.str().empty();
  if (ss_is_not_empty)
    result.emplace_back(GetValueFromStream<T>(ss));

  return result;
}

// Parse tensor from string
template <typename T>
paddle::PaddleTensor ParseTensor(const std::string& field) {
  const auto data = Split<std::string>(field, ':');
  if (data.size() < 2) throw "invalid field";

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
  std::copy(mat.cbegin(), mat.cend(), static_cast<T*>(tensor.data.data()));
  tensor.dtype = GetPaddleDType<T>();

  return tensor;
}

// Parse input tensors from string
std::vector<paddle::PaddleTensor> ParseLine(const std::string &line) {
  const auto fields = Split<std::string>(line, ';');

  if (fields.size() < 5) throw "invalid line";

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

std::vector<std::vector<paddle::PaddleTensor>> LoadInputData() {
  if (FLAGS_infer_data.empty()) {
    LOG(ERROR) << "please set input data path";
    throw "missing input data path";
  }

  std::ifstream fin(FLAGS_infer_data);
  std::string line;
  int sample = 0;

  std::vector<std::vector<paddle::PaddleTensor>> inputs;

  // The unit-test dataset only have 10 samples, each sample have 5 feeds.
  while (std::getline(fin, line)) {
    inputs.push_back(ParseLine(line));
    sample++;
    if (!FLAGS_test_all_data && sample == FLAGS_batch_size) break;
  }
  LOG(INFO) << "number of samples: " << sample;

  return inputs;
}

AnalysisConfig SetConfig() {
  AnalysisConfig config;
  config.SetModel(FLAGS_infer_model);
  config.DisableFCPadding();
  return config;
}

void profile(bool use_mkldnn = false) {
  auto config(SetConfig());

  if (use_mkldnn) {
    config.EnableMKLDNN();
    config.pass_builder()->AppendPass("fc_mkldnn_pass");
    config.pass_builder()->AppendPass("fc_act_mkldnn_fuse_pass");
    config.pass_builder()->AppendPass("fc_elementwise_add_mkldnn_fuse_pass");
  }

  std::vector<std::vector<PaddleTensor>> outputs;
  auto inputs = LoadInputData();
  TestPrediction(reinterpret_cast<const PaddlePredictor::Config *>(&config),
                 inputs, &outputs, FLAGS_num_threads);
}

TEST(Analyzer_bert, profile) { profile(); }
#ifdef PADDLE_WITH_MKLDNN
TEST(Analyzer_bert, profile_mkldnn) { profile(true); }
#endif

// Check the fuse status
TEST(Analyzer_bert, fuse_statis) {
  auto cfg(SetConfig());
  int num_ops;
  auto predictor = CreatePaddlePredictor<AnalysisConfig>(cfg);
  auto fuse_statis = GetFuseStatis(
      static_cast<AnalysisPredictor *>(predictor.get()), &num_ops);
  LOG(INFO) << "num_ops: " << num_ops;
}

// Compare result of NativeConfig and AnalysisConfig
void compare(bool use_mkldnn = false) {
  auto cfg(SetConfig());
  if (use_mkldnn) {
    cfg.EnableMKLDNN();
  }

  auto inputs = LoadInputData();
  CompareNativeAndAnalysis(
      reinterpret_cast<const PaddlePredictor::Config *>(&cfg), inputs);
}

TEST(Analyzer_bert, compare) { compare(); }
#ifdef PADDLE_WITH_MKLDNN
TEST(Analyzer_bert, compare_mkldnn) { compare(true /* use_mkldnn */); }
#endif

// Compare Deterministic result
TEST(Analyzer_bert, compare_determine) {
  auto cfg(SetConfig());

  auto inputs = LoadInputData();
  CompareDeterministic(reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
                       inputs);
}

TEST(Analyzer_bert, transfer_scope_cache) {
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
    threads.emplace_back([&, i]() {
      std::getline(fin, line);
      input = ParseLine(line);
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
      global_transfer_scope_cache.size(), threads_num,
      paddle::platform::errors::Fatal(
          "The size of scope cache is not equal to thread number."));
  PADDLE_ENFORCE_EQ(
      global_transfer_data_cache.size(), threads_num,
      paddle::platform::errors::Fatal(
          "The size of data cache is not equal to thread number."));
}

}  // namespace inference
}  // namespace paddle
