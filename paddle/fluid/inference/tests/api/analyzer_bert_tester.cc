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

bool LoadInputData(std::vector<std::vector<paddle::PaddleTensor>> *inputs) {
  if (FLAGS_infer_data.empty()) {
    LOG(ERROR) << "please set input data path";
    return false;
  }

  std::ifstream fin(FLAGS_infer_data);
  std::string line;
  int sample = 0;

  // The unit-test dataset only have 10 samples, each sample have 5 feeds.
  while (std::getline(fin, line)) {
    std::vector<paddle::PaddleTensor> feed_data;
    ParseLine(line, &feed_data);
    inputs->push_back(std::move(feed_data));
    sample++;
    if (!FLAGS_test_all_data && sample == FLAGS_batch_size) break;
  }
  LOG(INFO) << "number of samples: " << sample;

  return true;
}

void SetConfig(AnalysisConfig *config) {
  config->SetModel(FLAGS_infer_model);
  config->DisableFCPadding();
}

void profile(bool use_mkldnn = false) {
  AnalysisConfig config;
  SetConfig(&config);

  if (use_mkldnn) {
    config.EnableMKLDNN();
    config.pass_builder()->AppendPass("fc_mkldnn_pass");
    config.pass_builder()->AppendPass("fc_act_mkldnn_fuse_pass");
    config.pass_builder()->AppendPass("fc_elementwise_add_mkldnn_fuse_pass");
  }

  std::vector<std::vector<PaddleTensor>> outputs;
  std::vector<std::vector<PaddleTensor>> inputs;
  LoadInputData(&inputs);
  TestPrediction(reinterpret_cast<const PaddlePredictor::Config *>(&config),
                 inputs, &outputs, FLAGS_num_threads);
}

TEST(Analyzer_bert, profile) { profile(); }
#ifdef PADDLE_WITH_MKLDNN
TEST(Analyzer_bert, profile_mkldnn) { profile(true); }
#endif

// Check the fuse status
TEST(Analyzer_bert, fuse_statis) {
  AnalysisConfig cfg;
  SetConfig(&cfg);
  int num_ops;
  auto predictor = CreatePaddlePredictor<AnalysisConfig>(cfg);
  auto fuse_statis = GetFuseStatis(
      static_cast<AnalysisPredictor *>(predictor.get()), &num_ops);
  LOG(INFO) << "num_ops: " << num_ops;
}

// Compare result of NativeConfig and AnalysisConfig
void compare(bool use_mkldnn = false) {
  AnalysisConfig cfg;
  SetConfig(&cfg);
  if (use_mkldnn) {
    cfg.EnableMKLDNN();
  }

  std::vector<std::vector<PaddleTensor>> inputs;
  LoadInputData(&inputs);
  CompareNativeAndAnalysis(
      reinterpret_cast<const PaddlePredictor::Config *>(&cfg), inputs);
}

TEST(Analyzer_bert, compare) { compare(); }
#ifdef PADDLE_WITH_MKLDNN
TEST(Analyzer_bert, compare_mkldnn) { compare(true /* use_mkldnn */); }
#endif

// Compare Deterministic result
TEST(Analyzer_bert, compare_determine) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  std::vector<std::vector<PaddleTensor>> inputs;
  LoadInputData(&inputs);
  CompareDeterministic(reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
                       inputs);
}

TEST(Analyzer_bert, transfer_scope_cache) {
  AnalysisConfig config;
  SetConfig(&config);

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
      ParseLine(line, &input);
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
