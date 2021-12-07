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

  tensors->clear();
  tensors->reserve(4);

  int i = 0;
  auto input_name = FLAGS_ernie_large ? "eval_placeholder_" : "placeholder_";
  for (; i < 3; i++) {
    paddle::PaddleTensor temp;
    ParseTensor<int64_t>(fields[i], &temp);
    temp.name = input_name + std::to_string(i);
    tensors->push_back(temp);
  }

  // input_mask
  paddle::PaddleTensor input_mask;
  ParseTensor<float>(fields[i], &input_mask);
  input_mask.name = input_name + std::to_string(i);
  tensors->push_back(input_mask);

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

void SetConfig(AnalysisConfig *cfg, bool use_mkldnn = false,
               bool use_gpu = false) {
  cfg->SetModel(FLAGS_infer_model);
  if (use_mkldnn) {
    cfg->EnableMKLDNN();
  }
  if (use_gpu) {
    cfg->EnableUseGpu(100, 0);
  } else {
    cfg->DisableGpu();
  }
  cfg->SwitchSpecifyInputNames();
  cfg->SwitchIrOptim();
  cfg->SetCpuMathLibraryNumThreads(FLAGS_cpu_num_threads);
}

void profile(bool use_mkldnn = false, bool use_gpu = false) {
  AnalysisConfig config;
  SetConfig(&config, use_mkldnn, use_gpu);

  std::vector<std::vector<PaddleTensor>> outputs;
  std::vector<std::vector<PaddleTensor>> inputs;
  LoadInputData(&inputs);
  TestPrediction(reinterpret_cast<const PaddlePredictor::Config *>(&config),
                 inputs, &outputs, FLAGS_num_threads);
}

TEST(Analyzer_ernie, profile) { profile(); }
#ifdef PADDLE_WITH_MKLDNN
TEST(Analyzer_ernie, profile_mkldnn) { profile(true, false); }
#endif

// Check the model by gpu
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(Analyzer_ernie, profile_gpu) { profile(false, true); }
#endif

// Check the fuse status
TEST(Analyzer_Ernie, fuse_statis) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  int num_ops;
  auto predictor = CreatePaddlePredictor<AnalysisConfig>(cfg);
  auto fuse_statis = GetFuseStatis(
      static_cast<AnalysisPredictor *>(predictor.get()), &num_ops);
  ASSERT_TRUE(fuse_statis.count("fc_fuse"));
  LOG(INFO) << "num_ops: " << num_ops;
  if (FLAGS_ernie_large) {
    ASSERT_EQ(fuse_statis.at("fc_fuse"), 146);
    EXPECT_EQ(num_ops, 859);
  } else {
    ASSERT_EQ(fuse_statis.at("fc_fuse"), 74);
    EXPECT_EQ(num_ops, 295);
  }
}

// Compare result of NativeConfig and AnalysisConfig
void compare(bool use_mkldnn = false) {
  AnalysisConfig cfg;
  SetConfig(&cfg, use_mkldnn, false);

  std::vector<std::vector<PaddleTensor>> inputs;
  LoadInputData(&inputs);
  CompareNativeAndAnalysis(
      reinterpret_cast<const PaddlePredictor::Config *>(&cfg), inputs);
}

TEST(Analyzer_ernie, compare) { compare(); }
#ifdef PADDLE_WITH_MKLDNN
TEST(Analyzer_ernie, compare_mkldnn) { compare(true /* use_mkldnn */); }
#endif

// Compare Deterministic result
TEST(Analyzer_Ernie, compare_determine) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  LoadInputData(&input_slots_all);
  CompareDeterministic(reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
                       input_slots_all);
}

// Compare results
TEST(Analyzer_Ernie, compare_results) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  LoadInputData(&input_slots_all);

  std::ifstream fin(FLAGS_refer_result);
  std::string line;
  std::vector<float> ref;

  while (std::getline(fin, line)) {
    Split(line, ' ', &ref);
  }

  auto predictor = CreateTestPredictor(
      reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
      FLAGS_use_analysis);

  std::vector<PaddleTensor> outputs;
  for (size_t i = 0; i < input_slots_all.size(); i++) {
    outputs.clear();
    predictor->Run(input_slots_all[i], &outputs);
    auto outputs_size = outputs.front().data.length() / (sizeof(float));
    for (size_t j = 0; j < outputs_size; ++j) {
      EXPECT_NEAR(ref[i * outputs_size + j],
                  static_cast<float *>(outputs[0].data.data())[j],
                  FLAGS_accuracy);
    }
  }
}

}  // namespace inference
}  // namespace paddle
