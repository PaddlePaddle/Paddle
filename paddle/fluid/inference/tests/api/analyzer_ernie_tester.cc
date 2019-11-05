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
  // src_ids
  paddle::PaddleTensor src_ids;
  ParseTensor<int64_t>(fields[i++], &src_ids);
  src_ids.name = "placeholder_0";
  tensors->push_back(src_ids);

  // pos_ids
  paddle::PaddleTensor pos_ids;
  ParseTensor<int64_t>(fields[i++], &pos_ids);
  pos_ids.name = "placeholder_1";
  tensors->push_back(pos_ids);

  // sent_ids
  paddle::PaddleTensor sent_ids;
  ParseTensor<int64_t>(fields[i++], &sent_ids);
  sent_ids.name = "placeholder_2";
  tensors->push_back(sent_ids);

  // input_mask
  paddle::PaddleTensor input_mask;
  ParseTensor<float>(fields[i++], &input_mask);
  input_mask.name = "placeholder_3";
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

  int lineno = 0;
  while (std::getline(fin, line)) {
    std::vector<paddle::PaddleTensor> feed_data;
    if (!ParseLine(line, &feed_data)) {
      LOG(ERROR) << "Parse line[" << lineno << "] error!";
    } else {
      inputs->push_back(std::move(feed_data));
    }
    lineno++;
  }

  LOG(INFO) << "Load " << lineno << " samples from " << FLAGS_infer_data;
  return true;
}

void SetConfig(AnalysisConfig *cfg) {
  cfg->SetModel(FLAGS_infer_model);
  cfg->DisableGpu();
  cfg->SwitchSpecifyInputNames();
  cfg->SwitchIrOptim();
  cfg->SetCpuMathLibraryNumThreads(FLAGS_paddle_num_threads);
  if (FLAGS_zero_copy) {
    cfg->SwitchUseFeedFetchOps(false);
  }
}

// Check the fuse status
TEST(Analyzer_Ernie, fuse_statis) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  int num_ops;
  auto predictor = CreatePaddlePredictor<AnalysisConfig>(cfg);
  auto fuse_statis = GetFuseStatis(
      static_cast<AnalysisPredictor *>(predictor.get()), &num_ops);
}

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
TEST(Analyzer_Ernie, compare) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  LoadInputData(&input_slots_all);

  std::vector<float> ref = {
      2.2881,    -0.919723, -1.63951,  -1.42595, 2.1902,    -0.699561,
      1.20926,   -1.62216,  0.202339,  -0.51834, -0.255833, 0.403484,
      -1.61329,  2.44492,   -0.621071, 1.93191,  -2.43418,  0.252749,
      3.9044,    -2.59725,  -0.988252, -1.72084, 2.54846,   -0.660083,
      -0.907154, -0.903408, 1.83843,   0.94278,  -2.09705,  0.819825};

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
                  static_cast<float *>(outputs[0].data.data())[j], 1e-5);
    }
  }
}

}  // namespace inference
}  // namespace paddle
