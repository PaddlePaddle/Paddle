// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/api/paddle_inference_api.h"
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

bool LoadInputData(std::vector<std::vector<paddle::PaddleTensor>> *inputs,
                   int batch_size = 1) {
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
    if (!FLAGS_test_all_data && sample == batch_size) break;
  }
  LOG(INFO) << "number of samples: " << sample;
  return true;
}

// Compare results
TEST(Ernie_gpu_fp16_no_ir, compare_results) {
  AnalysisConfig config;
  config.SetModel(FLAGS_infer_model);
  config.EnableUseGpu(512, 0, paddle_infer::PrecisionType::kHalf);
  config.SwitchIrOptim(false);

  auto predictor = CreatePaddlePredictor(config);

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  LoadInputData(&input_slots_all);

  std::ifstream fin(FLAGS_refer_result);
  std::string line;
  std::vector<float> ref;

  while (std::getline(fin, line)) {
    Split(line, ' ', &ref);
  }

  std::vector<PaddleTensor> outputs;
  for (size_t i = 0; i < input_slots_all.size(); i++) {
    outputs.clear();
    predictor->Run(input_slots_all[i], &outputs);

    auto output = outputs.front();
    size_t outputs_size = 1;
    for (auto dim : output.shape) {
      outputs_size *= dim;
    }
    float *result = reinterpret_cast<float *>(output.data.data());
    for (size_t j = 0; j < outputs_size; ++j) {
      EXPECT_NEAR(ref[i * outputs_size + j], result[j], 5e-2);
    }
  }
}

// Compare results
TEST(Ernie_gpu_fp16_with_ir, compare_results) {
  AnalysisConfig config;
  config.SetModel(FLAGS_infer_model);
  config.EnableUseGpu(512, 0, paddle_infer::PrecisionType::kHalf);
  config.SwitchIrOptim(true);
  // The fc_fuse_pass has diff, which will be repaired later.
  config.pass_builder()->DeletePass("fc_fuse_pass");
  // There is a problem with the model itself, which has nothing to do with
  // constant_folding_pass.
  config.pass_builder()->DeletePass("constant_folding_pass");

  auto predictor = CreatePaddlePredictor(config);

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  LoadInputData(&input_slots_all);

  std::ifstream fin(FLAGS_refer_result);
  std::string line;
  std::vector<float> ref;

  while (std::getline(fin, line)) {
    Split(line, ' ', &ref);
  }

  std::vector<PaddleTensor> outputs;
  for (size_t i = 0; i < input_slots_all.size(); i++) {
    outputs.clear();
    predictor->Run(input_slots_all[i], &outputs);

    auto output = outputs.front();
    size_t outputs_size = 1;
    for (auto dim : output.shape) {
      outputs_size *= dim;
    }
    float *result = reinterpret_cast<float *>(output.data.data());
    for (size_t j = 0; j < outputs_size; ++j) {
      EXPECT_NEAR(ref[i * outputs_size + j], result[j], 5e-2);
    }
  }
}

// Compare results
TEST(Ernie_gpu_bf16_no_ir, compare_results) {
  AnalysisConfig config;
  config.SetModel(FLAGS_infer_model);
  config.EnableUseGpu(512, 0, paddle_infer::PrecisionType::kBf16);
  config.SwitchIrOptim(false);

  auto predictor = CreatePaddlePredictor(config);

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  LoadInputData(&input_slots_all);

  std::ifstream fin(FLAGS_refer_result);
  std::string line;
  std::vector<float> ref;

  while (std::getline(fin, line)) {
    Split(line, ' ', &ref);
  }

  std::vector<PaddleTensor> outputs;
  for (size_t i = 0; i < input_slots_all.size(); i++) {
    outputs.clear();
    predictor->Run(input_slots_all[i], &outputs);

    auto output = outputs.front();
    size_t outputs_size = 1;
    for (auto dim : output.shape) {
      outputs_size *= dim;
    }
    float *result = reinterpret_cast<float *>(output.data.data());
    for (size_t j = 0; j < outputs_size; ++j) {
      EXPECT_NEAR(ref[i * outputs_size + j], result[j], 7e-2);
    }
  }
}

// Compare results
TEST(Ernie_gpu_bf16_with_ir, compare_results) {
  AnalysisConfig config;
  config.SetModel(FLAGS_infer_model);
  config.EnableUseGpu(512, 0, paddle_infer::PrecisionType::kBf16);
  config.SwitchIrOptim(true);
  // The fc_fuse_pass has diff, which will be repaired later.
  config.pass_builder()->DeletePass("fc_fuse_pass");
  // There is a problem with the model itself, which has nothing to do with
  // constant_folding_pass.
  config.pass_builder()->DeletePass("constant_folding_pass");

  auto predictor = CreatePaddlePredictor(config);

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  LoadInputData(&input_slots_all);

  std::ifstream fin(FLAGS_refer_result);
  std::string line;
  std::vector<float> ref;

  while (std::getline(fin, line)) {
    Split(line, ' ', &ref);
  }

  std::vector<PaddleTensor> outputs;
  for (size_t i = 0; i < input_slots_all.size(); i++) {
    outputs.clear();
    predictor->Run(input_slots_all[i], &outputs);

    auto output = outputs.front();
    size_t outputs_size = 1;
    for (auto dim : output.shape) {
      outputs_size *= dim;
    }
    float *result = reinterpret_cast<float *>(output.data.data());
    for (size_t j = 0; j < outputs_size; ++j) {
      EXPECT_NEAR(ref[i * outputs_size + j], result[j], 7e-2);
    }
  }
}

}  // namespace inference
}  // namespace paddle
