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

#include "test/cpp/inference/api/tester_helper.h"

namespace paddle {
namespace inference {

struct DataRecord {
  std::vector<std::vector<int64_t>> query_basic, query_phrase, title_basic,
      title_phrase;
  std::vector<size_t> lod1, lod2, lod3, lod4;
  size_t batch_iter{0}, batch_size{1}, num_samples;  // total number of samples
  DataRecord() = default;
  explicit DataRecord(const std::string &path, int batch_size = 1)
      : batch_size(batch_size) {
    Load(path);
  }
  DataRecord NextBatch() {
    DataRecord data;
    size_t batch_end = batch_iter + batch_size;
    // NOTE skip the final batch, if no enough data is provided.
    if (batch_end <= query_basic.size()) {
      GetInputPerBatch(
          query_basic, &data.query_basic, &data.lod1, batch_iter, batch_end);
      GetInputPerBatch(
          query_phrase, &data.query_phrase, &data.lod2, batch_iter, batch_end);
      GetInputPerBatch(
          title_basic, &data.title_basic, &data.lod3, batch_iter, batch_end);
      GetInputPerBatch(
          title_phrase, &data.title_phrase, &data.lod4, batch_iter, batch_end);
    }
    batch_iter += batch_size;
    return data;
  }
  void Load(const std::string &path) {
    std::ifstream file(path);
    std::string line;
    int num_lines = 0;
    while (std::getline(file, line)) {
      std::vector<std::string> data;
      split(line, ';', &data);
      // load query data
      std::vector<int64_t> query_basic_data;
      split_to_int64(data[1], ' ', &query_basic_data);
      std::vector<int64_t> query_phrase_data;
      split_to_int64(data[2], ' ', &query_phrase_data);
      // load title data
      std::vector<int64_t> title_basic_data;
      split_to_int64(data[3], ' ', &title_basic_data);
      std::vector<int64_t> title_phrase_data;
      split_to_int64(data[4], ' ', &title_phrase_data);
      // filter the empty data
      bool flag =
          data[1].size() && data[2].size() && data[3].size() && data[4].size();
      if (flag) {
        query_basic.push_back(std::move(query_basic_data));
        query_phrase.push_back(std::move(query_phrase_data));
        title_basic.push_back(std::move(title_basic_data));
        title_phrase.push_back(std::move(title_phrase_data));
        num_lines++;
      }
    }
    num_samples = num_lines;
  }
};

void PrepareInputs(std::vector<PaddleTensor> *input_slots,
                   DataRecord *data,
                   int batch_size) {
  PaddleTensor query_basic_tensor, query_phrase_tensor, title_basic_tensor,
      title_phrase_tensor;
  query_basic_tensor.name = "query_basic";
  query_phrase_tensor.name = "query_phrase";
  title_basic_tensor.name = "pos_title_basic";
  title_phrase_tensor.name = "pos_title_phrase";
  auto one_batch = data->NextBatch();
  // assign data
  TensorAssignData<int64_t>(
      &query_basic_tensor, one_batch.query_basic, one_batch.lod1);
  TensorAssignData<int64_t>(
      &query_phrase_tensor, one_batch.query_phrase, one_batch.lod2);
  TensorAssignData<int64_t>(
      &title_basic_tensor, one_batch.title_basic, one_batch.lod3);
  TensorAssignData<int64_t>(
      &title_phrase_tensor, one_batch.title_phrase, one_batch.lod4);
  // Set inputs.
  input_slots->assign({query_basic_tensor,
                       query_phrase_tensor,
                       title_basic_tensor,
                       title_phrase_tensor});
  for (auto &tensor : *input_slots) {
    tensor.dtype = PaddleDType::INT64;
  }
}

void SetConfig(AnalysisConfig *cfg) {
  cfg->SetModel(FLAGS_infer_model);
  cfg->DisableGpu();
  cfg->SwitchSpecifyInputNames();
  cfg->SwitchIrOptim();
  cfg->SetCpuMathLibraryNumThreads(FLAGS_cpu_num_threads);
}

void SetInput(std::vector<std::vector<PaddleTensor>> *inputs) {
  DataRecord data(FLAGS_infer_data, FLAGS_batch_size);
  std::vector<PaddleTensor> input_slots;
  int epoch = FLAGS_test_all_data ? data.num_samples / FLAGS_batch_size : 1;
  LOG(INFO) << "number of samples: " << epoch * FLAGS_batch_size;
  for (int bid = 0; bid < epoch; ++bid) {
    PrepareInputs(&input_slots, &data, FLAGS_batch_size);
    (*inputs).emplace_back(input_slots);
  }
}

// Easy for profiling independently.
TEST(Analyzer_Pyramid_DNN, profile) {
  AnalysisConfig cfg;
  SetConfig(&cfg);
  std::vector<std::vector<PaddleTensor>> outputs;

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  TestPrediction(reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
                 input_slots_all,
                 &outputs,
                 FLAGS_num_threads);

  if (FLAGS_num_threads == 1 && !FLAGS_test_all_data && !FLAGS_zero_copy) {
    PADDLE_ENFORCE_GT(
        outputs.size(),
        0,
        common::errors::Fatal("The size of output should be greater than 0."));
    auto output = outputs.back();
    PADDLE_ENFORCE_EQ(
        output.size(),
        1UL,
        common::errors::Fatal("The size of output should be equal to 1."));
    size_t size = GetSize(output[0]);
    PADDLE_ENFORCE_GT(
        size,
        0,
        common::errors::Fatal("The size of output should be greater than 0."));
    float *result = static_cast<float *>(output[0].data.data());
    // output is probability, which is in (0, 1).
    for (size_t i = 0; i < size; i++) {
      EXPECT_GT(result[i], 0);
      EXPECT_LT(result[i], 1);
    }
  }
}

// Compare result of NativeConfig and AnalysisConfig
TEST(Analyzer_Pyramid_DNN, compare) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  CompareNativeAndAnalysis(
      reinterpret_cast<const PaddlePredictor::Config *>(&cfg), input_slots_all);
}

// Compare result of AnalysisConfig and AnalysisConfig + ZeroCopy
TEST(Analyzer_Pyramid_DNN, compare_zero_copy) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  AnalysisConfig cfg1;
  SetConfig(&cfg1);

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  std::vector<std::string> outputs_name;
  outputs_name.emplace_back("cos_sim_2.tmp_0");
  CompareAnalysisAndZeroCopy(reinterpret_cast<PaddlePredictor::Config *>(&cfg),
                             reinterpret_cast<PaddlePredictor::Config *>(&cfg1),
                             input_slots_all,
                             outputs_name);
}

// Compare Deterministic result
TEST(Analyzer_Pyramid_DNN, compare_determine) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  CompareDeterministic(reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
                       input_slots_all);
}

}  // namespace inference
}  // namespace paddle
