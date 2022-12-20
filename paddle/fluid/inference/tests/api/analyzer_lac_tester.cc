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

#include "paddle/fluid/inference/tests/api/tester_helper.h"

namespace paddle {
namespace inference {
namespace analysis {

struct DataRecord {
  std::vector<int64_t> data;
  std::vector<size_t> lod;
  // for dataset and nextbatch
  size_t batch_iter{0};
  std::vector<std::vector<size_t>> batched_lods;
  std::vector<std::vector<int64_t>> batched_datas;
  std::vector<std::vector<int64_t>> datasets;
  DataRecord() = default;
  explicit DataRecord(const std::string &path, int batch_size = 1) {
    Load(path);
    Prepare(batch_size);
    batch_iter = 0;
  }
  void Load(const std::string &path) {
    std::ifstream file(path);
    std::string line;
    int num_lines = 0;
    datasets.resize(0);
    while (std::getline(file, line)) {
      num_lines++;
      std::vector<std::string> data;
      split(line, ';', &data);
      std::vector<int64_t> words_ids;
      split_to_int64(data[1], ' ', &words_ids);
      datasets.emplace_back(words_ids);
    }
  }
  void Prepare(int bs) {
    if (bs == 1) {
      batched_datas = datasets;
      for (auto one_sentence : datasets) {
        batched_lods.push_back({0, one_sentence.size()});
      }
    } else {
      std::vector<int64_t> one_batch;
      std::vector<size_t> lod{0};
      int bs_id = 0;
      for (auto one_sentence : datasets) {
        bs_id++;
        one_batch.insert(
            one_batch.end(), one_sentence.begin(), one_sentence.end());
        lod.push_back(lod.back() + one_sentence.size());
        if (bs_id == bs) {
          bs_id = 0;
          batched_datas.push_back(one_batch);
          batched_lods.push_back(lod);
          one_batch.clear();
          one_batch.resize(0);
          lod.clear();
          lod.resize(0);
          lod.push_back(0);
        }
      }
      if (one_batch.size() != 0) {
        batched_datas.push_back(one_batch);
        batched_lods.push_back(lod);
      }
    }
  }

  DataRecord NextBatch() {
    DataRecord data;
    data.data = batched_datas[batch_iter];
    data.lod = batched_lods[batch_iter];
    batch_iter++;
    if (batch_iter >= batched_datas.size()) {
      batch_iter = 0;
    }
    return data;
  }
};

void GetOneBatch(std::vector<PaddleTensor> *input_slots,
                 DataRecord *data,
                 int batch_size) {
  auto one_batch = data->NextBatch();
  PaddleTensor input_tensor;
  input_tensor.name = "word";
  input_tensor.dtype = PaddleDType::INT64;
  TensorAssignData<int64_t>(&input_tensor, {one_batch.data}, one_batch.lod);
  PADDLE_ENFORCE_EQ(
      batch_size,
      static_cast<int>(one_batch.lod.size() - 1),
      paddle::platform::errors::Fatal("The lod size of one batch is invaild."));
  input_slots->assign({input_tensor});
}

void SetConfig(AnalysisConfig *cfg) {
  cfg->SetModel(FLAGS_infer_model);
  cfg->DisableGpu();
  cfg->SwitchSpecifyInputNames();
  cfg->SwitchIrOptim();
}

void SetInput(std::vector<std::vector<PaddleTensor>> *inputs) {
  DataRecord data(FLAGS_infer_data, FLAGS_batch_size);
  std::vector<PaddleTensor> input_slots;
  int epoch = FLAGS_test_all_data ? data.batched_datas.size() : 1;
  LOG(INFO) << "number of samples: " << epoch;
  for (int bid = 0; bid < epoch; ++bid) {
    GetOneBatch(&input_slots, &data, FLAGS_batch_size);
    (*inputs).emplace_back(input_slots);
  }
}

// Easy for profiling independently.
TEST(Analyzer_LAC, profile) {
  AnalysisConfig cfg;
  SetConfig(&cfg);
  std::vector<std::vector<PaddleTensor>> outputs;

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  TestPrediction(reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
                 input_slots_all,
                 &outputs,
                 FLAGS_num_threads);

  if (FLAGS_num_threads == 1 && !FLAGS_test_all_data) {
    // the first inference result
    const int64_t lac_ref_data[] = {
        24, 25, 25, 25, 38, 30, 31, 14, 15, 44, 24, 25, 25, 25, 25, 25,
        44, 24, 25, 25, 25, 36, 42, 43, 44, 14, 15, 44, 14, 15, 44, 14,
        15, 44, 38, 39, 14, 15, 44, 22, 23, 23, 23, 23, 23, 23, 23};
    PADDLE_ENFORCE_GT(outputs.size(),
                      0,
                      paddle::platform::errors::Fatal(
                          "The size of output should be greater than 0."));
    auto output = outputs.back();
    PADDLE_ENFORCE_EQ(output.size(),
                      1UL,
                      paddle::platform::errors::Fatal(
                          "The size of output should be equal to 1."));
    size_t size = GetSize(output[0]);
    size_t batch1_size = sizeof(lac_ref_data) / sizeof(int64_t);
    PADDLE_ENFORCE_GE(
        size,
        batch1_size,
        paddle::platform::errors::Fatal("The size of batch is invaild."));
    int64_t *pdata = static_cast<int64_t *>(output[0].data.data());
    for (size_t i = 0; i < batch1_size; ++i) {
      EXPECT_EQ(pdata[i], lac_ref_data[i]);
    }
  }
}

// Check the fuse status
TEST(Analyzer_LAC, fuse_statis) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  int num_ops;
  auto predictor = CreatePaddlePredictor<AnalysisConfig>(cfg);
  auto fuse_statis = GetFuseStatis(
      static_cast<AnalysisPredictor *>(predictor.get()), &num_ops);
  ASSERT_TRUE(fuse_statis.count("fc_fuse"));
  ASSERT_TRUE(fuse_statis.count("fc_gru_fuse"));
  EXPECT_EQ(fuse_statis.at("fc_fuse"), 1);
  EXPECT_EQ(fuse_statis.at("fc_gru_fuse"), 4);
}

// Compare result of NativeConfig and AnalysisConfig
TEST(Analyzer_LAC, compare) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  CompareNativeAndAnalysis(
      reinterpret_cast<const PaddlePredictor::Config *>(&cfg), input_slots_all);
}

// Compare Deterministic result
TEST(Analyzer_LAC, compare_determine) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  CompareDeterministic(reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
                       input_slots_all);
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
