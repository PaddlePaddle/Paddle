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
using contrib::AnalysisConfig;

struct DataRecord {
  std::vector<std::vector<int64_t>> word_data_all, mention_data_all;
  std::vector<size_t> lod;  // two inputs have the same lod info.
  size_t batch_iter{0};
  size_t batch_size{1};
  size_t num_samples;  // total number of samples
  DataRecord() = default;
  explicit DataRecord(const std::string &path, int batch_size = 1)
      : batch_size(batch_size) {
    Load(path);
  }
  DataRecord NextBatch() {
    DataRecord data;
    size_t batch_end = batch_iter + batch_size;
    // NOTE skip the final batch, if no enough data is provided.
    if (batch_end <= word_data_all.size()) {
      data.word_data_all.assign(word_data_all.begin() + batch_iter,
                                word_data_all.begin() + batch_end);
      data.mention_data_all.assign(mention_data_all.begin() + batch_iter,
                                   mention_data_all.begin() + batch_end);
      // Prepare LoDs
      data.lod.push_back(0);
      CHECK(!data.word_data_all.empty());
      CHECK(!data.mention_data_all.empty());
      CHECK_EQ(data.word_data_all.size(), data.mention_data_all.size());
      for (size_t j = 0; j < data.word_data_all.size(); j++) {
        // calculate lod
        data.lod.push_back(data.lod.back() + data.word_data_all[j].size());
      }
    }
    batch_iter += batch_size;
    return data;
  }
  void Load(const std::string &path) {
    std::ifstream file(path);
    std::string line;
    int num_lines = 0;
    while (std::getline(file, line)) {
      num_lines++;
      std::vector<std::string> data;
      split(line, ';', &data);
      // load word data
      std::vector<int64_t> word_data;
      split_to_int64(data[1], ' ', &word_data);
      // load mention data
      std::vector<int64_t> mention_data;
      split_to_int64(data[3], ' ', &mention_data);
      word_data_all.push_back(std::move(word_data));
      mention_data_all.push_back(std::move(mention_data));
    }
    num_samples = num_lines;
  }
};

void PrepareInputs(std::vector<PaddleTensor> *input_slots, DataRecord *data,
                   int batch_size) {
  PaddleTensor lod_word_tensor, lod_mention_tensor;
  lod_word_tensor.name = "word";
  lod_mention_tensor.name = "mention";
  auto one_batch = data->NextBatch();
  int size = one_batch.lod[one_batch.lod.size() - 1];  // token batch size
  lod_word_tensor.shape.assign({size, 1});
  lod_word_tensor.lod.assign({one_batch.lod});
  lod_mention_tensor.shape.assign({size, 1});
  lod_mention_tensor.lod.assign({one_batch.lod});
  // assign data
  TensorAssignData<int64_t>(&lod_word_tensor, one_batch.word_data_all);
  TensorAssignData<int64_t>(&lod_mention_tensor, one_batch.mention_data_all);
  // Set inputs.
  input_slots->assign({lod_word_tensor, lod_mention_tensor});
  for (auto &tensor : *input_slots) {
    tensor.dtype = PaddleDType::INT64;
  }
}

void SetConfig(contrib::AnalysisConfig *cfg) {
  cfg->prog_file = FLAGS_infer_model + "/__model__";
  cfg->param_file = FLAGS_infer_model + "/param";
  cfg->use_gpu = false;
  cfg->device = 0;
  cfg->specify_input_name = true;
  cfg->enable_ir_optim = true;
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
TEST(Analyzer_Chinese_ner, profile) {
  contrib::AnalysisConfig cfg;
  SetConfig(&cfg);
  std::vector<PaddleTensor> outputs;

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  TestPrediction(reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
                 input_slots_all, &outputs, FLAGS_num_threads);

  if (FLAGS_num_threads == 1 && !FLAGS_test_all_data) {
    // the first inference result
    const int chinese_ner_result_data[] = {30, 45, 41, 48, 17, 26,
                                           48, 39, 38, 16, 25};
    PADDLE_ENFORCE_EQ(outputs.size(), 1UL);
    size_t size = GetSize(outputs[0]);
    PADDLE_ENFORCE_GT(size, 0);
    int64_t *result = static_cast<int64_t *>(outputs[0].data.data());
    for (size_t i = 0; i < std::min(11UL, size); i++) {
      EXPECT_EQ(result[i], chinese_ner_result_data[i]);
    }
  }
}

// Check the fuse status
TEST(Analyzer_Chinese_ner, fuse_statis) {
  contrib::AnalysisConfig cfg;
  SetConfig(&cfg);

  int num_ops;
  auto predictor = CreatePaddlePredictor<AnalysisConfig>(cfg);
  auto fuse_statis = GetFuseStatis(
      static_cast<AnalysisPredictor *>(predictor.get()), &num_ops);
  ASSERT_TRUE(fuse_statis.count("fc_fuse"));
  ASSERT_TRUE(fuse_statis.count("fc_gru_fuse"));
  EXPECT_EQ(fuse_statis.at("fc_fuse"), 1);
  EXPECT_EQ(fuse_statis.at("fc_gru_fuse"), 2);
  EXPECT_EQ(num_ops, 14);
}

// Compare result of NativeConfig and AnalysisConfig
TEST(Analyzer_Chinese_ner, compare) {
  contrib::AnalysisConfig cfg;
  SetConfig(&cfg);

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  CompareNativeAndAnalysis(
      reinterpret_cast<const PaddlePredictor::Config *>(&cfg), input_slots_all);
}

}  // namespace inference
}  // namespace paddle
