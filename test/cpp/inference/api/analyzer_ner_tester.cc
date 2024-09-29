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
  std::vector<std::vector<int64_t>> word, mention;
  std::vector<size_t> lod;  // two inputs have the same lod info.
  size_t batch_iter{0}, batch_size{1}, num_samples;  // total number of samples
  DataRecord() : word(), mention(), lod(), num_samples(0) {}
  explicit DataRecord(const std::string &path, int batch_size = 1)
      : word(), mention(), lod(), batch_size(batch_size), num_samples(0) {
    Load(path);
  }
  DataRecord NextBatch() {
    DataRecord data;
    size_t batch_end = batch_iter + batch_size;
    // NOTE skip the final batch, if no enough data is provided.
    if (batch_end <= word.size()) {
      GetInputPerBatch(word, &data.word, &data.lod, batch_iter, batch_end);
      GetInputPerBatch(
          mention, &data.mention, &data.lod, batch_iter, batch_end);
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
      word.push_back(std::move(word_data));
      mention.push_back(std::move(mention_data));
    }
    num_samples = num_lines;
  }
};

void PrepareInputs(std::vector<PaddleTensor> *input_slots, DataRecord *data) {
  PaddleTensor lod_word_tensor, lod_mention_tensor;
  lod_word_tensor.name = "word";
  lod_mention_tensor.name = "mention";
  auto one_batch = data->NextBatch();
  // assign data
  TensorAssignData<int64_t>(&lod_word_tensor, one_batch.word, one_batch.lod);
  TensorAssignData<int64_t>(
      &lod_mention_tensor, one_batch.mention, one_batch.lod);
  // Set inputs.
  input_slots->assign({lod_word_tensor, lod_mention_tensor});
  for (auto &tensor : *input_slots) {
    tensor.dtype = PaddleDType::INT64;
  }
}

void SetConfig(AnalysisConfig *cfg, bool memory_load = false) {
  if (memory_load) {
    std::string buffer_prog, buffer_param;
    ReadBinaryFile(FLAGS_infer_model + "/__model__", &buffer_prog);
    ReadBinaryFile(FLAGS_infer_model + "/param", &buffer_param);
    cfg->SetModelBuffer(&buffer_prog[0],
                        buffer_prog.size(),
                        &buffer_param[0],
                        buffer_param.size());
  } else {
    cfg->SetModel(FLAGS_infer_model + "/__model__",
                  FLAGS_infer_model + "/param");
  }
  cfg->DisableGpu();
  cfg->SwitchSpecifyInputNames();
  cfg->SwitchIrOptim();
}

void SetInput(std::vector<std::vector<PaddleTensor>> *inputs) {
  DataRecord data(FLAGS_infer_data, FLAGS_batch_size);
  std::vector<PaddleTensor> input_slots;
  int epoch =
      FLAGS_test_all_data ? data.num_samples / FLAGS_batch_size : 1;  // NOLINT
  LOG(INFO) << "number of samples: " << epoch * FLAGS_batch_size;
  for (int bid = 0; bid < epoch; ++bid) {
    PrepareInputs(&input_slots, &data);
    (*inputs).emplace_back(input_slots);
  }
}

// Easy for profiling independently.
void profile(bool memory_load = false) {
  AnalysisConfig cfg;
  SetConfig(&cfg, memory_load);
  std::vector<std::vector<PaddleTensor>> outputs;

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  TestPrediction(reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
                 input_slots_all,
                 &outputs,
                 FLAGS_num_threads);

  if (FLAGS_num_threads == 1 && !FLAGS_test_all_data) {
    // the first inference result
    const std::array<int, 11> chinese_ner_result_data = {
        30, 45, 41, 48, 17, 26, 48, 39, 38, 16, 25};
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
    int64_t *result = static_cast<int64_t *>(output[0].data.data());
    for (size_t i = 0; i < std::min<size_t>(11, size); i++) {
      EXPECT_EQ(result[i], chinese_ner_result_data[i]);
    }
  }
}

TEST(Analyzer_Chinese_ner, profile) { profile(); }

TEST(Analyzer_Chinese_ner, profile_memory_load) {
  profile(true /* memory_load */);
}

// Compare result of NativeConfig and AnalysisConfig
TEST(Analyzer_Chinese_ner, compare) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  CompareNativeAndAnalysis(
      reinterpret_cast<const PaddlePredictor::Config *>(&cfg), input_slots_all);
}

// Compare Deterministic result
TEST(Analyzer_Chinese_ner, compare_determine) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  CompareDeterministic(reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
                       input_slots_all);
}

}  // namespace inference
}  // namespace paddle
