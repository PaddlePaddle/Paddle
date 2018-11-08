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
#define MAX_TURN_NUM 9
#define MAX_TURN_LEN 50
static std::vector<float> result_data;

struct DataRecord {
  std::vector<std::vector<int64_t>>
      turns[MAX_TURN_NUM];  // turns data : MAX_TURN_NUM
  std::vector<std::vector<float>>
      turns_mask[MAX_TURN_NUM];                // turns mask data : MAX_TURN_NUM
  std::vector<std::vector<int64_t>> response;  // response data : 1
  std::vector<std::vector<float>> response_mask;  // response mask data : 1
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
    if (batch_end <= response.size()) {
      for (int i = 0; i < MAX_TURN_NUM; ++i) {
        data.turns[i].assign(turns[i].begin() + batch_iter,
                             turns[i].begin() + batch_end);
      }
      for (int i = 0; i < MAX_TURN_NUM; ++i) {
        data.turns_mask[i].assign(turns_mask[i].begin() + batch_iter,
                                  turns_mask[i].begin() + batch_end);
      }
      data.response.assign(response.begin() + batch_iter,
                           response.begin() + batch_end);
      data.response_mask.assign(response_mask.begin() + batch_iter,
                                response_mask.begin() + batch_end);
      CHECK(!data.response.empty());
      CHECK(!data.response_mask.empty());
      CHECK_EQ(data.response.size(), data.response_mask.size());
    }
    batch_iter += batch_size;
    return data;
  }
  void Load(const std::string &path) {
    std::ifstream file(path);
    std::string line;
    size_t num_lines = 0;
    result_data.clear();
    while (std::getline(file, line)) {
      num_lines++;
      std::vector<std::string> data;
      split(line, ',', &data);
      CHECK_EQ(data.size(), 2 * MAX_TURN_NUM + 3);
      // load turn data
      std::vector<int64_t> turns_tmp[MAX_TURN_NUM];
      for (int i = 0; i < MAX_TURN_NUM; ++i) {
        split_to_int64(data[i], ' ', &turns_tmp[i]);
        turns[i].push_back(std::move(turns_tmp[i]));
      }
      // load turn_mask data
      std::vector<float> turns_mask_tmp[MAX_TURN_NUM];
      for (int i = 0; i < MAX_TURN_NUM; ++i) {
        split_to_float(data[MAX_TURN_NUM + i], ' ', &turns_mask_tmp[i]);
        turns_mask[i].push_back(std::move(turns_mask_tmp[i]));
      }
      // load response data
      std::vector<int64_t> response_tmp;
      split_to_int64(data[2 * MAX_TURN_NUM], ' ', &response_tmp);
      response.push_back(std::move(response_tmp));
      // load response_mask data
      std::vector<float> response_mask_tmp;
      split_to_float(data[2 * MAX_TURN_NUM + 1], ' ', &response_mask_tmp);
      response_mask.push_back(std::move(response_mask_tmp));
      // load result data
      float result_tmp;
      result_tmp = std::stof(data[2 * MAX_TURN_NUM + 2]);
      result_data.push_back(result_tmp);
    }
    num_samples = num_lines;
  }
};

void PrepareInputs(std::vector<PaddleTensor> *input_slots, DataRecord *data,
                   int batch_size) {
  PaddleTensor turns_tensor[MAX_TURN_NUM];
  PaddleTensor turns_mask_tensor[MAX_TURN_NUM];
  PaddleTensor response_tensor;
  PaddleTensor response_mask_tensor;
  std::string turn_pre = "turn_";
  std::string turn_mask_pre = "turn_mask_";

  auto one_batch = data->NextBatch();
  int size = one_batch.response[0].size();
  CHECK_EQ(size, MAX_TURN_LEN);
  // turn tensor assignment
  for (int i = 0; i < MAX_TURN_NUM; ++i) {
    turns_tensor[i].name = turn_pre + std::to_string(i);
    turns_tensor[i].shape.assign({batch_size, size, 1});
    turns_tensor[i].dtype = PaddleDType::INT64;
    TensorAssignData<int64_t>(&turns_tensor[i], one_batch.turns[i]);
  }
  // turn mask tensor assignment
  for (int i = 0; i < MAX_TURN_NUM; ++i) {
    turns_mask_tensor[i].name = turn_mask_pre + std::to_string(i);
    turns_mask_tensor[i].shape.assign({batch_size, size, 1});
    turns_mask_tensor[i].dtype = PaddleDType::FLOAT32;
    TensorAssignData<float>(&turns_mask_tensor[i], one_batch.turns_mask[i]);
  }
  // response tensor assignment
  response_tensor.name = "response";
  response_tensor.shape.assign({batch_size, size, 1});
  response_tensor.dtype = PaddleDType::INT64;
  TensorAssignData<int64_t>(&response_tensor, one_batch.response);
  // response mask tensor assignment
  response_mask_tensor.name = "response_mask";
  response_mask_tensor.shape.assign({batch_size, size, 1});
  response_mask_tensor.dtype = PaddleDType::FLOAT32;
  TensorAssignData<float>(&response_mask_tensor, one_batch.response_mask);

  // Set inputs.
  for (int i = 0; i < MAX_TURN_NUM; ++i) {
    input_slots->push_back(std::move(turns_tensor[i]));
  }
  for (int i = 0; i < MAX_TURN_NUM; ++i) {
    input_slots->push_back(std::move(turns_mask_tensor[i]));
  }
  input_slots->push_back(std::move(response_tensor));
  input_slots->push_back(std::move(response_mask_tensor));
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
  int test_batch_num =
      FLAGS_test_all_data ? data.num_samples / FLAGS_batch_size : 1;
  LOG(INFO) << "The number of samples to be test: "
            << test_batch_num * FLAGS_batch_size;
  for (int bid = 0; bid < test_batch_num; ++bid) {
    input_slots.clear();
    PrepareInputs(&input_slots, &data, FLAGS_batch_size);
    (*inputs).emplace_back(input_slots);
  }
}

// Easy for profiling independently.
TEST(Analyzer_dam, profile) {
  contrib::AnalysisConfig cfg;
  SetConfig(&cfg);

  std::vector<PaddleTensor> outputs;
  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  TestPrediction(cfg, input_slots_all, &outputs, FLAGS_num_threads);

  if (FLAGS_num_threads == 1 && !FLAGS_test_all_data) {
    PADDLE_ENFORCE_GT(outputs.size(), 0);
    size_t size = GetSize(outputs[0]);
    PADDLE_ENFORCE_GT(size, 0);
    float *result = static_cast<float *>(outputs[0].data.data());
    for (size_t i = 0; i < size; i++) {
      EXPECT_NEAR(result[i], result_data[i], 1e-3);
    }
  }
}

// Check the fuse status
TEST(Analyzer_dam, fuse_statis) {
  contrib::AnalysisConfig cfg;
  SetConfig(&cfg);

  if (FLAGS_use_analysis) {
    int num_ops;
    auto predictor = CreatePaddlePredictor<AnalysisConfig>(cfg);
    auto fuse_statis = GetFuseStatis(
        static_cast<AnalysisPredictor *>(predictor.get()), &num_ops);
    ASSERT_TRUE(fuse_statis.count("fc_fuse"));
    EXPECT_EQ(fuse_statis.at("fc_fuse"), 317);
    EXPECT_EQ(num_ops, 2020);
  }
}

// Compare result of NativeConfig and AnalysisConfig
TEST(Analyzer_dam, compare) {
  contrib::AnalysisConfig cfg;
  SetConfig(&cfg);

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);

  if (FLAGS_use_analysis) {
    CompareNativeAndAnalysis(cfg, input_slots_all);
  }
}

}  // namespace inference
}  // namespace paddle
