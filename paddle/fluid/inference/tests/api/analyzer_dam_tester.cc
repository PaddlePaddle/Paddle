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

#include <vector>

#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/tests/api/tester_helper.h"

const int FLAGS_max_turn_num = 1;

namespace paddle {
namespace inference {

constexpr int32_t kMaxTurnLen = 50;

static std::vector<float> result_data;

struct DataRecord {
  std::vector<std::vector<int64_t>> *turns;
  std::vector<std::vector<float>> *turns_mask;
  std::vector<std::vector<int64_t>> response;     // response data : 1
  std::vector<std::vector<float>> response_mask;  // response mask data : 1
  size_t batch_iter{0};
  size_t batch_size{1};
  size_t num_samples;  // total number of samples

  DataRecord() {
    turns = new std::vector<std::vector<
        int64_t>>[FLAGS_max_turn_num];  // turns data : FLAGS_max_turn_num
    turns_mask = new std::vector<std::vector<
        float>>[FLAGS_max_turn_num];  // turns mask data : FLAGS_max_turn_num
  }

  explicit DataRecord(const std::string &path, int batch_size = 1)
      : DataRecord() {
    this->batch_size = batch_size;
    Load(path);
  }

  ~DataRecord() {
    delete[] turns;
    delete[] turns_mask;
  }

  DataRecord NextBatch() {
    DataRecord data;
    size_t batch_end = batch_iter + batch_size;
    // NOTE skip the final batch, if no enough data is provided.
    if (batch_end <= response.size()) {
      for (int i = 0; i < FLAGS_max_turn_num; ++i) {
        data.turns[i].assign(turns[i].begin() + batch_iter,
                             turns[i].begin() + batch_end);
      }
      for (int i = 0; i < FLAGS_max_turn_num; ++i) {
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
      CHECK_EQ(data.size(), (size_t)(2 * FLAGS_max_turn_num + 3));
      // load turn data
      std::vector<int64_t> turns_tmp[FLAGS_max_turn_num];
      for (int i = 0; i < FLAGS_max_turn_num; ++i) {
        split_to_int64(data[i], ' ', &turns_tmp[i]);
        turns[i].push_back(std::move(turns_tmp[i]));
      }
      // load turn_mask data
      std::vector<float> turns_mask_tmp[FLAGS_max_turn_num];
      for (int i = 0; i < FLAGS_max_turn_num; ++i) {
        split_to_float(data[FLAGS_max_turn_num + i], ' ', &turns_mask_tmp[i]);
        turns_mask[i].push_back(std::move(turns_mask_tmp[i]));
      }
      // load response data
      std::vector<int64_t> response_tmp;
      split_to_int64(data[2 * FLAGS_max_turn_num], ' ', &response_tmp);
      response.push_back(std::move(response_tmp));
      // load response_mask data
      std::vector<float> response_mask_tmp;
      split_to_float(data[2 * FLAGS_max_turn_num + 1], ' ', &response_mask_tmp);
      response_mask.push_back(std::move(response_mask_tmp));
      // load result data
      float result_tmp;
      result_tmp = std::stof(data[2 * FLAGS_max_turn_num + 2]);
      result_data.push_back(result_tmp);
    }
    num_samples = num_lines;
  }
};

void PrepareInputs(std::vector<PaddleTensor> *input_slots,
                   DataRecord *data,
                   int batch_size) {
  PaddleTensor turns_tensor[FLAGS_max_turn_num];
  PaddleTensor turns_mask_tensor[FLAGS_max_turn_num];
  PaddleTensor response_tensor;
  PaddleTensor response_mask_tensor;
  std::string turn_pre = "turn_";
  std::string turn_mask_pre = "turn_mask_";

  auto one_batch = data->NextBatch();
  PADDLE_ENFORCE(
      !one_batch.response.empty(),
      paddle::platform::errors::Fatal("The response of one batch is empty."));
  int size = one_batch.response[0].size();
  CHECK_EQ(size, kMaxTurnLen);
  // turn tensor assignment
  for (int i = 0; i < FLAGS_max_turn_num; ++i) {
    turns_tensor[i].name = turn_pre + std::to_string(i);
    turns_tensor[i].shape.assign({batch_size, size, 1});
    turns_tensor[i].dtype = PaddleDType::INT64;
    TensorAssignData<int64_t>(&turns_tensor[i], one_batch.turns[i]);
  }
  // turn mask tensor assignment
  for (int i = 0; i < FLAGS_max_turn_num; ++i) {
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
  for (int i = 0; i < FLAGS_max_turn_num; ++i) {
    input_slots->push_back(std::move(turns_tensor[i]));
  }
  for (int i = 0; i < FLAGS_max_turn_num; ++i) {
    input_slots->push_back(std::move(turns_mask_tensor[i]));
  }
  input_slots->push_back(std::move(response_tensor));
  input_slots->push_back(std::move(response_mask_tensor));
}

/*
 * this model is unreasonable, it set a output tensor persistable, so
 * ridiculous! so I disable constant_folding_pass
 */

void SetConfig(AnalysisConfig *cfg) {
  cfg->SetModel(FLAGS_infer_model + "/__model__", FLAGS_infer_model + "/param");
  cfg->SwitchSpecifyInputNames();
  auto pass_builder = cfg->pass_builder();
  pass_builder->DeletePass("constant_folding_pass");
  cfg->SwitchIrOptim(true);
}

void SetOptimConfig(AnalysisConfig *cfg) {
  std::string optimModelPath = FLAGS_infer_model + "/saved_optim_model";
  cfg->SetModel(optimModelPath + "/model", optimModelPath + "/params");
  cfg->SwitchIrOptim(true);
  cfg->SwitchSpecifyInputNames();
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
void profile(bool use_mkldnn = false) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  if (use_mkldnn) {
    cfg.EnableMKLDNN();
    // Enable all the mkldnn supported ops except conv3d in dam
    std::unordered_set<std::string> op_list = {
        "softmax", "elementwise_add", "relu", "fc"};
    cfg.SetMKLDNNOp(op_list);
  }

  std::vector<std::vector<PaddleTensor>> outputs;
  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);

  TestPrediction(reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
                 input_slots_all,
                 &outputs,
                 FLAGS_num_threads);

  if (FLAGS_num_threads == 1 && !FLAGS_test_all_data) {
    PADDLE_ENFORCE_GT(outputs.size(),
                      0,
                      paddle::platform::errors::Fatal(
                          "The size of outputs should be greater than 0."));
    auto output = outputs.back();
    PADDLE_ENFORCE_GT(output.size(),
                      0,
                      paddle::platform::errors::Fatal(
                          "The size of output should be greater than 0."));
    size_t size = GetSize(output[0]);
    PADDLE_ENFORCE_GT(size,
                      0,
                      paddle::platform::errors::Fatal(
                          "The size of output should be greater than 0."));
    float *result = static_cast<float *>(output[0].data.data());
    for (size_t i = 0; i < size; i++) {
      EXPECT_NEAR(result[i], result_data[i], 1e-3);
    }
  }
}

TEST(Analyzer_dam, profile) { profile(); }
#ifdef PADDLE_WITH_MKLDNN
TEST(Analyzer_dam, profile_mkldnn) { profile(true /* use_mkldnn */); }
#endif

// Check the fuse status
TEST(Analyzer_dam, fuse_statis) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  int num_ops;
  auto predictor = CreatePaddlePredictor<AnalysisConfig>(cfg);
  auto fuse_statis = GetFuseStatis(
      static_cast<AnalysisPredictor *>(predictor.get()), &num_ops);
  ASSERT_TRUE(fuse_statis.count("fc_fuse"));
}

// Compare result of NativeConfig and AnalysisConfig
void compare(bool use_mkldnn = false) {
  AnalysisConfig cfg;
  SetConfig(&cfg);
  if (use_mkldnn) {
    cfg.EnableMKLDNN();
    // Enable all the mkldnn supported ops except conv3d in dam
    std::unordered_set<std::string> op_list = {
        "softmax", "elementwise_add", "relu"};
    cfg.SetMKLDNNOp(op_list);
  }

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);

  CompareNativeAndAnalysis(
      reinterpret_cast<const PaddlePredictor::Config *>(&cfg), input_slots_all);
}

TEST(Analyzer_dam, compare_with_dynamic_memory_optim) {
  // The small dam will core in CI, but works in local.
  if (FLAGS_max_turn_num == 9) {
    AnalysisConfig cfg, cfg1;
    DataRecord data(FLAGS_infer_data, FLAGS_batch_size);

    std::vector<std::vector<PaddleTensor>> input_slots_all;
    SetInput(&input_slots_all);
    // Run the first time to force to update memory cache
    SetConfig(&cfg);
    cfg.EnableMemoryOptim();

    CompareNativeAndAnalysis(
        reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
        input_slots_all);
  }
}

TEST(Analyzer_dam, compare) { compare(); }

#ifdef PADDLE_WITH_MKLDNN
TEST(Analyzer_dam, compare_mkldnn) { compare(true /* use_mkldnn */); }
#endif

// Compare Deterministic result
TEST(Analyzer_dam, compare_determine) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  CompareDeterministic(reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
                       input_slots_all);
}
// Save optim model
TEST(Analyzer_dam, save_optim_model) {
  AnalysisConfig cfg;
  std::string optimModelPath = FLAGS_infer_model + "/saved_optim_model";
  MKDIR(optimModelPath.c_str());
  SetConfig(&cfg);
  SaveOptimModel(&cfg, optimModelPath);
}

void CompareOptimAndOrig(const PaddlePredictor::Config *orig_config,
                         const PaddlePredictor::Config *optim_config,
                         const std::vector<std::vector<PaddleTensor>> &inputs) {
  PrintConfig(orig_config, true);
  PrintConfig(optim_config, true);
  std::vector<std::vector<PaddleTensor>> orig_outputs, optim_outputs;
  TestOneThreadPrediction(orig_config, inputs, &orig_outputs, false);
  TestOneThreadPrediction(optim_config, inputs, &optim_outputs, false);
  CompareResult(orig_outputs.back(), optim_outputs.back());
}

TEST(Analyzer_dam, compare_optim_orig) {
  AnalysisConfig orig_cfg;
  AnalysisConfig optim_cfg;
  SetConfig(&orig_cfg);
  SetOptimConfig(&optim_cfg);
  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  CompareOptimAndOrig(
      reinterpret_cast<const PaddlePredictor::Config *>(&orig_cfg),
      reinterpret_cast<const PaddlePredictor::Config *>(&optim_cfg),
      input_slots_all);
}

}  // namespace inference
}  // namespace paddle
