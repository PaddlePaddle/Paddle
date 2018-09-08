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

#include "paddle/fluid/inference/analysis/analyzer.h"
#include <gtest/gtest.h>
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/inference/analysis/ut_helper.h"
#include "paddle/fluid/inference/api/analysis_predictor.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_inference_pass.h"
#include "paddle/fluid/platform/profiler.h"

DEFINE_string(infer_model, "", "model path");
DEFINE_string(infer_data, "", "data path");
DEFINE_int32(batch_size, 10, "batch size.");
DEFINE_int32(repeat, 1, "Running the inference program repeat times.");
DEFINE_bool(test_all_data, false, "Test the all dataset in data file.");

namespace paddle {
namespace inference {

struct DataRecord {
  std::vector<std::vector<int64_t>> word_data_all, mention_data_all;
  std::vector<std::vector<int64_t>> rnn_word_datas, rnn_mention_datas;
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
        data.rnn_word_datas.push_back(data.word_data_all[j]);
        data.rnn_mention_datas.push_back(data.mention_data_all[j]);
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
  TensorAssignData<int64_t>(&lod_word_tensor, one_batch.rnn_word_datas);
  TensorAssignData<int64_t>(&lod_mention_tensor, one_batch.rnn_mention_datas);
  // Set inputs.
  input_slots->assign({lod_word_tensor, lod_mention_tensor});
  for (auto &tensor : *input_slots) {
    tensor.dtype = PaddleDType::INT64;
  }
}

// the first inference result
const int chinese_ner_result_data[] = {30, 45, 41, 48, 17, 26,
                                       48, 39, 38, 16, 25};

void TestChineseNERPrediction(bool use_analysis) {
  NativeConfig config;
  config.prog_file = FLAGS_infer_model + "/__model__";
  config.param_file = FLAGS_infer_model + "/param";
  config.use_gpu = false;
  config.device = 0;
  config.specify_input_name = true;

  std::vector<PaddleTensor> input_slots, outputs;
  std::unique_ptr<PaddlePredictor> predictor;
  Timer timer;
  if (use_analysis) {
    AnalysisConfig cfg;
    cfg.prog_file = FLAGS_infer_model + "/__model__";
    cfg.param_file = FLAGS_infer_model + "/param";
    cfg.use_gpu = false;
    cfg.device = 0;
    cfg.specify_input_name = true;
    cfg.enable_ir_optim = true;
    predictor =
        CreatePaddlePredictor<AnalysisConfig, PaddleEngineKind::kAnalysis>(cfg);
  } else {
    predictor =
        CreatePaddlePredictor<NativeConfig, PaddleEngineKind::kNative>(config);
  }

  if (FLAGS_test_all_data) {
    LOG(INFO) << "test all data";
    double sum = 0;
    size_t num_samples;
    for (int i = 0; i < FLAGS_repeat; i++) {
      DataRecord data(FLAGS_infer_data, FLAGS_batch_size);
      num_samples = data.num_samples;
      for (size_t bid = 0; bid < num_samples; ++bid) {
        PrepareInputs(&input_slots, &data, FLAGS_batch_size);
        timer.tic();
        predictor->Run(input_slots, &outputs);
        sum += timer.toc();
      }
    }
    LOG(INFO) << "total number of samples: " << num_samples;
    PrintTime(FLAGS_batch_size, FLAGS_repeat, 1, 0, sum / FLAGS_repeat);
    LOG(INFO) << "average latency of each sample: "
              << sum / FLAGS_repeat / num_samples;
    return;
  }
  // Prepare inputs.
  DataRecord data(FLAGS_infer_data, FLAGS_batch_size);
  PrepareInputs(&input_slots, &data, FLAGS_batch_size);

  timer.tic();
  for (int i = 0; i < FLAGS_repeat; i++) {
    predictor->Run(input_slots, &outputs);
  }
  PrintTime(FLAGS_batch_size, FLAGS_repeat, 1, 0, timer.toc() / FLAGS_repeat);

  PADDLE_ENFORCE(outputs.size(), 1UL);
  auto &out = outputs[0];
  size_t size = std::accumulate(out.shape.begin(), out.shape.end(), 1,
                                [](int a, int b) { return a * b; });
  PADDLE_ENFORCE_GT(size, 0);
  int64_t *result = static_cast<int64_t *>(out.data.data());
  for (size_t i = 0; i < std::min(11UL, size); i++) {
    PADDLE_ENFORCE(result[i], chinese_ner_result_data[i]);
  }

  if (use_analysis) {
    // run once for comparion as reference
    auto ref_predictor =
        CreatePaddlePredictor<NativeConfig, PaddleEngineKind::kNative>(config);
    std::vector<PaddleTensor> ref_outputs_slots;
    ref_predictor->Run(input_slots, &ref_outputs_slots);
    EXPECT_EQ(ref_outputs_slots.size(), outputs.size());
    auto &ref_out = ref_outputs_slots[0];
    size_t ref_size =
        std::accumulate(ref_out.shape.begin(), ref_out.shape.end(), 1,
                        [](int a, int b) { return a * b; });
    EXPECT_EQ(size, ref_size);
    int64_t *pdata_ref = static_cast<int64_t *>(ref_out.data.data());
    for (size_t i = 0; i < size; ++i) {
      EXPECT_EQ(pdata_ref[i], result[i]);
    }

    AnalysisPredictor *analysis_predictor =
        dynamic_cast<AnalysisPredictor *>(predictor.get());
    auto &fuse_statis = analysis_predictor->analysis_argument()
                            .Get<std::unordered_map<std::string, int>>(
                                framework::ir::kFuseStatisAttr);
    for (auto &item : fuse_statis) {
      LOG(INFO) << "fused " << item.first << " " << item.second;
    }
    int num_ops = 0;
    for (auto &node :
         analysis_predictor->analysis_argument().main_dfg->nodes.nodes()) {
      if (node->IsFunction()) {
        ++num_ops;
      }
    }
    LOG(INFO) << "has num ops: " << num_ops;
    ASSERT_TRUE(fuse_statis.count("fc_fuse"));
    ASSERT_TRUE(fuse_statis.count("fc_gru_fuse"));
    EXPECT_EQ(fuse_statis.at("fc_fuse"), 1);
    EXPECT_EQ(fuse_statis.at("fc_gru_fuse"), 2);
    EXPECT_EQ(num_ops, 14);
  }
}

TEST(Analyzer_Chinese_ner, native) { TestChineseNERPrediction(false); }

TEST(Analyzer_Chinese_ner, analysis) { TestChineseNERPrediction(true); }

}  // namespace inference
}  // namespace paddle
