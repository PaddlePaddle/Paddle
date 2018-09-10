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

#include <google/protobuf/text_format.h>
#include <gtest/gtest.h>
#include <thread>  // NOLINT
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/inference/analysis/ut_helper.h"
#include "paddle/fluid/inference/api/analysis_predictor.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/api/paddle_inference_pass.h"
#include "paddle/fluid/inference/utils/singleton.h"

DEFINE_string(infer_model, "", "model path");
DEFINE_string(infer_data, "", "data path");
DEFINE_int32(batch_size, 10, "batch size.");
DEFINE_int32(repeat, 1, "Running the inference program repeat times.");
DEFINE_int32(num_threads, 1, "Running the inference program in multi-threads.");

namespace paddle {
namespace inference {
namespace analysis {

using namespace framework;  // NOLINT

TEST(Analyzer, analysis_without_tensorrt) {
  FLAGS_IA_enable_tensorrt_subgraph_engine = false;
  Argument argument;
  argument.fluid_model_dir.reset(new std::string(FLAGS_inference_model_dir));
  Analyzer analyser;
  analyser.Run(&argument);
}

TEST(Analyzer, analysis_with_tensorrt) {
  FLAGS_IA_enable_tensorrt_subgraph_engine = true;
  Argument argument;
  argument.fluid_model_dir.reset(new std::string(FLAGS_inference_model_dir));
  Analyzer analyser;
  analyser.Run(&argument);
}

void TestWord2vecPrediction(const std::string &model_path) {
  NativeConfig config;
  config.model_dir = model_path;
  config.use_gpu = false;
  config.device = 0;
  auto predictor =
      ::paddle::CreatePaddlePredictor<NativeConfig, PaddleEngineKind::kNative>(
          config);

  // One single batch

  int64_t data[4] = {1, 2, 3, 4};
  PaddleTensor tensor;
  tensor.shape = std::vector<int>({4, 1});
  tensor.data = PaddleBuf(data, sizeof(data));
  tensor.dtype = PaddleDType::INT64;

  // For simplicity, we set all the slots with the same data.
  std::vector<PaddleTensor> slots(4, tensor);
  std::vector<PaddleTensor> outputs;
  CHECK(predictor->Run(slots, &outputs));

  PADDLE_ENFORCE(outputs.size(), 1UL);
  // Check the output buffer size and result of each tid.
  PADDLE_ENFORCE(outputs.front().data.length(), 33168UL);
  float result[5] = {0.00129761, 0.00151112, 0.000423564, 0.00108815,
                     0.000932706};
  const size_t num_elements = outputs.front().data.length() / sizeof(float);
  // The outputs' buffers are in CPU memory.
  for (size_t i = 0; i < std::min(5UL, num_elements); i++) {
    LOG(INFO) << "data: "
              << static_cast<float *>(outputs.front().data.data())[i];
    PADDLE_ENFORCE(static_cast<float *>(outputs.front().data.data())[i],
                   result[i]);
  }
}

namespace {

struct DataRecord {
  std::vector<std::vector<std::vector<float>>> link_step_data_all;
  std::vector<std::vector<float>> week_data_all, minute_data_all;
  std::vector<size_t> lod1, lod2, lod3;
  std::vector<std::vector<float>> rnn_link_data, rnn_week_datas,
      rnn_minute_datas;
  size_t batch_iter{0};
  size_t batch_size{1};
  DataRecord() = default;
  explicit DataRecord(const std::string &path, int batch_size = 1)
      : batch_size(batch_size) {
    Load(path);
  }
  DataRecord NextBatch() {
    DataRecord data;
    size_t batch_end = batch_iter + batch_size;
    // NOTE skip the final batch, if no enough data is provided.
    if (batch_end <= link_step_data_all.size()) {
      data.link_step_data_all.assign(link_step_data_all.begin() + batch_iter,
                                     link_step_data_all.begin() + batch_end);
      data.week_data_all.assign(week_data_all.begin() + batch_iter,
                                week_data_all.begin() + batch_end);
      data.minute_data_all.assign(minute_data_all.begin() + batch_iter,
                                  minute_data_all.begin() + batch_end);
      // Prepare LoDs
      data.lod1.push_back(0);
      data.lod2.push_back(0);
      data.lod3.push_back(0);
      CHECK(!data.link_step_data_all.empty()) << "empty";
      CHECK(!data.week_data_all.empty());
      CHECK(!data.minute_data_all.empty());
      CHECK_EQ(data.link_step_data_all.size(), data.week_data_all.size());
      CHECK_EQ(data.minute_data_all.size(), data.link_step_data_all.size());
      for (size_t j = 0; j < data.link_step_data_all.size(); j++) {
        for (const auto &d : data.link_step_data_all[j]) {
          data.rnn_link_data.push_back(d);
        }
        data.rnn_week_datas.push_back(data.week_data_all[j]);
        data.rnn_minute_datas.push_back(data.minute_data_all[j]);
        // calculate lod
        data.lod1.push_back(data.lod1.back() +
                            data.link_step_data_all[j].size());
        data.lod3.push_back(data.lod3.back() + 1);
        for (size_t i = 1; i < data.link_step_data_all[j].size() + 1; i++) {
          data.lod2.push_back(data.lod2.back() +
                              data.link_step_data_all[j].size());
        }
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
      split(line, ':', &data);
      std::vector<std::vector<float>> link_step_data;
      std::vector<std::string> link_datas;
      split(data[0], '|', &link_datas);
      for (auto &step_data : link_datas) {
        std::vector<float> tmp;
        split_to_float(step_data, ',', &tmp);
        link_step_data.push_back(tmp);
      }
      // load week data
      std::vector<float> week_data;
      split_to_float(data[2], ',', &week_data);
      // load minute data
      std::vector<float> minute_data;
      split_to_float(data[1], ',', &minute_data);
      link_step_data_all.push_back(std::move(link_step_data));
      week_data_all.push_back(std::move(week_data));
      minute_data_all.push_back(std::move(minute_data));
    }
  }
};
void PrepareInputs(std::vector<PaddleTensor> *input_slots, DataRecord *data,
                   int batch_size) {
  PaddleTensor lod_attention_tensor, init_zero_tensor, lod_tensor_tensor,
      week_tensor, minute_tensor;
  lod_attention_tensor.name = "data_lod_attention";
  init_zero_tensor.name = "cell_init";
  lod_tensor_tensor.name = "data";
  week_tensor.name = "week";
  minute_tensor.name = "minute";
  auto one_batch = data->NextBatch();
  std::vector<int> rnn_link_data_shape(
      {static_cast<int>(one_batch.rnn_link_data.size()),
       static_cast<int>(one_batch.rnn_link_data.front().size())});
  lod_attention_tensor.shape.assign({1, 2});
  lod_attention_tensor.lod.assign({one_batch.lod1, one_batch.lod2});
  init_zero_tensor.shape.assign({batch_size, 15});
  init_zero_tensor.lod.assign({one_batch.lod3});
  lod_tensor_tensor.shape = rnn_link_data_shape;
  lod_tensor_tensor.lod.assign({one_batch.lod1});
  // clang-format off
  week_tensor.shape.assign(
      {static_cast<int>(one_batch.rnn_week_datas.size()),
       static_cast<int>(one_batch.rnn_week_datas.front().size())});
  week_tensor.lod.assign({one_batch.lod3});
  minute_tensor.shape.assign(
      {static_cast<int>(one_batch.rnn_minute_datas.size()),
       static_cast<int>(one_batch.rnn_minute_datas.front().size())});
  minute_tensor.lod.assign({one_batch.lod3});
  // clang-format on
  // assign data
  TensorAssignData<float>(&lod_attention_tensor,
                          std::vector<std::vector<float>>({{0, 0}}));
  std::vector<float> tmp_zeros(batch_size * 15, 0.);
  TensorAssignData<float>(&init_zero_tensor, {tmp_zeros});
  TensorAssignData<float>(&lod_tensor_tensor, one_batch.rnn_link_data);
  TensorAssignData<float>(&week_tensor, one_batch.rnn_week_datas);
  TensorAssignData<float>(&minute_tensor, one_batch.rnn_minute_datas);
  // Set inputs.
  auto init_zero_tensor1 = init_zero_tensor;
  init_zero_tensor1.name = "hidden_init";
  input_slots->assign({week_tensor, init_zero_tensor, minute_tensor,
                       init_zero_tensor1, lod_attention_tensor,
                       lod_tensor_tensor});
  for (auto &tensor : *input_slots) {
    tensor.dtype = PaddleDType::FLOAT32;
  }
}

}  // namespace

void CompareResult(const std::vector<PaddleTensor> &outputs,
                   const std::vector<PaddleTensor> &base_outputs) {
  PADDLE_ENFORCE_GT(outputs.size(), 0);
  PADDLE_ENFORCE_EQ(outputs.size(), base_outputs.size());
  for (size_t i = 0; i < outputs.size(); i++) {
    auto &out = outputs[i];
    auto &base_out = base_outputs[i];
    size_t size = std::accumulate(out.shape.begin(), out.shape.end(), 1,
                                  [](int a, int b) { return a * b; });
    size_t size1 = std::accumulate(base_out.shape.begin(), base_out.shape.end(),
                                   1, [](int a, int b) { return a * b; });
    PADDLE_ENFORCE_EQ(size, size1);
    PADDLE_ENFORCE_GT(size, 0);
    float *data = static_cast<float *>(out.data.data());
    float *base_data = static_cast<float *>(base_out.data.data());
    for (size_t i = 0; i < size; i++) {
      EXPECT_NEAR(data[i], base_data[i], 1e-3);
    }
  }
}
// Test with a really complicate model.
void TestRNN1Prediction(bool use_analysis, bool activate_ir, int num_threads) {
  AnalysisConfig config;
  config.prog_file = FLAGS_infer_model + "/__model__";
  config.param_file = FLAGS_infer_model + "/param";
  config.use_gpu = false;
  config.device = 0;
  config.specify_input_name = true;
  config.enable_ir_optim = activate_ir;
  PADDLE_ENFORCE(config.ir_mode ==
                 AnalysisConfig::IrPassMode::kExclude);  // default
  config.ir_passes.clear();  // Do not exclude any pass.

  int batch_size = FLAGS_batch_size;
  int num_times = FLAGS_repeat;

  auto base_predictor =
      CreatePaddlePredictor<NativeConfig, PaddleEngineKind::kNative>(config);
  auto predictor =
      CreatePaddlePredictor<AnalysisConfig, PaddleEngineKind::kAnalysis>(
          config);
  std::vector<PaddleTensor> input_slots;
  DataRecord data(FLAGS_infer_data, batch_size);
  // Prepare inputs.
  PrepareInputs(&input_slots, &data, batch_size);
  std::vector<PaddleTensor> outputs, base_outputs;

  base_predictor->Run(input_slots, &base_outputs);

  if (num_threads == 1) {
    // Prepare inputs.
    Timer timer;
    timer.tic();
    for (int i = 0; i < num_times; i++) {
      predictor->Run(input_slots, &outputs);
    }
    PrintTime(batch_size, num_times, 1, 0, timer.toc() / num_times);
    CompareResult(outputs, base_outputs);
  } else {
    std::vector<std::thread> threads;
    std::vector<std::unique_ptr<PaddlePredictor>> predictors;
    // TODO(yanchunwei): Bug here, the analyzer phase can't be parallelled
    // because AttentionLSTM's hard code nodeid will be damanged.
    for (int tid = 0; tid < num_threads; ++tid) {
      predictors.emplace_back(
          CreatePaddlePredictor<AnalysisConfig, PaddleEngineKind::kAnalysis>(
              config));
    }
    for (int tid = 0; tid < num_threads; ++tid) {
      threads.emplace_back([&, tid]() {
        // Each thread should have local input_slots and outputs.
        std::vector<PaddleTensor> input_slots;
        DataRecord data(FLAGS_infer_data, batch_size);
        PrepareInputs(&input_slots, &data, batch_size);
        std::vector<PaddleTensor> outputs;
        Timer timer;
        timer.tic();
        for (int i = 0; i < num_times; i++) {
          predictors[tid]->Run(input_slots, &outputs);
        }
        PrintTime(batch_size, num_times, num_threads, tid,
                  timer.toc() / num_times);
        CompareResult(outputs, base_outputs);
      });
    }
    for (int i = 0; i < num_threads; ++i) {
      threads[i].join();
    }
  }

  if (use_analysis && activate_ir) {
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
    EXPECT_EQ(fuse_statis.at("fc_fuse"), 1);
    EXPECT_EQ(fuse_statis.at("fc_nobias_lstm_fuse"), 2);  // bi-directional LSTM
    EXPECT_EQ(fuse_statis.at("seq_concat_fc_fuse"), 1);
    EXPECT_EQ(num_ops,
              13);  // After graph optimization, only 13 operators exists.
  }
}

// Inference with analysis and IR, easy for profiling independently.
TEST(Analyzer, rnn1) { TestRNN1Prediction(true, true, FLAGS_num_threads); }

// Other unit-tests of RNN1, test different options of use_analysis,
// activate_ir and multi-threads.
TEST(Analyzer, RNN_tests) {
  int num_threads[2] = {1, 4};
  for (auto i : num_threads) {
    // Directly infer with the original model.
    TestRNN1Prediction(false, false, i);
    // Inference with the original model with the analysis turned on, the
    // analysis
    // module will transform the program to a data flow graph.
    TestRNN1Prediction(true, false, i);
    // Inference with analysis and IR. The IR module will fuse some large
    // kernels.
    TestRNN1Prediction(true, true, i);
  }
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
