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

DEFINE_bool(with_precision_check, true, "turn on test");

namespace paddle {
namespace inference {

using namespace framework;  // NOLINT
using namespace contrib;    // NOLINT

struct DataRecord {
  std::vector<std::vector<std::vector<float>>> link_step_data_all;
  std::vector<std::vector<float>> week_data_all, minute_data_all;
  std::vector<size_t> lod1, lod2, lod3;
  std::vector<std::vector<float>> rnn_link_data, rnn_week_datas,
      rnn_minute_datas;
  size_t num_samples;  // total number of samples
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
    num_samples = num_lines;
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

void PrepareZeroCopyInputs(ZeroCopyTensor *lod_attention_tensor,
                           ZeroCopyTensor *cell_init_tensor,
                           ZeroCopyTensor *data_tensor,
                           ZeroCopyTensor *hidden_init_tensor,
                           ZeroCopyTensor *week_tensor,
                           ZeroCopyTensor *minute_tensor,
                           DataRecord *data_record, int batch_size) {
  auto one_batch = data_record->NextBatch();
  std::vector<int> rnn_link_data_shape(
      {static_cast<int>(one_batch.rnn_link_data.size()),
       static_cast<int>(one_batch.rnn_link_data.front().size())});
  lod_attention_tensor->Reshape({1, 2});
  lod_attention_tensor->SetLoD({one_batch.lod1, one_batch.lod2});

  cell_init_tensor->Reshape({batch_size, 15});
  cell_init_tensor->SetLoD({one_batch.lod3});

  hidden_init_tensor->Reshape({batch_size, 15});
  hidden_init_tensor->SetLoD({one_batch.lod3});

  data_tensor->Reshape(rnn_link_data_shape);
  data_tensor->SetLoD({one_batch.lod1});

  week_tensor->Reshape(
      {static_cast<int>(one_batch.rnn_week_datas.size()),
       static_cast<int>(one_batch.rnn_week_datas.front().size())});
  week_tensor->SetLoD({one_batch.lod3});

  minute_tensor->Reshape(
      {static_cast<int>(one_batch.rnn_minute_datas.size()),
       static_cast<int>(one_batch.rnn_minute_datas.front().size())});
  minute_tensor->SetLoD({one_batch.lod3});

  // assign data
  float arr0[] = {0, 0};
  std::vector<float> zeros(batch_size * 15, 0);
  std::copy_n(arr0, 2,
              lod_attention_tensor->mutable_data<float>(PaddlePlace::kCPU));
  std::copy_n(arr0, 2, data_tensor->mutable_data<float>(PaddlePlace::kCPU));
  std::copy_n(zeros.begin(), zeros.size(),
              cell_init_tensor->mutable_data<float>(PaddlePlace::kCPU));
  std::copy_n(zeros.begin(), zeros.size(),
              hidden_init_tensor->mutable_data<float>(PaddlePlace::kCPU));
  ZeroCopyTensorAssignData(data_tensor, one_batch.rnn_link_data);
  ZeroCopyTensorAssignData(week_tensor, one_batch.rnn_week_datas);
  ZeroCopyTensorAssignData(minute_tensor, one_batch.rnn_minute_datas);
}

void SetConfig(AnalysisConfig *cfg) {
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
TEST(Analyzer_rnn1, profile) {
  contrib::AnalysisConfig cfg(false);
  SetConfig(&cfg);
  cfg.fraction_of_gpu_memory = 0.1;
  cfg.pass_builder()->TurnOnDebug();
  std::vector<PaddleTensor> outputs;

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  TestPrediction(reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
                 input_slots_all, &outputs, FLAGS_num_threads);
}

// Check the fuse status
TEST(Analyzer_rnn1, fuse_statis) {
  contrib::AnalysisConfig cfg;
  SetConfig(&cfg);

  int num_ops;
  auto predictor = CreatePaddlePredictor<AnalysisConfig>(cfg);
  auto fuse_statis = GetFuseStatis(
      static_cast<AnalysisPredictor *>(predictor.get()), &num_ops);
  ASSERT_TRUE(fuse_statis.count("fc_fuse"));
  EXPECT_EQ(fuse_statis.at("fc_fuse"), 1);
  EXPECT_EQ(fuse_statis.at("fc_nobias_lstm_fuse"), 2);  // bi-directional LSTM
  EXPECT_EQ(fuse_statis.at("seq_concat_fc_fuse"), 1);
  EXPECT_EQ(num_ops,
            13);  // After graph optimization, only 13 operators exists.
}

// Compare result of NativeConfig and AnalysisConfig
TEST(Analyzer_rnn1, compare) {
  contrib::AnalysisConfig cfg;
  SetConfig(&cfg);

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  CompareNativeAndAnalysis(
      reinterpret_cast<const PaddlePredictor::Config *>(&cfg), input_slots_all);
}

// Test Multi-Thread.
TEST(Analyzer_rnn1, multi_thread) {
  contrib::AnalysisConfig cfg;
  SetConfig(&cfg);
  std::vector<PaddleTensor> outputs;

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  TestPrediction(reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
                 input_slots_all, &outputs, 4 /* multi_thread */);
}

// Validate that the AnalysisPredictor + ZeroCopyTensor really works by testing
// on the complex RNN1 model.
TEST(Analyzer_rnn1, ZeroCopy) {
  AnalysisConfig config;
  SetConfig(&config);
  config.use_feed_fetch_ops = false;

  PaddlePlace place;

  auto predictor = CreatePaddlePredictor<AnalysisConfig>(config);

  config.use_feed_fetch_ops = true;
  auto native_predictor = CreatePaddlePredictor<NativeConfig>(config);

  config.use_feed_fetch_ops = true;  // the analysis predictor needs feed/fetch.
  auto analysis_predictor = CreatePaddlePredictor<AnalysisConfig>(config);

#define NEW_TENSOR(name__) \
  auto name__##_tensor = predictor->GetInputTensor(#name__);
  NEW_TENSOR(data_lod_attention);
  NEW_TENSOR(cell_init);
  NEW_TENSOR(data);
  NEW_TENSOR(week);
  NEW_TENSOR(minute);
  NEW_TENSOR(hidden_init);

  // Prepare data for AnalysisPredictor
  DataRecord data(FLAGS_infer_data, FLAGS_batch_size);
  PrepareZeroCopyInputs(data_lod_attention_tensor.get(), cell_init_tensor.get(),
                        data_tensor.get(), hidden_init_tensor.get(),
                        week_tensor.get(), minute_tensor.get(), &data,
                        FLAGS_batch_size);

  // Prepare data for NativePredictor
  std::vector<std::vector<PaddleTensor>> native_inputs;
  SetInput(&native_inputs);
  std::vector<PaddleTensor> native_outputs;
  std::vector<PaddleTensor> analysis_outputs;

  auto output_tensor = predictor->GetOutputTensor("final_output.tmp_1");
  // Run analysis predictor

  int num_ops;
  auto fuse_statis = GetFuseStatis(predictor.get(), &num_ops);
  ASSERT_TRUE(fuse_statis.count("fc_fuse"));
  ASSERT_EQ(fuse_statis.at("fc_fuse"), 1);
  ASSERT_EQ(fuse_statis.at("fc_nobias_lstm_fuse"), 2);  // bi-directional LSTM
  ASSERT_EQ(fuse_statis.at("seq_concat_fc_fuse"), 1);
  ASSERT_EQ(num_ops,
            13);  // After graph optimization, only 13 operators exists.

  Timer timer;
  double total_time{0};
  for (int i = 0; i < FLAGS_repeat; i++) {
    timer.tic();
    predictor->ZeroCopyRun();
    total_time += timer.toc();
  }
  LOG(INFO) << "ZeroCopy output: " << DescribeZeroCopyTensor(*output_tensor);

  ASSERT_TRUE(native_predictor->Run(native_inputs.front(), &native_outputs));
  LOG(INFO) << "native output " << DescribeTensor(native_outputs.front());

  int output_size{0};
  auto *zero_copy_data = output_tensor->data<float>(&place, &output_size);
  auto *native_data = static_cast<float *>(native_outputs.front().data.data());
  for (size_t i = 0; i < output_size / sizeof(float); i++) {
    EXPECT_NEAR(zero_copy_data[i], native_data[i], 1e-3);
  }
}

TEST(Analyzer_rnn1, ZeroCopyMultiThread) {
  AnalysisConfig config;
  SetConfig(&config);
  config.use_feed_fetch_ops = false;

#define NEW_TENSOR(name__) \
  auto name__##_tensor = predictor->GetInputTensor(#name__);

  auto base_predictor = CreatePaddlePredictor<AnalysisConfig>(config);
  double total_time_of_threads{0};
  std::vector<std::thread> threads;
  std::vector<std::unique_ptr<PaddlePredictor>> predictors;
  for (int tid = 0; tid < FLAGS_num_threads; tid++) {
    predictors.emplace_back(CreatePaddlePredictor<AnalysisConfig>(config));
  }

  for (int tid = 0; tid < FLAGS_num_threads; tid++) {
    threads.emplace_back([config, &total_time_of_threads, &predictors, tid] {
      // auto predictor = base_predictor->Clone();
      auto &predictor = predictors[tid];
      NEW_TENSOR(data_lod_attention);
      NEW_TENSOR(cell_init);
      NEW_TENSOR(data);
      NEW_TENSOR(week);
      NEW_TENSOR(minute);
      NEW_TENSOR(hidden_init);

      // Prepare data for AnalysisPredictor
      DataRecord data(FLAGS_infer_data, FLAGS_batch_size);
      Timer timer;
      double total_time{0};

      for (int i = 0; i < FLAGS_repeat; i++) {
        PrepareZeroCopyInputs(data_lod_attention_tensor.get(),
                              cell_init_tensor.get(), data_tensor.get(),
                              hidden_init_tensor.get(), week_tensor.get(),
                              minute_tensor.get(), &data, FLAGS_batch_size);

        timer.tic();
        predictor->ZeroCopyRun();
        total_time += timer.toc();
      }

      total_time_of_threads += total_time;

      LOG(INFO) << "thread time: " << total_time / FLAGS_repeat;
    });
  }

  for (auto &t : threads) {
    t.join();
  }

  LOG(INFO) << "average time: "
            << total_time_of_threads / FLAGS_num_threads / FLAGS_repeat;
}

}  // namespace inference
}  // namespace paddle
