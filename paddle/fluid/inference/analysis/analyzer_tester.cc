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
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/inference/analysis/ut_helper.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"

DEFINE_string(infer_ditu_rnn_model, "", "model path for ditu RNN");
DEFINE_string(infer_ditu_rnn_data, "", "data path for ditu RNN");

namespace paddle {
namespace inference {
namespace analysis {

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
  DataRecord(const std::string &path, int batch_size = 1)
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
  // DataRecord data(FLAGS_datapath, batch_size);
  PaddleTensor lod_attention_tensor, init_zero_tensor, lod_tensor_tensor,
      week_tensor, minute_tensor;
  lod_attention_tensor.name = "lod_attention";
  init_zero_tensor.name = "init_zero";
  lod_tensor_tensor.name = "lod_tensor";
  week_tensor.name = "week";
  minute_tensor.name = "minute";
  auto one_batch = data->NextBatch();
  // clang-format off
  std::vector<int> rnn_link_data_shape
      ({static_cast<int>(one_batch.rnn_link_data.size()), static_cast<int>(one_batch.rnn_link_data.front().size())});
  lod_attention_tensor.shape.assign({1, 2});
  lod_attention_tensor.lod.assign({one_batch.lod1, one_batch.lod2});
  init_zero_tensor.shape.assign({batch_size, 15});
  init_zero_tensor.lod.assign({one_batch.lod3});
  lod_tensor_tensor.shape = rnn_link_data_shape;
  lod_tensor_tensor.lod.assign({one_batch.lod1});
  week_tensor.shape.assign({(int) one_batch.rnn_week_datas.size(), (int) one_batch.rnn_week_datas.front().size()});
  week_tensor.lod.assign({one_batch.lod3});
  minute_tensor.shape.assign({(int) one_batch.rnn_minute_datas.size(),
                              (int) one_batch.rnn_minute_datas.front().size()});
  minute_tensor.lod.assign({one_batch.lod3});
  // assign data
  LOG(INFO) << "to assian data";
  TensorAssignData(&lod_attention_tensor, std::vector<std::vector<float>>({{0, 0}}));
  std::vector<float> tmp_zeros(batch_size * 15, 0.);
  TensorAssignData(&init_zero_tensor, {tmp_zeros});
  TensorAssignData(&lod_tensor_tensor, one_batch.rnn_link_data);
  TensorAssignData(&week_tensor, one_batch.rnn_week_datas);
  TensorAssignData(&minute_tensor, one_batch.rnn_minute_datas);
  // clang-format on
  LOG(INFO) << "set input_slots";
  input_slots->assign({lod_tensor_tensor, lod_attention_tensor,
                       init_zero_tensor, init_zero_tensor, week_tensor,
                       minute_tensor});
  LOG(INFO) << "set type";
  for (auto &tensor : *input_slots) {
    tensor.dtype = PaddleDType::FLOAT32;
  }
}

}  // namespace

// Test with a really complicate model.
void TestDituRNNPrediction(const std::string &model_path,
                           const std::string &data_path, int batch_size,
                           bool activate_ir) {
  FLAGS_IA_enable_ir = activate_ir;
  FLAGS_IA_enable_tensorrt_subgraph_engine = false;
  FLAGS_IA_output_storage_path = "./analysis.out";

  Argument argument(model_path);
  argument.model_output_store_path.reset(new std::string("./analysis.out"));

  Analyzer analyzer;
  analyzer.Run(&argument);

  // Should get the transformed model stored to ./analysis.out
  std::string model_out = "./analysis.out";
  ASSERT_TRUE(PathExists(model_out));

  NativeConfig config;
  config.prog_file = model_out + "/__model__";
  config.param_file = model_out + "/param";
  config.use_gpu = false;
  config.device = 0;

  LOG(INFO) << "create predictor";
  auto predictor =
      CreatePaddlePredictor<NativeConfig, PaddleEngineKind::kNative>(config);
  std::vector<PaddleTensor> input_slots;
  LOG(INFO) << "open data";
  DataRecord data(data_path, batch_size);
  // Run multiple time to cancel the memory malloc or initialization of the
  // first time.
  // double whole_time = 0.;
  LOG(INFO) << "prepare input";
  PrepareInputs(&input_slots, &data, batch_size);
  std::vector<PaddleTensor> outputs;

  LOG(INFO) << "run";
  for (int i = 0; i < 1000; i++) {
    predictor->Run(input_slots, &outputs);
  }

  for (auto &out : outputs) {
    size_t size = std::accumulate(out.shape.begin(), out.shape.end(), 1,
                                  [](int a, int b) { return a * b; });
    float *data = static_cast<float *>(out.data.data());
    for (int i = 0; i < size; i++) {
      LOG(INFO) << data[i];
    }
  }
}

// Turn on the IR pass supportion, run a real inference and check the result.
TEST(Analyzer, SupportIRPass) {
  FLAGS_IA_enable_ir = true;
  FLAGS_IA_enable_tensorrt_subgraph_engine = false;
  FLAGS_IA_output_storage_path = "./analysis.out";

  Argument argument(FLAGS_inference_model_dir);
  argument.model_output_store_path.reset(new std::string("./analysis.out"));

  Analyzer analyzer;
  analyzer.Run(&argument);

  // Should get the transformed model stored to ./analysis.out
  ASSERT_TRUE(PathExists("./analysis.out"));

  // Inference from this path.
  TestWord2vecPrediction("./analysis.out");
}

TEST(Analyzer, DituRNN) {
  TestDituRNNPrediction(FLAGS_infer_ditu_rnn_model, FLAGS_infer_ditu_rnn_data,
                        1, false);
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle

USE_PASS(fc_fuse_pass);
USE_PASS(graph_viz_pass);
USE_PASS(infer_clean_graph_pass);
