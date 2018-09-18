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

DEFINE_string(infer_model, "", "model path");
DEFINE_string(infer_data, "", "data path");
DEFINE_int32(batch_size, 1, "batch size.");
DEFINE_int32(repeat, 1, "Running the inference program repeat times.");
DEFINE_int32(num_threads, 1, "Running the inference program in multi-threads.");

namespace paddle {
namespace inference {

using namespace framework;  // NOLINT

struct DataRecord {
  std::vector<std::vector<std::vector<float>>> link_step_data_all;
  std::vector<size_t> lod;
  std::vector<std::vector<float>> rnn_link_data;
  std::vector<float> result_data;
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
      // Prepare LoDs
      data.lod.push_back(0);
      CHECK(!data.link_step_data_all.empty()) << "empty";
      for (size_t j = 0; j < data.link_step_data_all.size(); j++) {
        for (const auto &d : data.link_step_data_all[j]) {
          data.rnn_link_data.push_back(d);
          // calculate lod
          data.lod.push_back(data.lod.back() + 11);
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
      if (num_lines % 2) {  // feature
        std::vector<std::string> feature_data;
        split(data[1], ' ', &feature_data);
        std::vector<std::vector<float>> link_step_data;
        int feature_count = 1;
        std::vector<float> feature;
        for (auto &step_data : feature_data) {
          std::vector<float> tmp;
          split_to_float(step_data, ',', &tmp);
          feature.insert(feature.end(), tmp.begin(), tmp.end());
          if (feature_count % 11 == 0) {  // each sample has 11 features
            link_step_data.push_back(feature);
            feature.clear();
          }
          feature_count++;
        }
        link_step_data_all.push_back(std::move(link_step_data));
      } else {  // result
        std::vector<float> tmp;
        split_to_float(data[1], ',', &tmp);
        result_data.insert(result_data.end(), tmp.begin(), tmp.end());
      }
    }
  }
};
void PrepareInputs(std::vector<PaddleTensor> *input_slots, DataRecord *data,
                   int batch_size) {
  PaddleTensor feed_tensor;
  feed_tensor.name = "feed";
  auto one_batch = data->NextBatch();
  int token_size = one_batch.rnn_link_data.size();
  // each token has 11 features, each feature's dim is 54.
  std::vector<int> rnn_link_data_shape({token_size * 11, 54});
  feed_tensor.shape = rnn_link_data_shape;
  feed_tensor.lod.assign({one_batch.lod});
  feed_tensor.dtype = PaddleDType::FLOAT32;
  TensorAssignData<float>(&feed_tensor, one_batch.rnn_link_data);
  // Set inputs.
  input_slots->assign({feed_tensor});
}

void CompareResult(const std::vector<PaddleTensor> &outputs,
                   const std::vector<float> &base_result) {
  PADDLE_ENFORCE_GT(outputs.size(), 0);
  for (size_t i = 0; i < outputs.size(); i++) {
    auto &out = outputs[i];
    size_t size = std::accumulate(out.shape.begin(), out.shape.end(), 1,
                                  [](int a, int b) { return a * b; });
    PADDLE_ENFORCE_GT(size, 0);
    float *data = static_cast<float *>(out.data.data());
    for (size_t i = 0; i < size; i++) {
      EXPECT_NEAR(data[i], base_result[i], 1e-3);
    }
  }
}
// Test with a really complicate model.
void TestRNN2Prediction() {
  AnalysisConfig config;
  config.prog_file = FLAGS_infer_model + "/__model__";
  config.param_file = FLAGS_infer_model + "/param";
  config.use_gpu = false;
  config.device = 0;
  config.specify_input_name = true;
  config.enable_ir_optim = true;
  PADDLE_ENFORCE(config.ir_mode ==
                 AnalysisConfig::IrPassMode::kExclude);  // default

  int batch_size = FLAGS_batch_size;
  int num_times = FLAGS_repeat;

  auto base_predictor =
      CreatePaddlePredictor<NativeConfig, PaddleEngineKind::kNative>(config);
  auto predictor =
      CreatePaddlePredictor<AnalysisConfig, PaddleEngineKind::kAnalysis>(
          config);
  std::vector<PaddleTensor> input_slots;
  DataRecord data(FLAGS_infer_data, batch_size);
  PrepareInputs(&input_slots, &data, batch_size);
  std::vector<PaddleTensor> outputs, base_outputs;

  Timer timer1;
  timer1.tic();
  for (int i = 0; i < num_times; i++) {
    base_predictor->Run(input_slots, &base_outputs);
  }
  PrintTime(batch_size, num_times, 1, 0, timer1.toc() / num_times);

  Timer timer2;
  timer2.tic();
  for (int i = 0; i < num_times; i++) {
    predictor->Run(input_slots, &outputs);
  }
  PrintTime(batch_size, num_times, 1, 0, timer2.toc() / num_times);

  CompareResult(base_outputs, data.result_data);
  CompareResult(outputs, data.result_data);
}

TEST(Analyzer, rnn2) { TestRNN2Prediction(); }

}  // namespace inference
}  // namespace paddle
