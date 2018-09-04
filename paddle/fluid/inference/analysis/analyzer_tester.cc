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
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/inference/analysis/ut_helper.h"
#include "paddle/fluid/inference/api/analysis_predictor.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/api/paddle_inference_pass.h"
#include "paddle/fluid/inference/utils/singleton.h"
#include "paddle/fluid/platform/profiler.h"

DEFINE_string(infer_ditu_rnn_model, "", "model path for ditu RNN");
DEFINE_string(infer_ditu_rnn_data, "", "data path for ditu RNN");
DEFINE_int32(batch_size, 10, "batch size.");
DEFINE_int32(repeat, 1, "Running the inference program repeat times.");

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

std::string DescribeTensor(const PaddleTensor &tensor) {
  std::stringstream os;
  os << "Tensor [" << tensor.name << "]\n";
  os << " - type: ";
  switch (tensor.dtype) {
    case PaddleDType::FLOAT32:
      os << "float32";
      break;
    case PaddleDType::INT64:
      os << "int64";
      break;
    default:
      os << "unset";
  }
  os << '\n';

  os << " - shape: " << to_string(tensor.shape) << '\n';
  os << " - lod: ";
  for (auto &l : tensor.lod) {
    os << to_string(l) << "; ";
  }
  os << "\n";
  os << " - data: ";

  int dim = std::accumulate(tensor.shape.begin(), tensor.shape.end(), 1,
                            [](int a, int b) { return a * b; });
  for (int i = 0; i < dim; i++) {
    os << static_cast<float *>(tensor.data.data())[i] << " ";
  }
  os << '\n';
  return os.str();
}

}  // namespace

const float ditu_rnn_target_data[] = {
    104.711, 11.2431, 1.35422, 0,       0,       0,       0,       0,
    27.7039, 1.41486, 7.09526, 0,       0,       0,       0,       0,
    7.6481,  6.5324,  56.383,  2.88018, 8.92918, 132.007, 4.27429, 2.02934,
    14.1727, 10.7461, 25.0616, 16.0197, 14.4163, 16.9199, 6.75517, 0,
    80.0249, 4.77739, 0,       0,       0,       0,       0,       0,
    47.5643, 2.67029, 8.76252, 0,       0,       0,       0,       0,
    51.8822, 4.4411,  0,       0,       0,       0,       0,       0,
    10.7286, 12.0595, 10.6672, 0,       0,       0,       0,       0,
    93.5771, 3.84641, 0,       0,       0,       0,       0,       0,
    169.426, 0,       0,       0,       0,       0,       0,       0};
// Test with a really complicate model.
void TestDituRNNPrediction(const std::string &model_path,
                           const std::string &data_path, int batch_size,
                           bool use_analysis, bool activate_ir,
                           int num_times = 1) {
  NativeConfig config;
  config.prog_file = FLAGS_infer_ditu_rnn_model + "/__model__";
  config.param_file = FLAGS_infer_ditu_rnn_model + "/param";
  config.use_gpu = false;
  config.device = 0;
  config.specify_input_name = true;

  auto base_predictor =
      CreatePaddlePredictor<NativeConfig, PaddleEngineKind::kNative>(config);
  auto predictor =
      CreatePaddlePredictor<NativeConfig, PaddleEngineKind::kAnalysis>(config);
  std::vector<PaddleTensor> input_slots;
  DataRecord data(data_path, batch_size);
  // Prepare inputs.
  PrepareInputs(&input_slots, &data, batch_size);
  std::vector<PaddleTensor> outputs, base_outputs;

  base_predictor->Run(input_slots, &base_outputs);

  Timer timer;
  timer.tic();
  for (int i = 0; i < num_times; i++) {
    predictor->Run(input_slots, &outputs);
  }
  LOG(INFO) << "===========profile result===========";
  LOG(INFO) << "batch_size: " << batch_size << ", repeat: " << num_times
            << ", latency: " << timer.toc() / num_times << "ms";
  LOG(INFO) << "=====================================";

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
    for (size_t j = 0; j < size; j++) {
      EXPECT_NEAR(data[j], base_data[j], 1e-3);
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
    EXPECT_EQ(num_ops,
              13);  // After graph optimization, only 13 operators exists.
  }
}

// Directly infer with the original model.
TEST(Analyzer, DituRNN_without_analysis) {
  TestDituRNNPrediction(FLAGS_infer_ditu_rnn_model, FLAGS_infer_ditu_rnn_data,
                        FLAGS_batch_size, false, false, FLAGS_repeat);
}

// Inference with the original model with the analysis turned on, the analysis
// module will transform the program to a data flow graph.
TEST(Analyzer, DituRNN_with_analysis) {
  LOG(INFO) << "ditu rnn with analysis";
  TestDituRNNPrediction(FLAGS_infer_ditu_rnn_model, FLAGS_infer_ditu_rnn_data,
                        FLAGS_batch_size, true, false, FLAGS_repeat);
}

// Inference with analysis and IR. The IR module will fuse some large kernels.
TEST(Analyzer, DituRNN_with_analysis_with_IR) {
  LOG(INFO) << "ditu rnn with analysis and IR fuse";
  TestDituRNNPrediction(FLAGS_infer_ditu_rnn_model, FLAGS_infer_ditu_rnn_data,
                        FLAGS_batch_size, true, true, FLAGS_repeat);
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
