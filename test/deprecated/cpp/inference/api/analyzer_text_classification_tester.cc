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

struct DataReader {
  explicit DataReader(const std::string &path)
      : file(new std::ifstream(path)) {}

  bool NextBatch(std::vector<PaddleTensor> *input, int batch_size) {
    PADDLE_ENFORCE_EQ(
        batch_size,
        1,
        common::errors::Fatal("The size of batch should be equal to 1."));
    std::string line;
    PaddleTensor tensor;
    tensor.dtype = PaddleDType::INT64;
    tensor.lod.emplace_back(std::vector<size_t>({0}));
    std::vector<int64_t> data;

    for (int i = 0; i < batch_size; i++) {
      if (!std::getline(*file, line)) return false;
      inference::split_to_int64(line, ' ', &data);
    }
    tensor.lod.front().push_back(data.size());

    tensor.data.Resize(data.size() * sizeof(int64_t));
    PADDLE_ENFORCE_NE(
        tensor.data.data(),
        nullptr,
        common::errors::Fatal("Variable `tensor.data.data()` is nullptr"));
    PADDLE_ENFORCE_NE(
        data.data(),
        nullptr,
        common::errors::Fatal("Variable `data.data()` is nullptr"));
    memcpy(tensor.data.data(), data.data(), data.size() * sizeof(int64_t));
    tensor.shape.push_back(data.size());
    tensor.shape.push_back(1);
    input->assign({tensor});
    return true;
  }

  std::unique_ptr<std::ifstream> file = nullptr;
};

void SetConfig(AnalysisConfig *cfg) {
  cfg->SetModel(FLAGS_infer_model);
  cfg->DisableGpu();
  cfg->SwitchSpecifyInputNames();
  cfg->SwitchIrOptim();
}

void SetInput(std::vector<std::vector<PaddleTensor>> *inputs) {
  std::vector<PaddleTensor> input_slots;
  DataReader reader(FLAGS_infer_data);
  int num_batches = 0;
  while (reader.NextBatch(&input_slots, FLAGS_batch_size)) {
    (*inputs).emplace_back(input_slots);
    ++num_batches;
    if (!FLAGS_test_all_data) return;
  }
  LOG(INFO) << "total number of samples: " << num_batches * FLAGS_batch_size;
}

// Easy for profiling independently.
TEST(Analyzer_Text_Classification, profile) {
  AnalysisConfig cfg;
  SetConfig(&cfg);
  cfg.SwitchIrDebug();
  std::vector<std::vector<PaddleTensor>> outputs;

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  TestPrediction(reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
                 input_slots_all,
                 &outputs,
                 FLAGS_num_threads);

  if (FLAGS_num_threads == 1) {
    // Get output
    PADDLE_ENFORCE_GT(
        outputs.size(),
        0,
        common::errors::Fatal("The size of output should be greater than 0."));
    LOG(INFO) << "get outputs " << outputs.back().size();
    for (auto &output : outputs.back()) {
      LOG(INFO) << "output.shape: " << to_string(output.shape);
      // no lod ?
      PADDLE_ENFORCE_EQ(
          output.lod.size(),
          0UL,
          common::errors::InvalidArgument(
              "The 'lod' size of 'output' should be 0, but received size %d.",
              output.lod.size()));
      LOG(INFO) << "output.dtype: " << output.dtype;
      std::stringstream ss;
      int num_data = 1;
      for (auto i : output.shape) {
        num_data *= i;
      }

      for (int i = 0; i < num_data; i++) {
        ss << static_cast<float *>(output.data.data())[i] << " ";
      }
      LOG(INFO) << "output.data summary: " << ss.str();
      // one batch ends
    }
  }
}

// Compare result of NativeConfig and AnalysisConfig
TEST(Analyzer_Text_Classification, compare) {
  AnalysisConfig cfg;
  SetConfig(&cfg);
  cfg.EnableMemoryOptim();

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  CompareNativeAndAnalysis(
      reinterpret_cast<const PaddlePredictor::Config *>(&cfg), input_slots_all);
}

// Compare Deterministic result
TEST(Analyzer_Text_Classification, compare_determine) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  CompareDeterministic(reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
                       input_slots_all);
}

TEST(Analyzer_Text_Classification, compare_against_embedding_fc_lstm_fused) {
  AnalysisConfig cfg;
  SetConfig(&cfg);
  // Enable embedding_fc_lstm_fuse_pass (disabled by default)
  cfg.pass_builder()->InsertPass(2, "embedding_fc_lstm_fuse_pass");

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  CompareNativeAndAnalysis(
      reinterpret_cast<const PaddlePredictor::Config *>(&cfg), input_slots_all);
}

}  // namespace inference
}  // namespace paddle
