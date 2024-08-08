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

using namespace framework;  // NOLINT
static std::vector<float> result_data;

struct DataRecord {
  std::vector<std::vector<std::vector<float>>> link_step_data_all;
  std::vector<size_t> lod;
  std::vector<std::vector<float>> rnn_link_data;
  size_t num_samples;  // total number of samples
  size_t batch_iter{0};
  size_t batch_size{1};
  DataRecord() : link_step_data_all(), lod(), rnn_link_data(), num_samples(0) {}
  explicit DataRecord(const std::string &path, int batch_size = 1)
      : link_step_data_all(),
        lod(),
        rnn_link_data(),
        num_samples(0),
        batch_size(batch_size) {
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
      PADDLE_ENFORCE_EQ(
          !data.link_step_data_all.empty(),
          true,
          common::errors::InvalidArgument(
              "`data.link_step_data_all` is empty, please check"));
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
    result_data.clear();
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
    num_samples = num_lines / 2;
  }
};
void PrepareInputs(std::vector<PaddleTensor> *input_slots,
                   DataRecord *data,
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

void SetConfig(AnalysisConfig *cfg) {
  cfg->SetModel(FLAGS_infer_model + "/__model__", FLAGS_infer_model + "/param");
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
    PrepareInputs(&input_slots, &data, FLAGS_batch_size);
    (*inputs).emplace_back(input_slots);
  }
}

// Easy for profiling independently.
TEST(Analyzer_rnn2, profile) {
  AnalysisConfig cfg;
  SetConfig(&cfg);
  std::vector<std::vector<PaddleTensor>> outputs;

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  TestPrediction(reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
                 input_slots_all,
                 &outputs,
                 FLAGS_num_threads);

  if (FLAGS_num_threads == 1 && !FLAGS_test_all_data) {
    // the first inference result
    PADDLE_ENFORCE_GT(
        outputs.size(),
        0,
        common::errors::Fatal("The size of output should be greater than 0."));
    auto output = outputs.back();
    PADDLE_ENFORCE_GT(
        output.size(),
        0,
        common::errors::Fatal("The size of output should be greater than 0."));
    size_t size = GetSize(output[0]);
    PADDLE_ENFORCE_GT(
        size,
        0,
        common::errors::Fatal("The size of output should be greater than 0."));
    float *result = static_cast<float *>(output[0].data.data());
    for (size_t i = 0; i < size; i++) {
      EXPECT_NEAR(result[i], result_data[i], 1e-3);
    }
  }
}

// Compare result of NativeConfig and AnalysisConfig
TEST(Analyzer_rnn2, compare) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  CompareNativeAndAnalysis(
      reinterpret_cast<const PaddlePredictor::Config *>(&cfg), input_slots_all);
}

// Compare Deterministic result
TEST(Analyzer_rnn2, compare_determine) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  CompareDeterministic(reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
                       input_slots_all);
}

}  // namespace inference
}  // namespace paddle
