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

struct DataRecord {
  std::vector<std::vector<int64_t>> title1, title2, title3, l1;
  std::vector<size_t> lod1, lod2, lod3, l1_lod;
  size_t batch_iter{0}, batch_size{1}, num_samples;  // total number of samples
  DataRecord() = default;
  explicit DataRecord(const std::string &path, int batch_size = 1)
      : batch_size(batch_size) {
    Load(path);
  }
  DataRecord NextBatch() {
    DataRecord data;
    size_t batch_end = batch_iter + batch_size;
    // NOTE skip the final batch, if no enough data is provided.
    if (batch_end <= title1.size()) {
      GetInputPerBatch(title1, &data.title1, &data.lod1, batch_iter, batch_end);
      GetInputPerBatch(title2, &data.title2, &data.lod2, batch_iter, batch_end);
      GetInputPerBatch(title3, &data.title3, &data.lod3, batch_iter, batch_end);
      GetInputPerBatch(l1, &data.l1, &data.l1_lod, batch_iter, batch_end);
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
      split(line, '\t', &data);
      PADDLE_ENFORCE_GT(
          data.size(),
          4,
          paddle::platform::errors::Fatal("The size of data is invaild."));
      // load title1 data
      std::vector<int64_t> title1_data;
      split_to_int64(data[0], ' ', &title1_data);
      // load title2 data
      std::vector<int64_t> title2_data;
      split_to_int64(data[1], ' ', &title2_data);
      // load title3 data
      std::vector<int64_t> title3_data;
      split_to_int64(data[2], ' ', &title3_data);
      // load l1 data
      std::vector<int64_t> l1_data;
      split_to_int64(data[3], ' ', &l1_data);
      title1.push_back(std::move(title1_data));
      title2.push_back(std::move(title2_data));
      title3.push_back(std::move(title3_data));
      l1.push_back(std::move(l1_data));
    }
    num_samples = num_lines;
  }
};

void PrepareInputs(std::vector<PaddleTensor> *input_slots,
                   DataRecord *data,
                   int batch_size) {
  PaddleTensor title1_tensor, title2_tensor, title3_tensor, l1_tensor;
  title1_tensor.name = "title1";
  title2_tensor.name = "title2";
  title3_tensor.name = "title3";
  l1_tensor.name = "l1";
  auto one_batch = data->NextBatch();
  // assign data
  TensorAssignData<int64_t>(&title1_tensor, one_batch.title1, one_batch.lod1);
  TensorAssignData<int64_t>(&title2_tensor, one_batch.title2, one_batch.lod2);
  TensorAssignData<int64_t>(&title3_tensor, one_batch.title3, one_batch.lod3);
  TensorAssignData<int64_t>(&l1_tensor, one_batch.l1, one_batch.l1_lod);
  // Set inputs.
  input_slots->assign({title1_tensor, title2_tensor, title3_tensor, l1_tensor});
  for (auto &tensor : *input_slots) {
    tensor.dtype = PaddleDType::INT64;
  }
}

void SetConfig(AnalysisConfig *cfg) {
  cfg->SetModel(FLAGS_infer_model);
  cfg->DisableGpu();
  cfg->SwitchSpecifyInputNames();
  cfg->SwitchIrOptim();
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
TEST(Analyzer_seq_conv1, profile) {
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
    PADDLE_ENFORCE_GT(outputs.size(),
                      0,
                      paddle::platform::errors::Fatal(
                          "The size of output should be greater than 0."));
    auto output = outputs.back();
    PADDLE_ENFORCE_EQ(output.size(),
                      1UL,
                      paddle::platform::errors::Fatal(
                          "The size of output should be equal to 0."));
    size_t size = GetSize(output[0]);
    PADDLE_ENFORCE_GT(size,
                      0,
                      paddle::platform::errors::Fatal(
                          "The size of output should be greater than 0."));
    float *result = static_cast<float *>(output[0].data.data());
    // output is probability, which is in (0, 1).
    for (size_t i = 0; i < size; i++) {
      EXPECT_GT(result[i], 0);
      EXPECT_LT(result[i], 1);
    }
  }
}

// Check the fuse status
TEST(Analyzer_seq_conv1, fuse_statis) {
  AnalysisConfig cfg;
  SetConfig(&cfg);
  int num_ops;
  auto predictor = CreatePaddlePredictor<AnalysisConfig>(cfg);

  auto fuse_statis = GetFuseStatis(predictor.get(), &num_ops);
  ASSERT_TRUE(fuse_statis.count("fc_fuse"));
  ASSERT_TRUE(fuse_statis.count("seqconv_eltadd_relu_fuse"));
  EXPECT_EQ(fuse_statis.at("fc_fuse"), 2);
  EXPECT_EQ(fuse_statis.at("seqconv_eltadd_relu_fuse"), 6);
}

// Compare result of NativeConfig and AnalysisConfig
TEST(Analyzer_seq_conv1, compare) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  CompareNativeAndAnalysis(
      reinterpret_cast<const PaddlePredictor::Config *>(&cfg), input_slots_all);
}

// Compare Deterministic result
TEST(Analyzer_seq_conv1, compare_determine) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  CompareDeterministic(reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
                       input_slots_all);
}

}  // namespace inference
}  // namespace paddle
