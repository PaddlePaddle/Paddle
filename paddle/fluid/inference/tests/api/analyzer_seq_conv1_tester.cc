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
  std::vector<std::vector<int64_t>> title1_all, title2_all, title3_all, l1_all;
  std::vector<std::vector<int64_t>> title1, title2, title3, l1;
  std::vector<size_t> title1_lod, title2_lod, title3_lod, l1_lod;
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
    if (batch_end <= title1_all.size()) {
      data.title1_all.assign(title1_all.begin() + batch_iter,
                             title1_all.begin() + batch_end);
      data.title2_all.assign(title2_all.begin() + batch_iter,
                             title2_all.begin() + batch_end);
      data.title3_all.assign(title3_all.begin() + batch_iter,
                             title3_all.begin() + batch_end);
      data.l1_all.assign(l1_all.begin() + batch_iter,
                         l1_all.begin() + batch_end);
      // Prepare LoDs
      data.title1_lod.push_back(0);
      data.title2_lod.push_back(0);
      data.title3_lod.push_back(0);
      data.l1_lod.push_back(0);
      CHECK(!data.title1_all.empty());
      CHECK(!data.title2_all.empty());
      CHECK(!data.title3_all.empty());
      CHECK(!data.l1_all.empty());
      CHECK_EQ(data.title1_all.size(), data.title2_all.size());
      CHECK_EQ(data.title1_all.size(), data.title3_all.size());
      CHECK_EQ(data.title1_all.size(), data.l1_all.size());
      for (size_t j = 0; j < data.title1_all.size(); j++) {
        data.title1.push_back(data.title1_all[j]);
        data.title2.push_back(data.title2_all[j]);
        data.title3.push_back(data.title3_all[j]);
        data.l1.push_back(data.l1_all[j]);
        // calculate lod
        data.title1_lod.push_back(data.title1_lod.back() +
                                  data.title1_all[j].size());
        data.title2_lod.push_back(data.title2_lod.back() +
                                  data.title2_all[j].size());
        data.title3_lod.push_back(data.title3_lod.back() +
                                  data.title3_all[j].size());
        data.l1_lod.push_back(data.l1_lod.back() + data.l1_all[j].size());
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
      split(line, '\t', &data);
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
      title1_all.push_back(std::move(title1_data));
      title2_all.push_back(std::move(title2_data));
      title3_all.push_back(std::move(title3_data));
      l1_all.push_back(std::move(l1_data));
    }
    num_samples = num_lines;
  }
};

void PrepareInputs(std::vector<PaddleTensor> *input_slots, DataRecord *data,
                   int batch_size) {
  PaddleTensor title1_tensor, title2_tensor, title3_tensor, l1_tensor;
  title1_tensor.name = "title1";
  title2_tensor.name = "title2";
  title3_tensor.name = "title3";
  l1_tensor.name = "l1";
  auto one_batch = data->NextBatch();
  int title1_size = one_batch.title1_lod[one_batch.title1_lod.size() - 1];
  title1_tensor.shape.assign({title1_size, 1});
  title1_tensor.lod.assign({one_batch.title1_lod});
  int title2_size = one_batch.title2_lod[one_batch.title2_lod.size() - 1];
  title2_tensor.shape.assign({title2_size, 1});
  title2_tensor.lod.assign({one_batch.title2_lod});
  int title3_size = one_batch.title3_lod[one_batch.title3_lod.size() - 1];
  title3_tensor.shape.assign({title3_size, 1});
  title3_tensor.lod.assign({one_batch.title3_lod});
  int l1_size = one_batch.l1_lod[one_batch.l1_lod.size() - 1];
  l1_tensor.shape.assign({l1_size, 1});
  l1_tensor.lod.assign({one_batch.l1_lod});

  // assign data
  TensorAssignData<int64_t>(&title1_tensor, one_batch.title1);
  TensorAssignData<int64_t>(&title2_tensor, one_batch.title2);
  TensorAssignData<int64_t>(&title3_tensor, one_batch.title3);
  TensorAssignData<int64_t>(&l1_tensor, one_batch.l1);
  // Set inputs.
  input_slots->assign({title1_tensor, title2_tensor, title3_tensor, l1_tensor});
  for (auto &tensor : *input_slots) {
    tensor.dtype = PaddleDType::INT64;
  }
}

void SetConfig(AnalysisConfig *cfg) {
  cfg->model_dir = FLAGS_infer_model;
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
TEST(Analyzer_seq_conv1, profile) {
  AnalysisConfig cfg;
  SetConfig(&cfg);
  std::vector<PaddleTensor> outputs;

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  TestPrediction(reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
                 input_slots_all, &outputs, FLAGS_num_threads);

  if (FLAGS_num_threads == 1 && !FLAGS_test_all_data) {
    // the first inference result
    PADDLE_ENFORCE_EQ(outputs.size(), 1UL);
    size_t size = GetSize(outputs[0]);
    PADDLE_ENFORCE_GT(size, 0);
    float *result = static_cast<float *>(outputs[0].data.data());
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
  EXPECT_EQ(num_ops, 32);
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

}  // namespace inference
}  // namespace paddle
