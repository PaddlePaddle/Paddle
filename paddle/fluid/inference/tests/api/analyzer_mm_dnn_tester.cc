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
  std::vector<std::vector<int64_t>> query, title;
  std::vector<size_t> lod1, lod2;
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
    if (batch_end <= query.size()) {
      GetInputPerBatch(query, &data.query, &data.lod1, batch_iter, batch_end);
      GetInputPerBatch(title, &data.title, &data.lod2, batch_iter, batch_end);
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
      // load query data
      std::vector<int64_t> query_data;
      split_to_int64(data[0], ' ', &query_data);
      // load title data
      std::vector<int64_t> title_data;
      split_to_int64(data[1], ' ', &title_data);
      query.push_back(std::move(query_data));
      title.push_back(std::move(title_data));
    }
    num_samples = num_lines;
  }
};

void PrepareInputs(std::vector<PaddleTensor> *input_slots, DataRecord *data,
                   int batch_size) {
  PaddleTensor lod_query_tensor, lod_title_tensor;
  lod_query_tensor.name = "left";
  lod_title_tensor.name = "right";
  auto one_batch = data->NextBatch();
  // assign data
  TensorAssignData<int64_t>(&lod_query_tensor, one_batch.query, one_batch.lod1);
  TensorAssignData<int64_t>(&lod_title_tensor, one_batch.title, one_batch.lod2);
  // Set inputs.
  input_slots->assign({lod_query_tensor, lod_title_tensor});
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
void profile(bool use_mkldnn = false) {
  AnalysisConfig cfg;
  SetConfig(&cfg);
  std::vector<PaddleTensor> outputs;

  if (use_mkldnn) {
    cfg.EnableMKLDNN();
  }

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  TestPrediction(reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
                 input_slots_all, &outputs, FLAGS_num_threads);

  if (FLAGS_num_threads == 1 && !FLAGS_test_all_data) {
    PADDLE_ENFORCE_EQ(outputs.size(), 2UL);
    for (auto &output : outputs) {
      size_t size = GetSize(output);
      PADDLE_ENFORCE_GT(size, 0);
      float *result = static_cast<float *>(output.data.data());
      // output is probability, which is in (-1, 1).
      for (size_t i = 0; i < size; i++) {
        EXPECT_GT(result[i], -1);
        EXPECT_LT(result[i], 1);
      }
    }
  }
}

TEST(Analyzer_MM_DNN, profile) { profile(); }
#ifdef PADDLE_WITH_MKLDNN
TEST(Analyzer_MM_DNN, profile_mkldnn) { profile(true /* use_mkldnn */); }
#endif

// Check the fuse status
TEST(Analyzer_MM_DNN, fuse_statis) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  int num_ops;
  auto predictor = CreatePaddlePredictor<AnalysisConfig>(cfg);
  auto fuse_statis = GetFuseStatis(
      static_cast<AnalysisPredictor *>(predictor.get()), &num_ops);
}

// Compare result of NativeConfig and AnalysisConfig
void compare(bool use_mkldnn = false) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  if (use_mkldnn) {
    cfg.EnableMKLDNN();
  }

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  CompareNativeAndAnalysis(
      reinterpret_cast<const PaddlePredictor::Config *>(&cfg), input_slots_all);
}

TEST(Analyzer_MM_DNN, compare) { compare(); }
#ifdef PADDLE_WITH_MKLDNN
TEST(Analyzer_MM_DNN, compare_mkldnn) { compare(true /* use_mkldnn */); }
#endif

// Compare Deterministic result
TEST(Analyzer_MM_DNN, compare_determine) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  CompareDeterministic(reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
                       input_slots_all);
}

}  // namespace inference
}  // namespace paddle
