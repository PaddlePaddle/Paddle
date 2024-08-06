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
#pragma once
#include <string>
#include <utility>
#include <vector>

#include "test/cpp/inference/api/tester_helper.h"

namespace paddle {
namespace inference {
namespace analysis {
namespace transformer_tester {

struct DataRecord {
  std::vector<std::vector<int64_t>> src_word, src_pos, trg_word, init_idx;
  std::vector<std::vector<float>> src_slf_attn_bias, init_score,
      trg_src_attn_bias;
  std::vector<std::vector<int32_t>> batch_data_shape;
  std::vector<std::vector<size_t>> lod;
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
    if (batch_end <= src_word.size()) {
      data.src_word.assign(src_word.begin() + batch_iter,
                           src_word.begin() + batch_end);
      data.src_pos.assign(src_pos.begin() + batch_iter,
                          src_pos.begin() + batch_end);
      data.src_slf_attn_bias.assign(src_slf_attn_bias.begin() + batch_iter,
                                    src_slf_attn_bias.begin() + batch_end);
      data.trg_word.assign(trg_word.begin() + batch_iter,
                           trg_word.begin() + batch_end);
      data.init_score.assign(init_score.begin() + batch_iter,
                             init_score.begin() + batch_end);
      data.init_idx.assign(init_idx.begin() + batch_iter,
                           init_idx.begin() + batch_end);
      data.trg_src_attn_bias.assign(trg_src_attn_bias.begin() + batch_iter,
                                    trg_src_attn_bias.begin() + batch_end);
      std::vector<int32_t> batch_shape =
          *(batch_data_shape.begin() + batch_iter);
      data.batch_data_shape.push_back(batch_shape);
      data.lod.resize(2);
      for (int i = 0; i < batch_shape[0] + 1; i++) {
        data.lod[0].push_back(i);
        data.lod[1].push_back(i);
      }
    }
    batch_iter += batch_size;
    return data;
  }
  void Load(const std::string &path) {
    std::ifstream file(path);
    std::string line;
    size_t num_lines = 0;
    while (std::getline(file, line)) {
      num_lines++;
      std::vector<std::string> data;
      split(line, ',', &data);
      PADDLE_ENFORCE_EQ(data.size(),
                        static_cast<size_t>(8),
                        common::errors::InvalidArgument(
                            "The size of data should be euqal to 8. "));
      // load src_word
      std::vector<int64_t> src_word_data;
      split_to_int64(data[0], ' ', &src_word_data);
      src_word.push_back(std::move(src_word_data));
      // load src_pos
      std::vector<int64_t> src_pos_data;
      split_to_int64(data[1], ' ', &src_pos_data);
      src_pos.push_back(std::move(src_pos_data));
      // load src_slf_attn_bias
      std::vector<float> src_slf_attn_bias_data;
      split_to_float(data[2], ' ', &src_slf_attn_bias_data);
      src_slf_attn_bias.push_back(std::move(src_slf_attn_bias_data));
      // load trg_word
      std::vector<int64_t> trg_word_data;
      split_to_int64(data[3], ' ', &trg_word_data);
      trg_word.push_back(std::move(trg_word_data));
      // load init_score
      std::vector<float> init_score_data;
      split_to_float(data[4], ' ', &init_score_data);
      init_score.push_back(std::move(init_score_data));
      // load init_idx
      std::vector<int64_t> init_idx_data;
      split_to_int64(data[5], ' ', &init_idx_data);
      init_idx.push_back(std::move(init_idx_data));
      // load trg_src_attn_bias
      std::vector<float> trg_src_attn_bias_data;
      split_to_float(data[6], ' ', &trg_src_attn_bias_data);
      trg_src_attn_bias.push_back(std::move(trg_src_attn_bias_data));
      // load shape for variant data shape
      std::vector<int> batch_data_shape_data;
      split_to_int(data[7], ' ', &batch_data_shape_data);
      batch_data_shape.push_back(std::move(batch_data_shape_data));
    }
    num_samples = num_lines;
  }
};

void PrepareInputs(std::vector<PaddleTensor> *input_slots,
                   DataRecord *data,
                   int batch_size) {
  auto one_batch = data->NextBatch();
  batch_size = one_batch.batch_data_shape[0][0];
  auto n_head = one_batch.batch_data_shape[0][1];
  auto trg_seq_len = one_batch.batch_data_shape[0][2];  // 1 for inference
  auto src_seq_len = one_batch.batch_data_shape[0][3];

  PaddleTensor src_word, src_pos, src_slf_attn_bias, trg_word, init_score,
      init_idx, trg_src_attn_bias;

  src_word.name = "src_word";
  src_word.shape.assign({batch_size, src_seq_len, 1});
  src_word.dtype = PaddleDType::INT64;
  TensorAssignData<int64_t>(&src_word, one_batch.src_word);

  src_pos.name = "src_pos";
  src_pos.shape.assign({batch_size, src_seq_len, 1});
  src_pos.dtype = PaddleDType::INT64;
  TensorAssignData<int64_t>(&src_pos, one_batch.src_pos);

  src_slf_attn_bias.name = "src_slf_attn_bias";
  src_slf_attn_bias.shape.assign(
      {batch_size, n_head, src_seq_len, src_seq_len});
  src_slf_attn_bias.dtype = PaddleDType::FLOAT32;
  TensorAssignData<float>(&src_slf_attn_bias, one_batch.src_slf_attn_bias);

  trg_word.name = "trg_word";
  trg_word.shape.assign({batch_size, 1});
  trg_word.dtype = PaddleDType::INT64;
  trg_word.lod.assign(one_batch.lod.begin(), one_batch.lod.end());
  TensorAssignData<int64_t>(&trg_word, one_batch.trg_word);

  init_score.name = "init_score";
  init_score.shape.assign({batch_size, 1});
  init_score.dtype = PaddleDType::FLOAT32;
  init_score.lod.assign(one_batch.lod.begin(), one_batch.lod.end());
  TensorAssignData<float>(&init_score, one_batch.init_score);

  init_idx.name = "init_idx";
  init_idx.shape.assign({batch_size});
  init_idx.dtype = PaddleDType::INT64;
  TensorAssignData<int64_t>(&init_idx, one_batch.init_idx);

  trg_src_attn_bias.name = "trg_src_attn_bias";
  trg_src_attn_bias.shape.assign(
      {batch_size, n_head, trg_seq_len, src_seq_len});
  trg_src_attn_bias.dtype = PaddleDType::FLOAT32;
  TensorAssignData<float>(&trg_src_attn_bias, one_batch.trg_src_attn_bias);

  input_slots->assign({src_word,
                       src_pos,
                       src_slf_attn_bias,
                       trg_word,
                       init_score,
                       init_idx,
                       trg_src_attn_bias});
}

void SetConfig(AnalysisConfig *cfg) {
  cfg->SetModel(FLAGS_infer_model + "/model", FLAGS_infer_model + "/params");
  cfg->DisableGpu();
  cfg->SwitchSpecifyInputNames();
  cfg->SwitchIrOptim();
  cfg->SetCpuMathLibraryNumThreads(FLAGS_cpu_num_threads);
}

void SetInput(std::vector<std::vector<PaddleTensor>> *inputs) {
  DataRecord data(FLAGS_infer_data, FLAGS_batch_size);
  std::vector<PaddleTensor> input_slots;
  int test_batch_num =
      FLAGS_test_all_data ? data.num_samples / FLAGS_batch_size : 1;
  LOG(INFO) << "The number of samples to be test: "
            << test_batch_num * FLAGS_batch_size;
  for (int bid = 0; bid < test_batch_num; ++bid) {
    input_slots.clear();
    PrepareInputs(&input_slots, &data, FLAGS_batch_size);
    (*inputs).emplace_back(input_slots);
  }
}

}  // namespace transformer_tester
}  // namespace analysis
}  // namespace inference
}  // namespace paddle
