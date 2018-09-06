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
#include <gflags/gflags.h>
#include <glog/logging.h>  // use glog instead of PADDLE_ENFORCE to avoid importing other paddle header files.
#include <gtest/gtest.h>
#include <fstream>
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/inference/analysis/ut_helper.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/api/paddle_inference_pass.h"
#include "paddle/fluid/inference/api/timer.h"

DEFINE_string(infer_model, "", "Directory of the inference model.");
DEFINE_string(infer_data, "", "Path of the dataset.");
DEFINE_int32(batch_size, 1, "batch size.");
DEFINE_int32(repeat, 1, "How many times to repeat run.");
DEFINE_int32(topn, -1, "Run top n batches of data to save time");

namespace paddle {
namespace inference {

struct DataReader {
  explicit DataReader(const std::string &path)
      : file(new std::ifstream(path)) {}

  bool NextBatch(PaddleTensor *tensor, int batch_size) {
    PADDLE_ENFORCE_EQ(batch_size, 1);
    std::string line;
    tensor->lod.clear();
    tensor->lod.emplace_back(std::vector<size_t>({0}));
    std::vector<int64_t> data;

    for (int i = 0; i < batch_size; i++) {
      if (!std::getline(*file, line)) return false;
      inference::split_to_int64(line, ' ', &data);
    }
    tensor->lod.front().push_back(data.size());

    tensor->data.Resize(data.size() * sizeof(int64_t));
    memcpy(tensor->data.data(), data.data(), data.size() * sizeof(int64_t));
    tensor->shape.clear();
    tensor->shape.push_back(data.size());
    tensor->shape.push_back(1);
    return true;
  }

  std::unique_ptr<std::ifstream> file;
};

void Main(int batch_size) {
  // shape --
  // Create Predictor --
  AnalysisConfig config;
  config.model_dir = FLAGS_infer_model;
  config.use_gpu = false;
  config.enable_ir_optim = true;
  auto predictor =
      CreatePaddlePredictor<AnalysisConfig, PaddleEngineKind::kAnalysis>(
          config);

  std::vector<PaddleTensor> input_slots(1);
  // one batch starts
  // data --
  auto &input = input_slots[0];
  input.dtype = PaddleDType::INT64;

  inference::Timer timer;
  double sum = 0;
  std::vector<PaddleTensor> output_slots;

  int num_batches = 0;
  for (int t = 0; t < FLAGS_repeat; t++) {
    DataReader reader(FLAGS_infer_data);
    while (reader.NextBatch(&input, FLAGS_batch_size)) {
      if (FLAGS_topn > 0 && num_batches > FLAGS_topn) break;
      timer.tic();
      CHECK(predictor->Run(input_slots, &output_slots));
      sum += timer.toc();
      ++num_batches;
    }
  }
  PrintTime(batch_size, FLAGS_repeat, 1, 0, sum / FLAGS_repeat);

  // Get output
  LOG(INFO) << "get outputs " << output_slots.size();

  for (auto &output : output_slots) {
    LOG(INFO) << "output.shape: " << to_string(output.shape);
    // no lod ?
    CHECK_EQ(output.lod.size(), 0UL);
    LOG(INFO) << "output.dtype: " << output.dtype;
    std::stringstream ss;
    for (int i = 0; i < 5; i++) {
      ss << static_cast<float *>(output.data.data())[i] << " ";
    }
    LOG(INFO) << "output.data summary: " << ss.str();
    // one batch ends
  }
}

TEST(text_classification, basic) { Main(FLAGS_batch_size); }

}  // namespace inference
}  // namespace paddle
