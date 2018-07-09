/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

/*
 * This file contains demo for mobilenet, se-resnext50 and ocr.
 */

#include <gflags/gflags.h>
#include <glog/logging.h>  // use glog instead of PADDLE_ENFORCE to avoid importing other paddle header files.
#include <gtest/gtest.h>
#include <fstream>
#include <iostream>
#include "paddle/contrib/inference/demo/utils.h"
#include "paddle/contrib/inference/paddle_inference_api.h"

#ifdef PADDLE_WITH_CUDA
DECLARE_double(fraction_of_gpu_memory_to_use);
#endif

namespace paddle {
namespace demo {

DEFINE_string(modeldir, "", "Directory of the inference model.");
DEFINE_string(refer, "", "path to reference result for comparison.");
DEFINE_string(
    data,
    "",
    "path of data; each line is a record, format is "
    "'<space splitted floats as data>\t<space splitted ints as shape'");

struct Record {
  std::vector<float> data;
  std::vector<int32_t> shape;
};

void split(const std::string& str, char sep, std::vector<std::string>* pieces);

Record ProcessALine(const std::string& line) {
  LOG(INFO) << "process a line";
  std::vector<std::string> columns;
  split(line, '\t', &columns);
  CHECK_EQ(columns.size(), 2UL)
      << "data format error, should be <data>\t<shape>";

  Record record;
  std::vector<std::string> data_strs;
  split(columns[0], ' ', &data_strs);
  for (auto& d : data_strs) {
    record.data.push_back(std::stof(d));
  }

  std::vector<std::string> shape_strs;
  split(columns[1], ' ', &shape_strs);
  for (auto& s : shape_strs) {
    record.shape.push_back(std::stoi(s));
  }
  LOG(INFO) << "data size " << record.data.size();
  LOG(INFO) << "data shape size " << record.shape.size();
  return record;
}

void CheckOutput(const std::string& referfile, const PaddleTensor& output) {
  std::string line;
  std::ifstream file(referfile);
  std::getline(file, line);
  auto refer = ProcessALine(line);
  file.close();

  size_t numel = output.data.length() / PaddleDtypeSize(output.dtype);
  LOG(INFO) << "predictor output numel " << numel;
  LOG(INFO) << "reference output numel " << refer.data.size();
  EXPECT_EQ(numel, refer.data.size());
  switch (output.dtype) {
    case PaddleDType::INT64: {
      for (size_t i = 0; i < numel; ++i) {
        EXPECT_EQ(static_cast<int64_t*>(output.data.data())[i], refer.data[i]);
      }
      break;
    }
    case PaddleDType::FLOAT32:
      for (size_t i = 0; i < numel; ++i) {
        EXPECT_NEAR(
            static_cast<float*>(output.data.data())[i], refer.data[i], 1e-5);
      }
      break;
  }
}

/*
 * Use the native fluid engine to inference the demo.
 */
void Main(bool use_gpu) {
  NativeConfig config;
  config.param_file = FLAGS_modeldir + "/__params__";
  config.prog_file = FLAGS_modeldir + "/__model__";
  config.use_gpu = use_gpu;
  config.device = 0;
#ifdef PADDLE_WITH_CUDA
  config.fraction_of_gpu_memory = FLAGS_fraction_of_gpu_memory_to_use;
#endif

  LOG(INFO) << "init predictor";
  auto predictor =
      CreatePaddlePredictor<NativeConfig, PaddleEngineKind::kNative>(config);

  LOG(INFO) << "begin to process data";
  // Just a single batch of data.
  std::string line;
  std::ifstream file(FLAGS_data);
  std::getline(file, line);
  auto record = ProcessALine(line);
  file.close();

  // Inference.
  PaddleTensor input{
      .name = "xx",
      .shape = record.shape,
      .data = PaddleBuf(record.data.data(), record.data.size() * sizeof(float)),
      .dtype = PaddleDType::FLOAT32};

  LOG(INFO) << "run executor";
  std::vector<PaddleTensor> output;
  predictor->Run({input}, &output);

  LOG(INFO) << "output.size " << output.size();
  auto& tensor = output.front();
  LOG(INFO) << "output: " << SummaryTensor(tensor);

  // compare with reference result
  CheckOutput(FLAGS_refer, tensor);
}

TEST(demo, vis_demo_cpu) { Main(false /*use_gpu*/); }
#ifdef PADDLE_WITH_CUDA
TEST(demo, vis_demo_gpu) { Main(true /*use_gpu*/); }
#endif
}  // namespace demo
}  // namespace paddle
