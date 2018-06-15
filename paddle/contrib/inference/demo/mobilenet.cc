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
 * This file contains a demo for mobilenet.
 * TODO(Superjomn) add some links of the actual models.
 */

#include <gflags/gflags.h>
#include <glog/logging.h>  // use glog instead of PADDLE_ENFORCE to avoid importing other paddle header files.
#include <gtest/gtest.h>
#include <iostream>
#include "paddle/contrib/inference/paddle_inference_api.h"
#include "paddle/utils/StringUtil.h"

namespace paddle {
namespace demo {

DEFINE_string(modeldir, "", "Directory of the inference model.");
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
  std::vector<std::string> columns;
  split(line, '\t', &columns);
  CHECK_EQ(columns.size(), 2UL)
      << "data format error, should be <data>\t<shape>";

  Record record;

  std::vector<std::string> data_strs;
  split(line, ' ', &data_strs);
  for (auto& d : data_strs) {
    record.data.push_back(std::stof(d));
  }

  std::vector<std::string> shape_strs;
  split(line, ' ', &shape_strs);
  for (auto& s : shape_strs) {
    record.shape.push_back(std::stoi(s));
  }
  return record;
}


/*
 * Use the native fluid engine to inference the mobilenet.
 */
void Main(bool use_gpu) {
  NativeConfig config;
  config.model_dir = FLAGS_modeldir;
  config.use_gpu = use_gpu;
  config.fraction_of_gpu_memory = 0.15;
  config.device = 0;

  auto predictor =
      CreatePaddlePredictor<NativeConfig, PaddleEngineKind::kNative>(config);

  // Just a single batch of data.
  std::string line;
  std::getline(std::cin, line);
  auto record = ProcessALine(line);

  // Inference.
  PaddleBuf buf{.data = record.data.data(),
                .length = record.data.size() * sizeof(float)};
  PaddleTensor input{.name = "xx",
                     .shape = record.shape,
                     .data = buf,
                     .dtype = PaddleDType::FLOAT32};

  std::vector<PaddleTensor> output;
  predictor->Run({input}, &output);
}

TEST(demo, mobilenet) { Main(false /*use_gpu*/); }

void split(const std::string& str, char sep, std::vector<std::string>* pieces) {
  pieces->clear();
  if (str.empty()) {
    return;
  }
  size_t pos = 0;
  size_t next = str.find(sep, pos);
  while (next != std::string::npos) {
    pieces->push_back(str.substr(pos, next - pos));
    pos = next + 1;
    next = str.find(sep, pos);
  }
  if (!str.substr(pos).empty()) {
    pieces->push_back(str.substr(pos));
  }
}

}  // namespace demo
}  // namespace paddle
