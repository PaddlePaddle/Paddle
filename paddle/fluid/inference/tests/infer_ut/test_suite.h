// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <math.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

#include "paddle/include/paddle_inference_api.h"

namespace paddle {
namespace test {

class Record {
 public:
  std::vector<float> data;
  std::vector<int32_t> shape;
  paddle::PaddleDType type;
  int label;
};

std::string read_file(std::string filename) {
  std::ifstream file(filename);
  return std::string((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
}

void SingleThreadPrediction(paddle_infer::Predictor *predictor,
                            std::map<std::string, Record> *input_data_map,
                            std::map<std::string, Record> *output_data_map,
                            int repeat_times = 2) {
  // prepare input tensor
  auto input_names = predictor->GetInputNames();
  for (const auto & [ key, value ] : *input_data_map) {
    switch (value.type) {
      case paddle::PaddleDType::INT64: {
        std::vector<int64_t> input_value =
            std::vector<int64_t>(value.data.begin(), value.data.end());
        auto input_tensor = predictor->GetInputHandle(key);
        input_tensor->Reshape(value.shape);
        input_tensor->CopyFromCpu(input_value.data());
        break;
      }
      case paddle::PaddleDType::INT32: {
        std::vector<int32_t> input_value =
            std::vector<int32_t>(value.data.begin(), value.data.end());
        auto input_tensor = predictor->GetInputHandle(key);
        input_tensor->Reshape(value.shape);
        input_tensor->CopyFromCpu(input_value.data());
        break;
      }
      case paddle::PaddleDType::FLOAT32: {
        std::vector<float> input_value =
            std::vector<float>(value.data.begin(), value.data.end());
        auto input_tensor = predictor->GetInputHandle(key);
        input_tensor->Reshape(value.shape);
        input_tensor->CopyFromCpu(input_value.data());
        break;
      }
    }
  }

  // inference
  for (size_t i = 0; i < repeat_times; ++i) {
    ASSERT_TRUE(predictor->Run());
  }

  // get output data to Record
  auto output_names = predictor->GetOutputNames();
  for (auto &output_name : output_names) {
    Record output_Record;
    auto output_tensor = predictor->GetOutputHandle(output_name);
    std::vector<int> output_shape = output_tensor->shape();
    int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                  std::multiplies<int>());

    switch (output_tensor->type()) {
      case paddle::PaddleDType::INT64: {
        std::cout << "int64" << std::endl;
        std::vector<int64_t> out_data;
        output_Record.type = paddle::PaddleDType::INT64;
        out_data.resize(out_num);
        output_tensor->CopyToCpu(out_data.data());
        output_Record.shape = output_shape;
        std::vector<float> floatVec(out_data.begin(), out_data.end());
        output_Record.data = floatVec;
        (*output_data_map)[output_name] = output_Record;
        break;
      }
      case paddle::PaddleDType::FLOAT32: {
        std::cout << "float32" << std::endl;
        std::vector<float> out_data;
        output_Record.type = paddle::PaddleDType::FLOAT32;
        out_data.resize(out_num);
        output_tensor->CopyToCpu(out_data.data());
        output_Record.shape = output_shape;
        output_Record.data = out_data;
        (*output_data_map)[output_name] = output_Record;
        break;
      }
      case paddle::PaddleDType::INT32: {
        std::cout << "int32" << std::endl;
        std::vector<int32_t> out_data;
        output_Record.type = paddle::PaddleDType::INT32;
        out_data.resize(out_num);
        output_tensor->CopyToCpu(out_data.data());
        output_Record.shape = output_shape;
        std::vector<float> floatVec(out_data.begin(), out_data.end());
        output_Record.data = floatVec;
        (*output_data_map)[output_name] = output_Record;
        break;
      }
    }
  }
}

void CompareRecord(std::map<std::string, Record> *truth_output_data,
                   std::map<std::string, Record> *infer_output_data,
                   float epislon = 1e-5) {
  for (const auto & [ key, value ] : *infer_output_data) {
    auto truth_record = (*truth_output_data)[key];
    LOG(INFO) << "output name: " << key;
    size_t numel = value.data.size() / sizeof(float);
    EXPECT_EQ(value.data.size(), truth_record.data.size());
    for (size_t i = 0; i < numel; ++i) {
      ASSERT_LT(fabs(value.data.data()[i] - truth_record.data.data()[i]),
                epislon);
    }
  }
}

}  // namespace test
}  // namespace paddle
