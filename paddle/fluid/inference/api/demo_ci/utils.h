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
#include <math.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "paddle/include/paddle_inference_api.h"

namespace paddle {
namespace demo {

struct Record {
  std::vector<float> data;
  std::vector<int32_t> shape;
};

static void split(const std::string& str, char sep,
                  std::vector<std::string>* pieces) {
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

Record ProcessALine(const std::string& line) {
  VLOG(3) << "process a line";
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
  VLOG(3) << "data size " << record.data.size();
  VLOG(3) << "data shape size " << record.shape.size();
  return record;
}

void CheckOutput(const std::string& referfile, const PaddleTensor& output,
                 float threshold = 1e-5) {
  std::string line;
  std::ifstream file(referfile);
  std::getline(file, line);
  auto refer = ProcessALine(line);
  file.close();

  size_t numel = output.data.length() / PaddleDtypeSize(output.dtype);
  VLOG(3) << "predictor output numel " << numel;
  VLOG(3) << "reference output numel " << refer.data.size();
  CHECK_EQ(numel, refer.data.size());
  switch (output.dtype) {
    case PaddleDType::INT64: {
      for (size_t i = 0; i < numel; ++i) {
        CHECK_EQ(static_cast<int64_t*>(output.data.data())[i], refer.data[i]);
      }
      break;
    }
    case PaddleDType::FLOAT32: {
      for (size_t i = 0; i < numel; ++i) {
        CHECK_LT(
            fabs(static_cast<float*>(output.data.data())[i] - refer.data[i]),
            threshold);
      }
      break;
    }
    case PaddleDType::INT32: {
      for (size_t i = 0; i < numel; ++i) {
        CHECK_EQ(static_cast<int32_t*>(output.data.data())[i], refer.data[i]);
      }
      break;
    }
  }
}

/*
 * Get a summary of a PaddleTensor content.
 */
static std::string SummaryTensor(const PaddleTensor& tensor) {
  std::stringstream ss;
  int num_elems = tensor.data.length() / PaddleDtypeSize(tensor.dtype);

  ss << "data[:10]\t";
  switch (tensor.dtype) {
    case PaddleDType::INT64: {
      for (int i = 0; i < std::min(num_elems, 10); i++) {
        ss << static_cast<int64_t*>(tensor.data.data())[i] << " ";
      }
      break;
    }
    case PaddleDType::FLOAT32: {
      for (int i = 0; i < std::min(num_elems, 10); i++) {
        ss << static_cast<float*>(tensor.data.data())[i] << " ";
      }
      break;
    }
    case PaddleDType::INT32: {
      for (int i = 0; i < std::min(num_elems, 10); i++) {
        ss << static_cast<int32_t*>(tensor.data.data())[i] << " ";
      }
      break;
    }
  }
  return ss.str();
}

}  // namespace demo
}  // namespace paddle
