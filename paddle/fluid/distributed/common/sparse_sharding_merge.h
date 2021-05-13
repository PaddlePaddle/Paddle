// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

#include "boost/lexical_cast.hpp"
#include "glog/logging.h"

#include "paddle/fluid/framework/dim.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/string/split.h"

constexpr int ALLOC = 100000;

using boost::lexical_cast;

namespace paddle {
namespace distributed {

class ShardingMerge {
 public:
  ShardingMerge() {}
  ~ShardingMerge() {}

  void Merge(const std::vector<std::string> &inputs, const std::string &output,
             const int embedding_dim) {
    std::vector<std::vector<int64_t>> rows;
    rows.resize(inputs.size());

    for (int x = 0; x < inputs.size(); ++x) {
      DeserializeRowsFromFile(inputs[x], &rows[x]);
    }

    int64_t total_rows = 0;
    for (auto x = 0; x < rows.size(); x++) {
      total_rows += rows[x].size();
    }

    std::vector<int64_t> total_dims = {total_rows,
                                       static_cast<int64_t>(embedding_dim)};

    std::ofstream out(output, std::ios::binary);
    SerializeRowsToStream(out, rows);
    SerializePreTensorToStream(out, total_dims);
    SerializeValueToStream(out, inputs, embedding_dim);
  }

 private:
  void SerializeRowsToStream(std::ostream &os,
                             const std::vector<std::vector<int64_t>> &rows) {
    {  // the 1st field, uint32_t version
      constexpr uint32_t version = 0;
      os.write(reinterpret_cast<const char *>(&version), sizeof(version));
    }

    {
      // the 2st field, rows information
      uint64_t size = 0;
      for (auto &row : rows) {
        size += row.size();
      }

      os.write(reinterpret_cast<const char *>(&size), sizeof(size));
      for (auto i = 0; i < rows.size(); i++) {
        for (auto row : rows[i]) {
          os.write(reinterpret_cast<const char *>(&row), sizeof(row));
        }
      }

      // the 3st field, the height of SelectedRows
      int64_t height = size;
      os.write(reinterpret_cast<const char *>(&height), sizeof(height));
    }
  }

  void SerializePreTensorToStream(std::ostream &os,
                                  const std::vector<int64_t> &dims) {
    {  // the 1st field, uint32_t version
      constexpr uint32_t version = 0;
      os.write(reinterpret_cast<const char *>(&version), sizeof(version));
    }
    {  // the 2nd field, tensor description
      // int32_t  size
      framework::proto::VarType::TensorDesc desc;
      desc.set_data_type(framework::proto::VarType::FP32);
      auto *pb_dims = desc.mutable_dims();
      pb_dims->Resize(static_cast<int>(dims.size()), 0);
      std::copy(dims.begin(), dims.end(), pb_dims->begin());
      int32_t size = desc.ByteSize();
      os.write(reinterpret_cast<const char *>(&size), sizeof(size));
      auto out = desc.SerializeAsString();
      os.write(out.data(), size);
    }
  }

  void SerializeValueToStream(std::ostream &out,
                              const std::vector<std::string> &ins,
                              const int embedding_dim) {
    for (int x = 0; x < ins.size(); ++x) {
      std::string line;
      std::vector<std::string> columns;
      std::vector<std::string> values_str;
      std::vector<float> values;
      values.resize(embedding_dim);
      std::ifstream file(ins[x]);

      while (std::getline(file, line)) {
        columns = string::Split(line, '\t');
        if (columns.size() != 5) {
          VLOG(0) << "unexpected line: " << line << ", skip it";
          continue;
        }

        values_str = string::Split(columns[4], ',');

        for (int x = 0; x < embedding_dim; ++x) {
          try {
            values[x] = lexical_cast<float>(values_str[x]);
          } catch (boost::bad_lexical_cast &e) {
            VLOG(0) << " get unexpected line: " << line;
            values[x] = 0.0;
          }
        }

        // the 3rd field, tensor data
        out.write(reinterpret_cast<const char *>(values.data()),
                  static_cast<std::streamsize>(sizeof(float) * values.size()));
      }
    }
  }

  void DeserializeRowsFromFile(const std::string &input_file,
                               std::vector<int64_t> *rows) {
    std::string line;
    std::vector<std::string> columns;
    std::ifstream file(input_file);

    rows->reserve(ALLOC);

    while (std::getline(file, line)) {
      columns = string::Split(line, '\t');
      if (columns.size() != 5) {
        VLOG(0) << "unexpected line: " << line << ", skip it";
        continue;
      }
      rows->push_back(std::stoull(columns[0]));
    }

    VLOG(0) << "parse " << rows->size() << " embedding rows from "
            << input_file;
  }
};
}  // namespace distributed
}  // namespace paddle
