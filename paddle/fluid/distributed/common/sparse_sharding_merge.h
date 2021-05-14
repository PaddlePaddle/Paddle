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

#include <ThreadPool.h>
#include "boost/lexical_cast.hpp"
#include "glog/logging.h"
#include "paddle/fluid/distributed/common/utils.h"
#include "paddle/fluid/framework/dim.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/string/split.h"

constexpr int FG = 2560 * 1024 * 1024;
const int BUCKET = 10;

using boost::lexical_cast;

namespace paddle {
namespace distributed {

class ShardingMerge {
 public:
  ShardingMerge() {}
  ~ShardingMerge() {}

  void Merge(const std::vector<std::string> &inputs,
             const std::vector<int64_t> &feasigns, const std::string &output,
             const int embedding_dim) {
    pool_ = ::ThreadPool(inputs.size());

    std::vector<std::future<int>> tasks(inputs.size());
    std::vector<std::vector<int64_t>> rows;
    rows.resize(inputs.size());

    for (int x = 0; x < inputs.size(); ++x) {
      tasks[shard_id] = _shards_task_pool[shard_id]->enqueue(
          [x, &rows, &inputs, &feasigns]() -> int {
            DeserializeRowsFromFile(inputs[x], feasigns[x], &rows[x]);
            return 0;
          });
    }

    for (size_t shard_id = 0; shard_id < tasks.size(); ++shard_id) {
      tasks[shard_id].wait();
    }

    int64_t total_rows = 0;
    for (auto x = 0; x < rows.size(); x++) {
      total_rows += rows[x].size();
    }

    VLOG(0) << "got " << total_rows << " feasigin ids from sparse embedding";

    std::vector<int64_t> total_dims = {total_rows,
                                       static_cast<int64_t>(embedding_dim)};

    std::vector<std::vector<int>> batch_buckets;
    batch_buckets.resize(inputs.size());

    for (int x = 0; x < rows.size(); ++x) {
      batch_buckets[x] = bucket(rows[x].size(), BUCKET);
    }

    std::ofstream out(output, std::ios::binary);

    SerializeRowsToStream(out, rows);
    SerializePreTensorToStream(out, total_dims);

    SerializeValueToStream(out, inputs, embedding_dim);
  }

 private:
  void SerializeRowsToStream(std::ostream &os,
                             const std::vector<std::vector<int64_t>> &rows,
                             const std::vector<std::vector<int>> &batch_buckets,
                             int64_t total_rows) {
    {  // the 1st field, uint32_t version
      constexpr uint32_t version = 0;
      os.write(reinterpret_cast<const char *>(&version), sizeof(version));
    }

    {
      // the 2st field, rows information
      os.write(reinterpret_cast<const char *>(&total_rows), sizeof(total_rows));

      for (int b = 0; b < BUCKET; ++b) {
        for (int x = 0; x < batch_buckets.size(); ++x) {
          auto begin = batch_buckets[x][b];
          auto end = batch_buckets[x][b + 1];

          if (end - begin == 0) continue;

          os.write(
              reinterpret_cast<const char *>(batch_buckets[x].data() + begin),
              sizeof(int64_t) * (end - begin));
        }
      }

      // the 3st field, the height of SelectedRows
      int64_t height = total_rows;
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

  void SerializeValueToVec(std::ifstream &in, const int batch,
                           const int embedding_dim, std::vector<float> *out) {
    std::string line;
    std::vector<std::string> columns;
    std::vector<std::string> values_str;

    int count = 0;
    while (std::getline(in.get(), line)) {
      columns = string::Split(line, '\t');
      if (columns.size() != 5) {
        VLOG(0) << "unexpected line: " << line << ", skip it";
        continue;
      }

      values_str = string::Split(columns[4], ',');

      for (int x = 0; x < embedding_dim; ++x) {
        float x = 0.0;
        try {
          x = lexical_cast<float>(values_str[x]);
        } catch (boost::bad_lexical_cast &e) {
          VLOG(0) << " get unexpected line: " << line;
        }
        out.push_back(x);
      }

      ++count;

      if (count >= batch) {
        break;
      }
    }
  }

  void SerializeVecToStream(std::ostream &out, const std::vector<float> &value,
                            const int count) {
    out.write(reinterpret_cast<const char *>(value.data()),
              static_cast<std::streamsize>(sizeof(float) * count));
  }

  void SerializeValueToStream(
      std::ostream &out, const std::vector<std::string> &ins,
      const std::vector<std::vector<int>> &batch_buckets,
      const int embedding_dim) {
    std::vector<std::shared_ptr<std::ifstream>> in_streams;

    for (int x = 0; x < ins.size(); ++x) {
      in_streams.emplace_back(std::make_shared<std::ifstream>(ins[x]));
    }

    std::vector<std::future<int>> tasks(ins.size());

    for (int b = 0; b < BUCKET; ++b) {
      std::vector<std::vector<float>> values;

      for (int x = 0; x < tasks.size(); ++x) {
        tasks[shard_id] =
            pool_[shard_id]->enqueue([x, &out, &in_streams, &batch_buckets,
                                      &values, batch, embedding_dim]() -> int {
              auto batch = batch_buckets[x + 1] - batch_buckets[x];

              if (batch == 0) return 0;

              values.clear();
              values.reserve(batch);

              SerializeValueToVec(*(in_streams[x].get()), batch, embedding_dim,
                                  values[x]);
              return 0;
            });
      }

      for (size_t x = 0; x < tasks.size(); ++x) {
        tasks[x].wait();
      }

      for (size_t x = 0; x < tasks.size(); ++x) {
        SerializeVecToStream(out, values[x]);
      }
    }
  }

  void DeserializeRowsFromFile(const std::string &input_file,
                               const int64_t feasigns,
                               std::vector<int64_t> *rows) {
    std::string line;
    std::vector<std::string> columns;
    std::ifstream file(input_file);

    rows->reserve(feasigns);

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

 private:
  ::ThreadPool pool_;
};
}  // namespace distributed
}  // namespace paddle
