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

#include <gflags/gflags.h>

#include <functional>
#include <future>  // NOLINT
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <thread>  // NOLINT

#include <ThreadPool.h>

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {
namespace distributed {

using SparseMeta = std::tuple<std::string, std::string, int>;

struct value {
  std::vector<std::string> names;
  std::vector<float> values;
  int constant = 0;
};

class SparseVariable {
 public:
  SparseVariable();

  explicit SparseVariable(const SparseMeta& meta) {
    name = std::get<0>(meta);
    auto value_names = std::get<1>(meta);
    auto dims = std::get<2>(meta);

    for (int i = 0; i < value_names.size(); i++) {
      value_mata[value_names[i]] = dims[i];
    }
  }

 private:
  std::string name;
  std::unordered_map<std::string, int> value_mata;
  std::unordered_map<int64_t, value> values;
};

class LargeScaleKV {
 public:
  LargeScaleKV();
  ~LargeScaleKV();
  explicit LargeScaleKV(const std::vector<SparseMeta>& table_metas) {
    for (auto& sparse_meta : table_metas) {
      auto table_name = std::get<0>(sparse_meta);
      auto meta = std::make_shared<SparseVariable>(sparse_meta);
      sparse_variables[table_name] = meta;
    }
  }

  static LargeScaleKV* GetInstance() { return scale_kv_.get(); }

  static LargeScaleKV* InitInstance() {
    std::call_once(init_flag_, &LargeScaleKV::Init);
    return scale_kv_.get();
  }

  static void Init(const std::vector<SparseMeta>& table_metas) {
    if (scale_kv_.get() == nullptr) {
      scale_kv_.reset(new LargeScaleKV(table_metas));
    }
  }

  SparseVariable* Get(std::string name) {
    auto variable = sparse_variables.at(name);
    return variable.get();
  }

 private:
  std::unordered_map<std::string, std::shared_ptr<SparseVariable>>
      sparse_variables;
};

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
