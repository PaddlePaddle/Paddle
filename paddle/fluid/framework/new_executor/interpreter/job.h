// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include <glog/logging.h>
#include <set>

#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"
#include "paddle/phi/core/macros.h"

namespace paddle {
namespace framework {
namespace interpreter {

class Job final {
 public:
  explicit Job(const std::string& type) : type_(type), micro_batch_id_(0) {}
  ~Job() = default;

  const std::string& Type() const { return type_; }

  int ColAttrForFetchOp(int fetch_op_id) const {
    return fetch_op_id_to_col_attr_.at(fetch_op_id);
  }

  int64_t MicroBatchId() const { return micro_batch_id_; }

  std::set<std::string> SkipGcVars() const { return skip_gc_vars_; }

  std::vector<int> AllFetchOpIds() const {
    std::vector<int> fetch_op_ids;
    fetch_op_ids.reserve(fetch_op_id_to_col_attr_.size());
    for (auto& item : fetch_op_id_to_col_attr_) {
      fetch_op_ids.push_back(item.first);
    }
    return fetch_op_ids;
  }

  void SetColAttrForFetchOp(int fetch_op_id, int col_attr) {
    fetch_op_id_to_col_attr_[fetch_op_id] = col_attr;
  }

  void SetMicroBatchId(int64_t micro_batch_id) {
    PADDLE_ENFORCE_GE(
        micro_batch_id,
        0,
        phi::errors::InvalidArgument(
            "The micro_batch_id should be greater or equal to 0."));
    micro_batch_id_ = micro_batch_id;
  }

  void SetSkipGcVars(const std::set<std::string>& skip_gc_vars) {
    PADDLE_ENFORCE_EQ(skip_gc_vars_.empty(),
                      true,
                      phi::errors::InvalidArgument(
                          "skip_gc_vars_ can only be initialized once, now "
                          "skip_gc_vars_ is not empty, "
                          "do not call SetSkipGcVars method repeatedly."));
    skip_gc_vars_ = skip_gc_vars;
  }

 private:
  const std::string type_;
  int64_t micro_batch_id_;
  std::unordered_map<int, int> fetch_op_id_to_col_attr_;
  std::set<std::string> skip_gc_vars_;
};

}  // namespace interpreter
}  // namespace framework
}  // namespace paddle
