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

#include "paddle/common/errors.h"
#include "paddle/common/macros.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace framework {
namespace interpreter {

class Job final {
 public:
  explicit Job(const std::string& type) : type_(type), micro_batch_id_(0) {}
  ~Job() = default;

  const std::string& Type() const { return type_; }

  int64_t MicroBatchId() const { return micro_batch_id_; }

  std::set<std::string> SkipGcVars() const { return skip_gc_vars_; }

  void SetMicroBatchId(int64_t micro_batch_id) {
    PADDLE_ENFORCE_GE(
        micro_batch_id,
        0,
        common::errors::InvalidArgument(
            "The micro_batch_id should be greater or equal to 0."));
    micro_batch_id_ = micro_batch_id;
  }

  void SetSkipGcVars(const std::set<std::string>& skip_gc_vars) {
    PADDLE_ENFORCE_EQ(skip_gc_vars_.empty(),
                      true,
                      common::errors::InvalidArgument(
                          "skip_gc_vars_ can only be initialized once, now "
                          "skip_gc_vars_ is not empty, "
                          "do not call SetSkipGcVars method repeatedly."));
    skip_gc_vars_ = skip_gc_vars;
  }

  void SetFetchVarName(const std::string& fetch_var_name) {
    fetch_var_names_.push_back(fetch_var_name);
  }

  std::vector<std::string> FetchVarNames() { return fetch_var_names_; }

 private:
  const std::string type_;
  int64_t micro_batch_id_;
  std::set<std::string> skip_gc_vars_;
  std::vector<std::string> fetch_var_names_;
};

}  // namespace interpreter
}  // namespace framework
}  // namespace paddle
