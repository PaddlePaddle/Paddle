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
#include "paddle/phi/core/macros.h"

namespace paddle {
namespace framework {

class Job final {
 public:
  explicit Job(const std::string& type) : type_(type), micro_batch_id_(-1) {}
  ~Job() = default;

  const std::string& GetJobType() const { return type_; }
  int64_t GetMicroBatchId() const { return micro_batch_id_; }

  void SetMicroBatchId(int64_t micro_batch_id) {
    micro_batch_id_ = micro_batch_id;
  }

 private:
  DISABLE_COPY_AND_ASSIGN(Job);

  std::string type_;
  int64_t micro_batch_id_;
};

}  // namespace framework
}  // namespace paddle
