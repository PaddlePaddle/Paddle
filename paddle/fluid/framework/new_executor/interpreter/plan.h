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
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/new_executor/interpreter/job.h"

#include "paddle/fluid/framework/program_desc.h"
#include "paddle/phi/core/macros.h"

namespace paddle {
namespace framework {
namespace interpreter {

class Plan final {
 public:
  Plan(const std::vector<Job>& job_list,
       const std::unordered_map<std::string, ProgramDesc*>& type_to_program);
  ~Plan() = default;

  const std::vector<std::string>& FetchNames(const std::string& job_type) const;
  const std::vector<Job>& JobList() const;
  const ProgramDesc* Program(const std::string& job_type) const;
  int64_t MicroBatchNum() const;

  void SetFetchNames(const std::string& job_type,
                     const std::vector<std::string>& fetch_names);

 private:
  const std::vector<Job> job_list_;
  const std::unordered_map<std::string, ProgramDesc*>
      type_to_program_;  // Not owned.
  int64_t micro_batch_num_;
  std::unordered_map<std::string, std::vector<std::string>>
      type_to_fetch_names_;
};

}  // namespace interpreter
}  // namespace framework
}  // namespace paddle
