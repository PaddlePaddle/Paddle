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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/phi/core/macros.h"

namespace paddle {
namespace framework {

class ProgramDesc;
class Job;

class Plan final {
 public:
  Plan(const std::vector<std::shared_ptr<Job>>& job_list,
       const std::unordered_map<std::string, ProgramDesc*>& type_to_program)
      : job_list_(job_list), type_to_program_(type_to_program) {}
  ~Plan() = default;

  const std::vector<std::shared_ptr<Job>>& GetJobList() const;
  const std::unordered_map<std::string, ProgramDesc*>& GetTypeToProgram() const;

 private:
  DISABLE_COPY_AND_ASSIGN(Plan);

  std::vector<std::shared_ptr<Job>> job_list_;
  std::unordered_map<std::string, ProgramDesc*> type_to_program_;
};

}  // namespace framework
}  // namespace paddle
