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

#include "paddle/fluid/framework/new_executor/interpreter/job.h"

#include "paddle/common/macros.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/pir/include/core/program.h"

namespace paddle {
namespace framework {
namespace interpreter {

class Plan final {
 public:
  Plan(const std::vector<std::shared_ptr<Job>>& job_list,
       const std::unordered_map<std::string, std::shared_ptr<ProgramDesc>>&
           type_to_program);
  Plan(const std::vector<std::shared_ptr<Job>>& job_list,
       const std::unordered_map<std::string, std::shared_ptr<::pir::Program>>&
           type_to_ir_program);

  ~Plan() = default;

  const std::vector<std::shared_ptr<Job>>& JobList() const;
  const std::vector<std::string> JobTypes() const;

  const std::shared_ptr<ProgramDesc> Program(const std::string& job_type) const;
  std::shared_ptr<::pir::Program> IrProgram(const std::string& job_type) const;

  void SetIrProgram(const std::string& job_type,
                    std::shared_ptr<::pir::Program> ir_prog);

  int64_t MicroBatchNum() const;

 private:
  const std::vector<std::shared_ptr<Job>> job_list_;
  const std::unordered_map<std::string, std::shared_ptr<ProgramDesc>>
      type_to_program_;
  std::unordered_map<std::string, std::shared_ptr<::pir::Program>>
      type_to_ir_program_;
  int64_t micro_batch_num_;
};

}  // namespace interpreter
}  // namespace framework
}  // namespace paddle
