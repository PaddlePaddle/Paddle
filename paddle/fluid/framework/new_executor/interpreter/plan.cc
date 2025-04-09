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

#include "paddle/fluid/framework/new_executor/interpreter/plan.h"

#include "paddle/fluid/framework/program_desc.h"

namespace paddle::framework::interpreter {

Plan::Plan(const std::vector<std::shared_ptr<Job>>& job_list,
           const std::unordered_map<std::string, std::shared_ptr<ProgramDesc>>&
               type_to_program)
    : job_list_(job_list),
      type_to_program_(type_to_program),
      micro_batch_num_(1) {
  for (size_t i = 0; i < job_list_.size(); ++i) {
    const auto& job = job_list_[i];
    PADDLE_ENFORCE(type_to_program_.find(job->Type()) != type_to_program_.end(),
                   common::errors::InvalidArgument(
                       "The %d-th job (type:%s, micro_batch_id:%d) has no "
                       "corresponding Program.",
                       i,
                       job->Type(),
                       job->MicroBatchId()));

    micro_batch_num_ = std::max(micro_batch_num_, job->MicroBatchId() + 1);
  }
}

Plan::Plan(
    const std::vector<std::shared_ptr<Job>>& job_list,
    const std::unordered_map<std::string, std::shared_ptr<::pir::Program>>&
        type_to_ir_program)
    : job_list_(job_list),
      type_to_ir_program_(type_to_ir_program),
      micro_batch_num_(1) {
  for (size_t i = 0; i < job_list_.size(); ++i) {
    const auto& job = job_list_[i];
    PADDLE_ENFORCE(
        type_to_ir_program_.find(job->Type()) != type_to_ir_program_.end(),
        common::errors::InvalidArgument(
            "The %d-th job (type:%s, micro_batch_id:%d) has no "
            "corresponding Program.",
            i,
            job->Type(),
            job->MicroBatchId()));

    micro_batch_num_ = std::max(micro_batch_num_, job->MicroBatchId() + 1);
  }
}

const std::vector<std::shared_ptr<Job>>& Plan::JobList() const {
  return job_list_;
}

const std::vector<std::string> Plan::JobTypes() const {
  std::vector<std::string> res;
  res.reserve(type_to_program_.size());
  for (auto const& kv : type_to_ir_program_) {
    res.emplace_back(kv.first);
  }
  return res;
}

const std::shared_ptr<ProgramDesc> Plan::Program(
    const std::string& job_type) const {
  return type_to_program_.at(job_type);
}

std::shared_ptr<::pir::Program> Plan::IrProgram(
    const std::string& job_type) const {
  return type_to_ir_program_.at(job_type);
}

void Plan::SetIrProgram(const std::string& job_type,
                        std::shared_ptr<::pir::Program> ir_prog) {
  type_to_ir_program_[job_type] = ir_prog;
}

int64_t Plan::MicroBatchNum() const { return micro_batch_num_; }

}  // namespace paddle::framework::interpreter
