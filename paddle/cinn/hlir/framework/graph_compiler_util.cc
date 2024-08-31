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

#include "paddle/cinn/hlir/framework/graph_compiler_util.h"
#include "paddle/common/enforce.h"

namespace cinn {
namespace hlir {
namespace framework {

void CompilationResult::InitCompilationResult(int group_size) {
  size_ = group_size;
  status_.resize(group_size, CompilationStatus::SUCCESS);
  messages_.resize(group_size);
  for (int idx = 0; idx < group_size; ++idx) {
    messages_[idx] =
        "Group Idx: " + std::to_string(idx) + ",  Compile Success.\n";
  }
  lowered_funcs_.resize(group_size, std::nullopt);
  source_codes_.resize(group_size, std::nullopt);
  source_ptxs_.resize(group_size, std::nullopt);
}

void CompilationResult::SetStatus(int idx, const CompilationStatus& status) {
  if (idx < status_.size()) {
    status_[idx] = status;
  }
}

void CompilationResult::SetMessage(int idx, const std::string& message) {
  if (idx < messages_.size()) {
    messages_[idx] = message;
  }
}

void CompilationResult::SetLoweredFuncs(
    int idx, const std::vector<ir::LoweredFunc>& funcs) {
  if (idx < lowered_funcs_.size()) {
    lowered_funcs_[idx] = funcs;
  }
}

void CompilationResult::SetSourceCode(int idx, const std::string& source_code) {
  if (idx < source_codes_.size()) {
    source_codes_[idx] = source_code;
  }
}

void CompilationResult::SetSourcePtx(int idx, const std::string& source_ptx) {
  if (idx < source_ptxs_.size()) {
    source_ptxs_[idx] = source_ptx;
  }
}

bool CompilationResult::IsSuccess() const {
  for (const CompilationStatus& s : status_) {
    if (s != CompilationStatus::SUCCESS) {
      return false;
    }
  }
  return true;
}

CompilationStatus CompilationResult::Status() const {
  CompilationStatus worst_status = CompilationStatus::SUCCESS;
  for (const CompilationStatus& s : status_) {
    if (s < worst_status) {
      worst_status = s;
    }
  }
  return worst_status;
}

CompilationStatus CompilationResult::Status(int idx) const {
  if (idx >= status_.size()) {
    return CompilationStatus::UNKNOWN_FAIL;
  }
  return status_[idx];
}

std::string CompilationResult::Message() const {
  std::string res;
  for (int idx = 0; idx < messages_.size(); ++idx) {
    res += messages_[idx];
  }
  return res;
}

std::string CompilationResult::Message(int idx) const {
  if (idx >= messages_.size()) {
    std::stringstream ss;
    ss << "The index(" << idx
       << ") is expected to be less than the size of group("
       << lowered_funcs_.size() << ").";
    PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
  }
  return messages_[idx];
}

std::vector<std::vector<ir::LoweredFunc>> CompilationResult::LoweredFuncs()
    const {
  std::vector<std::vector<ir::LoweredFunc>> res(lowered_funcs_.size());
  for (int idx = 0; idx < lowered_funcs_.size(); ++idx) {
    if (lowered_funcs_[idx].has_value()) {
      res[idx] = lowered_funcs_[idx].value();
    } else {
      std::stringstream ss;
      ss << "LoweredFuncs of group[" << idx << "] is not generated.\n"
         << "Some errors may have occurred during or before the lower "
            "process.\n"
         << Message();
      PADDLE_THROW(::common::errors::Fatal(ss.str()));
    }
  }
  return res;
}

std::vector<ir::LoweredFunc> CompilationResult::LoweredFuncs(int idx) const {
  if (idx >= lowered_funcs_.size()) {
    std::stringstream ss;
    ss << "The index(" << idx
       << ") is expected to be less than the size of group("
       << lowered_funcs_.size() << ").";
    PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
  }
  if (!lowered_funcs_[idx].has_value()) {
    std::stringstream ss;
    ss << "LoweredFuncs of group[" << idx << "] is not generated.\n"
       << "Some errors may have occurred during or before the lower process.\n"
       << Message();
    PADDLE_THROW(::common::errors::Fatal(ss.str()));
  }
  return lowered_funcs_[idx].value();
}

std::vector<std::string> CompilationResult::SourceCodes() const {
  std::vector<std::string> res(source_codes_.size());
  for (int idx = 0; idx < source_codes_.size(); ++idx) {
    if (source_codes_[idx].has_value()) {
      res[idx] = source_codes_[idx].value();
    } else {
      std::stringstream ss;
      ss << "Source Code of group[" << idx << "] is not generated.\n"
         << "Some errors may have occurred during or before the codegen "
            "process.\n"
         << Message();
      PADDLE_THROW(::common::errors::Fatal(ss.str()));
    }
  }
  return res;
}

std::string CompilationResult::SourceCode(int idx) const {
  if (idx >= source_codes_.size()) {
    std::stringstream ss;
    ss << "The index(" << idx
       << ") is expected to be less than the size of group("
       << lowered_funcs_.size() << ").";
    PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
  }
  if (!source_codes_[idx].has_value()) {
    std::stringstream ss;
    ss << "Source Code of group[" << idx << "] is not generated.\n"
       << "Some errors may have occurred during or before the codegen "
          "process.\n"
       << Message();
    PADDLE_THROW(::common::errors::Fatal(ss.str()));
  }
  return source_codes_[idx].value();
}

std::vector<std::string> CompilationResult::SourcePtxs() const {
  std::vector<std::string> res(source_ptxs_.size());
  for (int idx = 0; idx < source_ptxs_.size(); ++idx) {
    if (source_ptxs_[idx].has_value()) {
      res[idx] = source_ptxs_[idx].value();
    } else {
      std::stringstream ss;
      ss << "Source PTX of group[" << idx << "] is not generated.\n"
         << "Some errors may have occurred during or before the nvrtc compile "
            "process.\n"
         << Message();
      PADDLE_THROW(::common::errors::Fatal(ss.str()));
    }
  }
  return res;
}

std::string CompilationResult::SourcePtx(int idx) const {
  if (idx >= source_ptxs_.size()) {
    std::stringstream ss;
    ss << "The index(" << idx
       << ") is expected to be less than the size of group("
       << lowered_funcs_.size() << ").";
    PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
  }
  if (!source_ptxs_[idx].has_value()) {
    std::stringstream ss;
    ss << "Source PTX of group[" << idx << "] is not generated.\n"
       << "Some errors may have occurred during or before the nvrtc compile "
          "process.\n"
       << Message();
    PADDLE_THROW(::common::errors::Fatal(ss.str()));
  }
  return source_ptxs_[idx].value();
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
