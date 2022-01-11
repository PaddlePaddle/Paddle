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
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <unordered_set>

#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/eager/legacy/type_def.h"
#include "paddle/fluid/imperative/amp_auto_cast.h"

namespace egr {
namespace legacy {

class AmpOperators {
 public:
  ~AmpOperators();
  AmpOperators(const AmpOperators& o) = delete;
  const AmpOperators& operator=(const AmpOperators& o) = delete;

  static AmpOperators& Instance();

  std::shared_ptr<std::unordered_set<std::string>> GetMutableAllowOps();

  std::shared_ptr<std::unordered_set<std::string>> GetMutableBlockOps();

  std::shared_ptr<std::unordered_set<std::string>>
  GetMutableUnsupportedFp16Ops();

 private:
  AmpOperators();  // forbid calling default constructor

  // The set of ops that support fp16 calculation and are considered numerically
  // safe and performance critical. These ops are always converted to fp16.
  std::shared_ptr<std::unordered_set<std::string>> allow_ops_;

  // The set of ops that support fp16 calculation and are considered numerically
  // dangerous and whose effects may also be observed in downstream ops.
  std::shared_ptr<std::unordered_set<std::string>> block_ops_;

  // The set of ops that has no fp16 CUDA kennel.
  std::shared_ptr<std::unordered_set<std::string>> unsupported_fp16_ops_;
};

std::ostream& operator<<(std::ostream& os, AmpOperators& ops);

// NOTE(zhiqiu): AutoCastGuard is used for RAII.
class AutoCastGuard {
 public:
  explicit AutoCastGuard(paddle::imperative::AmpLevel guard_level) {
    pre_amp_level_ = Controller::Instance().GetAMPLevel();

    if (pre_amp_level_ != guard_level) {
      Controller::Instance().SetAMPLevel(guard_level);
    }
  }

  ~AutoCastGuard() { Controller::Instance().SetAMPLevel(pre_amp_level_); }

  // forbid copy and operator=
  AutoCastGuard(const AutoCastGuard& guard) = delete;
  AutoCastGuard& operator=(const AutoCastGuard& guard) = delete;

 private:
  paddle::imperative::AmpLevel pre_amp_level_;
};

NameTensorMap AutoCastInputs(const std::string& op_type,
                             const NameTensorMap& ins);

NameTensorMap CastPureFp16Inputs(const std::string& op_type,
                                 const NameTensorMap& ins);

}  // namespace legacy
}  // namespace egr
