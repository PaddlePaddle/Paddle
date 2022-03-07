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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/imperative/type_defs.h"

namespace paddle {
namespace imperative {

// NOTE(zhiqiu): only O1 and O2 are valid now
enum class AmpLevel {
  O0 = 0,  // fp32
  O1,      // amp, mixed fp32-fp16
  O2,      // almost fp16
  O3,      // fp16
};

std::tuple<std::unordered_set<std::string>, std::unordered_set<std::string>,
           std::unordered_set<std::string>>
OpSupportedInfos(const std::string& place,
                 framework::proto::VarType::Type dtype);

class Tracer;

// Singleton implementation with C++ 11
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

  std::shared_ptr<std::unordered_set<std::string>>
  GetMutableUnsupportedBf16Ops();

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

  // The set of ops that has no bf16 CUDA kennel.
  std::shared_ptr<std::unordered_set<std::string>> unsupported_bf16_ops_;
};

std::ostream& operator<<(std::ostream& os, AmpOperators& ops);

// NOTE(zhiqiu): AutoCastGuard is used for RAII.
class AutoCastGuard {
 public:
  AutoCastGuard(std::shared_ptr<Tracer> tracer, AmpLevel guard_level);

  ~AutoCastGuard();

  // forbid copy and operator=
  AutoCastGuard(const AutoCastGuard& guard) = delete;
  AutoCastGuard& operator=(const AutoCastGuard& guard) = delete;

 private:
  std::shared_ptr<Tracer> tracer_;
  AmpLevel pre_amp_level_;
};

template <typename VarType>
NameVarMap<VarType> AutoCastInputs(const std::string& op_type,
                                   const NameVarMap<VarType>& ins);
template <typename VarType>
NameVarMap<VarType> CastPureFp16Inputs(const std::string& op_type,
                                       const NameVarMap<VarType>& ins);
template <typename VarType>
NameVarMap<VarType> AutoCastBF16Inputs(const std::string& op_type,
                                       const NameVarMap<VarType>& ins);
template <typename VarType>
NameVarMap<VarType> CastPureBf16Inputs(const std::string& op_type,
                                       const NameVarMap<VarType>& ins);

}  // namespace imperative
}  // namespace paddle
