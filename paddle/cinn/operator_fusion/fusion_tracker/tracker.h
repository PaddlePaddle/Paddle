// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/operator_fusion/pattern.h"
#include "paddle/cinn/operator_fusion/pattern_fuser.h"
#include "paddle/cinn/operator_fusion/utils.h"

namespace cinn::fusion {

using TrivialOp = cinn::hlir::framework::pir::trivial_fusion_detail::TrivialOp;
using ReduceOp = cinn::hlir::framework::pir::trivial_fusion_detail::ReduceOp;
using FusionOp = std::variant<ReduceOp, TrivialOp>;

struct FusionScope {
  std::unordered_map<std::string, std::vector<FusionOp>> scope_;
  std::string DebugStr() const;
};

struct FusionInstruction {
  virtual void Apply(FusionScope* scope);
};

using FusionInstrPtr = std::shared_ptr<FusionInstruction>;

struct TrivialInlineInstr : public FusionInstruction {};
struct TmpTransformInstr : public FusionInstruction {};
struct AnchorMatchInstr : public FusionInstruction {};
struct InitPatternInstr : public FusionInstruction {};
struct RemovePatternInstr : public FusionInstruction {};

struct FusionTracker {
  FusionTracker() = default;
  FusionTracker(const FusionTracker& up, const FusionTracker& down) {
    ExtendVector(&instructions_, up.instructions_);
    ExtendVector(&instructions_, down.instructions_);
  }
  void append(FusionInstrPtr instr) { instructions_.emplace_back(instr); }
  std::vector<FusionInstrPtr> instructions_;
  std::string DebugStr() const;
};
}  // namespace cinn::fusion
