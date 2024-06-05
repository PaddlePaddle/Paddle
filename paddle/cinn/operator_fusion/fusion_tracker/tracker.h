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
#include "paddle/cinn/operator_fusion/pir_graph_analyzing/anchor_transform.h"

namespace cinn::fusion {

enum InstructionType {
  T_Rename,
  T_Combine,
  T_Return,
  T_InitPattern,
  T_TrivialInline,
  T_TmpTransform,
  T_TrivialLoopAlign,
  T_AnchorTransformAndReturn,
};

struct FusionInstruction {
  virtual InstructionType type();
};

using FusionInstrPtr = std::shared_ptr<FusionInstruction>;

struct RenameInstr : public FusionInstruction {
  RenameInstr(const std::string& origin_name, const std::string& new_name)
      : origin_name_(origin_name), new_name_(new_name) {}
  virtual InstructionType type() { return T_Rename; }
  std::string origin_name_;
  std::string new_name_;
};

struct CombineInstr : public FusionInstruction {
  CombineInstr(const std::vector<std::string>& names, const std::string& result)
      : names_(names), result_(result) {}
  virtual InstructionType type() { return T_Combine; }
  std::vector<std::string> names_;
  std::string result_;
};

struct ReturnInstr : public FusionInstruction {
  explicit ReturnInstr(const std::string& target) : target_(target) {}
  virtual InstructionType type() { return T_Return; }
  std::string target_;
};

// struct RemovePatternInstr : public FusionInstruction {};

struct InitPatternInstr : public FusionInstruction {
  InitPatternInstr(pir::Operation* op, const std::string& result)
      : op_(op), result_(result) {}
  virtual InstructionType type() { return T_InitPattern; }
  pir::Operation* op_;
  std::string result_;
};

struct TrivialInlineInstr : public FusionInstruction {
  TrivialInlineInstr(const std::string& upstream,
                     const std::string& downstream,
                     const std::string& result)
      : upstream_(upstream), downstream_(downstream), result_(result) {}
  virtual InstructionType type() { return T_TrivialInline; }
  std::string upstream_;
  std::string downstream_;
  std::string result_;
};

struct TmpTransformInstr : public FusionInstruction {
  TmpTransformInstr(const std::string& upstream,
                    const std::string& downstream,
                    const std::string& result,
                    const std::vector<size_t>& fake_reduce_iter_idx = {})
      : upstream_(upstream),
        downstream_(downstream),
        result_(result),
        fake_reduce_iter_idx_(fake_reduce_iter_idx) {}
  virtual InstructionType type() { return T_TmpTransform; }
  std::string upstream_;
  std::string downstream_;
  std::string result_;
  std::vector<size_t> fake_reduce_iter_idx_;
};

struct TrivialLoopAlignInstr : public FusionInstruction {
  TrivialLoopAlignInstr(const std::string& upstream,
                        const std::string& downstream,
                        const std::string& result,
                        const std::vector<size_t>& fake_reduce_iter_idx)
      : upstream_(upstream),
        downstream_(downstream),
        result_(result),
        fake_reduce_iter_idx_(fake_reduce_iter_idx) {}
  virtual InstructionType type() { return T_TrivialLoopAlign; }
  std::string upstream_;
  std::string downstream_;
  std::string result_;
  std::vector<size_t> fake_reduce_iter_idx_;
};

struct AnchorTransformAndReturnInstr : public FusionInstruction {
  AnchorTransformAndReturnInstr(const std::string& target,
                                const AnchorTransformRoute& transform_route)
      : target_(target), transform_route_(transform_route) {}
  virtual InstructionType type() { return T_AnchorTransformAndReturn; }
  std::string target_;
  AnchorTransformRoute transform_route_;
};

struct FusionTracker {
  using FusionTrackerPtr = std::shared_ptr<FusionTracker>;
  FusionTracker() = default;
  FusionTracker(const FusionTrackerPtr& up, const FusionTrackerPtr& down) {
    ExtendVector(&instructions_, up->instructions_);
    ExtendVector(&instructions_, down->instructions_);
  }
  void append(FusionInstrPtr instr) { instructions_.emplace_back(instr); }
  std::string DebugStr() const;
  std::vector<FusionInstrPtr> instructions_;
};

using FusionTrackerPtr = std::shared_ptr<FusionTracker>;
}  // namespace cinn::fusion
