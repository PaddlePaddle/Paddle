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
#include "paddle/cinn/operator_fusion/utils.h"

namespace cinn::fusion {

enum InstructionType {
  T_Base,
  T_Copy,
  T_Combine,
  T_Return,
  T_InitPattern,
  T_TrivialInline,
  T_TmpTransform,
  T_TrivialLoopAlign,
  T_AnchorTransform,
  T_Padding
};

struct FusionInstruction {
  using FusionInstrPtr = std::shared_ptr<FusionInstruction>;
  virtual InstructionType type() const { return T_Base; }
  virtual std::string DebugStr() const { return "BaseInstr"; }
  virtual FusionInstrPtr Clone() {
    return std::make_shared<FusionInstruction>(*this);
  }
};

using FusionInstrPtr = std::shared_ptr<FusionInstruction>;

struct CopyInstr : public FusionInstruction {
  CopyInstr(const std::string& origin_name, const std::string& new_name)
      : origin_name_(origin_name), new_name_(new_name) {}
  virtual InstructionType type() const { return T_Copy; }
  virtual FusionInstrPtr Clone() { return std::make_shared<CopyInstr>(*this); }
  std::string origin_name_;
  std::string new_name_;

  virtual std::string DebugStr() const {
    return "CopyInstr || " + origin_name_ + " => " + new_name_;
  }
};

struct CombineInstr : public FusionInstruction {
  CombineInstr(const std::vector<std::string>& names, const std::string& result)
      : names_(names), result_(result) {}
  virtual InstructionType type() const { return T_Combine; }
  virtual FusionInstrPtr Clone() {
    return std::make_shared<CombineInstr>(*this);
  }
  std::vector<std::string> names_;
  std::string result_;

  virtual std::string DebugStr() const {
    std::stringstream ss;
    ss << "CombineInstr || ";
    for (auto name : names_) {
      ss << name << ", ";
    }
    ss << "=> " << result_;
    return ss.str();
  }
};

struct ReturnInstr : public FusionInstruction {
  explicit ReturnInstr(const std::string& target) : target_(target) {}
  virtual InstructionType type() const { return T_Return; }
  virtual FusionInstrPtr Clone() {
    return std::make_shared<ReturnInstr>(*this);
  }
  std::string target_;

  virtual std::string DebugStr() const { return "ReturnInstr || " + target_; }
};

// struct RemovePatternInstr : public FusionInstruction {};

struct InitPatternInstr : public FusionInstruction {
 private:
  // we can't just use operation* to identify a unique op because pass change
  // the program.
  int idx_in_fusion_op = -1;
  std::string op_name;

 public:
  InitPatternInstr(pir::Operation* op, const std::string& result)
      : op_(op), result_(result) {
    op_name = op->name();
  }
  void set_idx(int idx) { idx_in_fusion_op = idx; }
  int get_idx() const {
    if (idx_in_fusion_op == -1) {
      PADDLE_THROW("Not initialized.");
    }
    return idx_in_fusion_op;
  }
  virtual InstructionType type() const { return T_InitPattern; }
  virtual FusionInstrPtr Clone() {
    return std::make_shared<InitPatternInstr>(*this);
  }
  pir::Operation* op_;
  std::string result_;

  virtual std::string DebugStr() const {
    std::ostringstream ss;
    ss << "InitPatternInstr || " + op_name + " => " + result_ + " index = ("
       << idx_in_fusion_op << ")";
    return ss.str();
  }
};

struct TrivialInlineInstr : public FusionInstruction {
  TrivialInlineInstr(const std::string& upstream,
                     const std::string& downstream,
                     const std::string& result)
      : upstream_(upstream), downstream_(downstream), result_(result) {}
  virtual InstructionType type() const { return T_TrivialInline; }
  virtual FusionInstrPtr Clone() {
    return std::make_shared<TrivialInlineInstr>(*this);
  }
  std::string upstream_;
  std::string downstream_;
  std::string result_;

  virtual std::string DebugStr() const {
    return "TrivialInlineInstr || " + upstream_ + ", " + downstream_ + " => " +
           result_;
  }
};

struct TmpTransformInstr : public FusionInstruction {
  TmpTransformInstr(const std::string& upstream,
                    const std::string& downstream,
                    const std::string& out_upstream,
                    const std::string& out_downstream,
                    const std::vector<size_t>& fake_reduce_iter_idx = {})
      : upstream_(upstream),
        downstream_(downstream),
        out_downstream_(out_downstream),
        out_upstream_(out_upstream),
        fake_reduce_iter_idx_(fake_reduce_iter_idx) {}
  InstructionType type() const override { return T_TmpTransform; }
  virtual FusionInstrPtr Clone() {
    return std::make_shared<TmpTransformInstr>(*this);
  }
  std::string upstream_;
  std::string downstream_;
  std::string out_upstream_;
  std::string out_downstream_;
  std::vector<size_t> fake_reduce_iter_idx_;

  std::string DebugStr() const override {
    return "TmpTransformInstr || " + upstream_ + ", " + downstream_ + " => " +
           out_upstream_ + ", " + out_downstream_;
  }
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
  virtual InstructionType type() const { return T_TrivialLoopAlign; }
  virtual FusionInstrPtr Clone() {
    return std::make_shared<TrivialLoopAlignInstr>(*this);
  }
  std::string upstream_;
  std::string downstream_;
  std::string result_;
  std::vector<size_t> fake_reduce_iter_idx_;

  virtual std::string DebugStr() const {
    return "TrivialLoopAlignInstr || " + upstream_ + ", " + downstream_ +
           " => " + result_;
  }
};

struct AnchorTransformInstr : public FusionInstruction {
  AnchorTransformInstr(const std::string& target,
                       const std::string& result,
                       const AnchorTransformRoute& transform_route)
      : target_(target), result_(result), transform_route_(transform_route) {}
  virtual InstructionType type() const { return T_AnchorTransform; }
  virtual FusionInstrPtr Clone() {
    return std::make_shared<AnchorTransformInstr>(*this);
  }
  std::string target_;
  std::string result_;
  AnchorTransformRoute transform_route_;

  virtual std::string DebugStr() const {
    return "AnchorTransformInstr || " + target_ + " => " + result_;
  }
};

struct PaddingInstr : public FusionInstruction {
  PaddingInstr(const std::string& target,
               const std::string& result,
               const std::vector<int>& padding_pos)
      : target_(target), result_(result), padding_pos_(padding_pos) {}
  virtual InstructionType type() const { return T_Padding; }
  virtual FusionInstrPtr Clone() {
    return std::make_shared<PaddingInstr>(*this);
  }
  std::string target_;
  std::string result_;
  std::vector<int> padding_pos_;

  virtual std::string DebugStr() const {
    return "PaddingInstr || " + target_ + " => " + result_;
  }
};

template <typename T>
std::shared_ptr<T> dynamic_cast_instr_with_err(FusionInstrPtr instr) {
  auto chile_instr = std::dynamic_pointer_cast<T>(instr);
  if (!chile_instr) PADDLE_THROW("Cast Fusion Instr Failed.");
  return chile_instr;
}

struct FusionTracker {
  using FusionTrackerPtr = std::shared_ptr<FusionTracker>;
  FusionTracker() = default;
  explicit FusionTracker(const FusionTrackerPtr& other) {
    ExtendVector(&instructions_, other->instructions_);
  }
  FusionTracker(const FusionTrackerPtr& up, const FusionTrackerPtr& down) {
    ExtendVector(&instructions_, up->instructions_);
    ExtendVector(&instructions_, down->instructions_);
  }
  void append(FusionInstrPtr instr) { instructions_.emplace_back(instr); }

  FusionTrackerPtr Clone() {
    auto new_tracker = std::make_shared<FusionTracker>();
    std::vector<FusionInstrPtr> new_instructions;
    for (auto instr : instructions_) {
      new_tracker->append(instr->Clone());
    }
    return new_tracker;
  }

  std::string DebugStr() const {
    std::stringstream ss;
    ss << "FusionTracker: \n";
    for (auto instr : instructions_) {
      ss << "  " << instr->DebugStr() << "\n";
    }
    return ss.str();
  }
  std::vector<FusionInstrPtr> instructions_;
};

using FusionTrackerPtr = std::shared_ptr<FusionTracker>;

}  // namespace cinn::fusion
