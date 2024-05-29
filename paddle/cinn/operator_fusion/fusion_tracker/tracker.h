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
#include "paddle/cinn/operator_fusion/pir_graph_analyzing/anchor_transform.h"
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

struct TrivialInlineInstr : public FusionInstruction {
  TrivialInlineInstr(const std::string& upstream,
                     const std::string& downstream const std::string& result)
      : upstream_(upstream), downstream_(downstream), result_(result) {}
  std::string upstream_;
  std::string downstream_;
  std::string result_;
};
struct TmpTransformInstr : public FusionInstruction {
  TmpTransformInstr(const std::string& upstream,
                    const std::string& downstream const std::string& result)
      : upstream_(upstream), downstream_(downstream), result_(result) {}
  std::string upstream_;
  std::string downstream_;
  std::string result_;
};
struct TmpTransformWithFakeReduceIterInstr : public FusionInstruction {
  TmpTransformInstr(const std::string& upstream,
                    const std::string& downstream,
                    const std::string& result,
                    const vector<size_t>& fake_reduce_iter_idx)
      : upstream_(upstream),
        downstream_(downstream),
        result_(result),
        fake_reduce_iter_idx_(fake_reduce_iter_idx) {}
  std::string upstream_;
  std::string downstream_;
  std::string result_;
  vector<size_t> fake_reduce_iter_idx_;
};
struct AnchorTransformInstr : public FusionInstruction {
  TmpTransformInstr(const std::string& upstream,
                    const std::string& downstream,
                    const std::string& result,
                    const AnchorTransformRoute& transform_route,
                    bool is_upstream_anchor)
      : upstream_(upstream),
        downstream_(downstream),
        result_(result),
        transform_route_(transform_route),
        is_upstream_anchor_(is_upstream_anchor) {}
  std::string upstream_;
  std::string downstream_;
  std::string result_;
  AnchorTransformRoute transform_route_;
  bool is_upstream_anchor_;
};
struct CombineInstr : public FusionInstruction {
  TmpTransformInstr(const std::string& first,
                    const std::string& second,
                    const std::string& result)
      : first_(first), second_(second), result_(result) {}
  std::string first_;
  std::string second_;
  std::string result_;
};
struct InitPatternInstr : public FusionInstruction {
  InitPatternInstr(pir::Operation* op, const std::string& result)
      : op_(op), result_(result) {}
  pir::Operation* op_;
  std::string result_;
};
struct RenamePatternInstr : public FusionInstruction {
  TmpTransformInstr(const std::string& origin_name, const std::string& new_name)
      : origin_name_(origin_name), new_name_(new_name) {}
  std::string origin_name_;
  std::string new_name_;
};
struct RemovePatternInstr : public FusionInstruction {};

struct FusionTracker {
  FusionTracker() = default;
  FusionTracker(const FusionTracker& up, const FusionTracker& down) {
    ExtendVector(&instructions_, up.instructions_);
    ExtendVector(&instructions_, down.instructions_);
  }
  void append(FusionInstrPtr instr) { instructions_.emplace_back(instr); }
  std::string DebugStr() const;

  std::vector<FusionInstrPtr> instructions_;
};
}  // namespace cinn::fusion
