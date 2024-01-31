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

#include "paddle/fluid/pir/transforms/build_cinn_pass.h"

#include <queue>
#include <regex>
#include <set>
#include <string>
#include <unordered_map>

#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/core/builder.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/dialect/control_flow/ir/cf_dialect.h"
#include "paddle/pir/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_registry.h"

#include "paddle/cinn/frontend/op_mapper_registry.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/utils/flags.h"

#include "paddle/fluid/pir/transforms/sub_graph_detector.h"

PD_DECLARE_string(allow_cinn_ops);
PD_DECLARE_string(deny_cinn_ops);

namespace {
using GroupOpsVec = std::vector<pir::Operation*>;
// The delim(`;`) that is used to split the FLAGS_allow_cinn_ops
// & FLAGS_deny_cinn_ops.
constexpr char kDelim[] = ";";
using CompatibleInfo = cinn::hlir::framework::pir::CompatibleInfo;

// OpTransInfo contains informations used to detect subgraphs
// supported by the CINN compiler.
class OpTransInfo {
  using DyOpCondT =
      std::unordered_map<std::string, std::function<bool(pir::Operation*)>>;
  using DeParamCondT =
      std::unordered_map<std::string, std::unordered_set<std::string>>;

 public:
  OpTransInfo() {
    // judgment condition for the dynamic slice
    dynamic_op_cond_.emplace("slice", [](pir::Operation* op) -> bool {
      if (!op->attributes().count("infer_flags")) return false;
      auto infer_flags = op->attributes()
                             .at("infer_flags")
                             .dyn_cast<pir::ArrayAttribute>()
                             .AsVector();
      return std::find_if(
                 infer_flags.begin(), infer_flags.end(), [](pir::Attribute& v) {
                   return v.dyn_cast<pir::Int32Attribute>().data() < 0;
                 }) != infer_flags.end();
    });
    // judgment condition for the dynamic reshape
    dynamic_op_cond_.emplace("reshape", [](pir::Operation* op) -> bool {
      bool shape_from_full = op->dyn_cast<paddle::dialect::ReshapeOp>()
                                 .shape()
                                 .dyn_cast<pir::OpResult>()
                                 .owner()
                                 ->isa<paddle::dialect::FullIntArrayOp>();
      return !shape_from_full;
    });
    // judgment condition for the dynamic expand
    dynamic_op_cond_.emplace("expand", [](pir::Operation* op) -> bool {
      bool shape_from_full = op->dyn_cast<paddle::dialect::ExpandOp>()
                                 .shape()
                                 .dyn_cast<pir::OpResult>()
                                 .owner()
                                 ->isa<paddle::dialect::FullIntArrayOp>();
      return !shape_from_full;
    });
  }

  const DyOpCondT& dynamic_op_cond() const { return dynamic_op_cond_; }

  const DeParamCondT& deny_param_cond() const { return deny_param_cond_; }

  const std::unordered_set<std::string>& default_deny_ops() const {
    return default_deny_ops_;
  }

  // TODO(Aurelius84): Deal with the special ops.
  std::unordered_set<pir::Value> GetDenyValues(const GroupOpsVec& group) const {
    return {};
  }

 private:
  DyOpCondT dynamic_op_cond_;

  DeParamCondT deny_param_cond_{{"batch_norm", {"ReserveSpace"}},
                                {"batch_norm_grad", {"ReserveSpace"}}};

  std::unordered_set<std::string> default_deny_ops_{"feed",
                                                    "fetch",
                                                    "conv2d",
                                                    "conv2d_grad",
                                                    "transpose",
                                                    "slice",
                                                    "concat",
                                                    "embedding",
                                                    "gather_nd",
                                                    "split",
                                                    "pool2d",
                                                    "arange",
                                                    "gather"};
};

std::unordered_set<std::string> StringSplit(const std::string& str,
                                            const std::string& delim) {
  std::regex reg(delim);
  std::unordered_set<std::string> elems{
      std::sregex_token_iterator(str.begin(), str.end(), reg, -1),
      std::sregex_token_iterator()};
  elems.erase("");
  return elems;
}

std::string GetDebugInfo(const std::unordered_set<std::string>& names) {
  std::string debug_info = "[";
  for (auto& name : names) {
    debug_info.append(name);
    debug_info.append(", ");
  }
  debug_info.append("]");
  return debug_info;
}

bool IsSupportCinn(pir::Operation* op);

// In case of op has some attributes generated by FullOp, it need
// implement OpPattern in pd_to_cinn_pass. Otherwise, we mark them
// as unimplement ops.
bool UnimplementOps(pir::Operation* op) {
  // cinn not support uniform, the FullOp of max and min support NOT generate by
  // CINN
  if (op->isa<paddle::dialect::FullOp>()) {
    auto out = op->result(0);
    if (out.use_count() == 1 &&
        out.first_use().owner()->name() == "builtin.set_parameter") {
      auto dim = out.type().dyn_cast<paddle::dialect::DenseTensorType>().dims();
      if (dim.size() == 1 && dim[0] == 1) {
        return true;
      }
    }

    if (out.use_count() > 0) {
      // std::cerr << "out.first_use().owner() " <<
      // out.first_use().owner()->name()
      //           << std::endl;

      bool unimplement = false;
      for (auto it = out.use_begin(); it != out.use_end(); ++it) {
        auto owner_name = it->owner()->name();
        if (owner_name == "builtin.combine") {
          return true;
        }
        if (owner_name == "pd_op.fetch" ||
            owner_name == "builtin.set_parameter" ||
            owner_name == "builtin.shadow_output") {
          continue;
        }

        if (!IsSupportCinn(it->owner())) {
          unimplement = true;
          break;
        }
      }
      return unimplement;
    }

    return false;
  } else if (op->isa<paddle::dialect::DropoutOp>()) {
    return true;
  }
  return false;
}

bool HaveZeroDimInput(pir::Operation* op) {
  bool have_zero_dim = false;
  for (size_t i = 0; i < op->num_operands(); ++i) {
    auto in = op->operand_source(i);
    if (in) {
      if (auto tensor_type =
              in.type().dyn_cast<paddle::dialect::DenseTensorType>()) {
        if (tensor_type.dims().size() == 0) {
          have_zero_dim = true;
        }
      }
    }
  }

  return have_zero_dim;
}

bool HaveNegativeDimInput(pir::Operation* op) {
  bool have_negtive_dim = false;
  for (size_t i = 0; i < op->num_operands(); ++i) {
    auto in = op->operand_source(i);
    if (in) {
      if (auto tensor_type =
              in.type().dyn_cast<paddle::dialect::DenseTensorType>()) {
        auto in_dim = tensor_type.dims();
        for (size_t j = 0; j < in_dim.size(); ++j) {
          if (in_dim[j] < 0) {
            have_negtive_dim = true;
            break;
          }
        }
      }
    }
  }

  return have_negtive_dim;
}

bool AllInputDenseTensor(pir::Operation* op) {
  bool all_denese_tensor = true;
  for (size_t i = 0; i < op->num_operands(); ++i) {
    auto in = op->operand_source(i);
    if (in) {
      if (!(in.type().isa<paddle::dialect::DenseTensorType>())) {
        all_denese_tensor = false;
      }
    }
  }

  return all_denese_tensor;
}

bool IsSupportCinn(pir::Operation* op) {
  if (op->name() == "pd_op.reshape") {
    return false;
  }

  if (!AllInputDenseTensor(op)) {
    return false;
  }

  if (HaveZeroDimInput(op)) {
    return false;
  }

  if (HaveNegativeDimInput(op)) {
    return false;
  }

  auto allow_ops = StringSplit(FLAGS_allow_cinn_ops, kDelim);
  auto deny_ops = StringSplit(FLAGS_deny_cinn_ops, kDelim);
  VLOG(4) << "The allowed Cinn Ops: " << GetDebugInfo(allow_ops);
  VLOG(4) << "The denied Cinn Ops: " << GetDebugInfo(deny_ops);

  if (UnimplementOps(op)) {
    VLOG(4) << "Found UnimplementOps: " << op->name();
    return false;
  }

  // Strip the dialect, like pd_op.abs -> abs
  const auto op_name = CompatibleInfo::OpName(*op);
  if (op_name == "matmul") {
    return false;
  }
  OpTransInfo trans_info;
  bool is_support = CompatibleInfo::IsSupportCinn(*op) &&
                    !trans_info.default_deny_ops().count(op_name);
  // if the op type is registered in CINN and allow_ops is not empty, return
  // true only when it is in allow_ops
  if (!allow_ops.empty()) {
    return is_support && allow_ops.count(op_name);
  }
  // if the op type is registered in CINN and deny_ops is not empty, return
  // true only when it is not in deny_ops
  if (!deny_ops.empty()) {
    return is_support && !deny_ops.count(op_name);
  }

  VLOG(4) << op->name() << " is_support: " << is_support << " "
          << CompatibleInfo::IsSupportCinn(*op);

  // if the user doesn't set FLAGS_allow_cinn_ops and FLAGS_deny_cinn_ops,
  // return true only when it is registered in CINN
  return is_support;
}

class BuildCinnPass : public pir::Pass {
 public:
  BuildCinnPass() : pir::Pass("build_cinn_pass", /*opt_level=*/1) {}

  void Run(pir::Operation* op) override {
    auto module_op = op->dyn_cast<pir::ModuleOp>();
    IR_ENFORCE(module_op, "build_cinn_pass should run on module op.");
    auto& block = module_op.block();

    std::vector<GroupOpsVec> groups =
        ::pir::SubgraphDetector(&block, IsSupportCinn)();
    AddStatistics(groups.size());
    for (auto& group_ops : groups) {
      if (group_ops.size() == 1 && group_ops[0]->name() == "pd_op.full") {
        continue;
      }
      VLOG(4) << "current group_ops.size(): " << group_ops.size();
      ::pir::ReplaceWithGroupOp(&block, group_ops);
    }
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
  }
};
}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateBuildCinnPass() {
  return std::make_unique<BuildCinnPass>();
}

}  // namespace pir

REGISTER_IR_PASS(build_cinn_pass, BuildCinnPass);
