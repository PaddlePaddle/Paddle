// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/ap/include/paddle/pass/put_ap_trivial_ops_in_range_pass.h"
#include "paddle/ap/include/paddle/hlir/manual_op.h"

#include "paddle/ap/include/axpr/abstract_list.h"
#include "paddle/ap/include/axpr/anf_expr_util.h"
#include "paddle/ap/include/axpr/atomic.h"
#include "paddle/ap/include/axpr/builtin_frame_util.h"
#include "paddle/ap/include/axpr/builtin_serializable_attr_map_to_axpr_helper.h"
#include "paddle/ap/include/axpr/data_type_util.h"
#include "paddle/ap/include/axpr/interpreter.h"
#include "paddle/ap/include/axpr/lambda_expr_builder.h"
#include "paddle/ap/include/memory/guard.h"
#include "paddle/ap/include/paddle/builtin_frame_util.h"
#include "paddle/ap/include/paddle/pir/ap_pir_attribute.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/ir_mapping.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace ap::paddle {

namespace {

class PutApTrivialOpsInRangeHelper {
 public:
  void MatchAndRewrite(::pir::ModuleOp module_op) const {
    const auto& ret = TryMatchAndRewrite(module_op);
    PADDLE_ENFORCE_EQ(
        ret.HasError(),
        false,
        phi::errors::Fatal(
            "PutApTrivialOpsInRangeHelper::MatchAndRewrite failed. "
            "\nTraceback (most recent call "
            "last):\n%s\n%s: %s. ",
            ret.GetError().CallStackToString(),
            ret.GetError().class_name(),
            ret.GetError().msg()));
  }

 private:
  adt::Result<bool> TryMatchAndRewrite(::pir::ModuleOp module_op) const {
    auto InsertBeginEnd = [&](const auto& op_group) {
      ::pir::Builder builder{::pir::IrContext::Instance(), &module_op.block()};
      builder.set_insertion_point(op_group.front());
      builder.Build<::paddle::dialect::ApTrivialFusionBeginOp>(pir::Value{});
      builder.SetInsertionPointAfter(op_group.back());
      builder.Build<::paddle::dialect::ApTrivialFusionEndOp>(pir::Value{});
    };
    ADT_RETURN_IF_ERR(GetOpGroups(module_op, InsertBeginEnd));
    return true;
  }

  template <typename YieldT>
  adt::Result<adt::Ok> GetOpGroups(::pir::ModuleOp module_op,
                                   const YieldT& Yield) const {
    std::function<adt::Result<bool>(pir::Operation*)> IsTrivialOp{};
    ADT_RETURN_IF_ERR(MakePredicatorIsTrivialOp(module_op, &IsTrivialOp));
    std::list<std::list<pir::Operation*>> op_groups{};
    bool last_is_trivial = false;
    for (auto& op : module_op.block()) {
      if (!last_is_trivial) {
        op_groups.push_back(std::list<pir::Operation*>{});
      }
      ADT_LET_CONST_REF(cur_is_trivial, IsTrivialOp(&op));
      if (cur_is_trivial) {
        op_groups.back().push_back(&op);
      }
      last_is_trivial = cur_is_trivial;
    }
    for (const auto& op_group : op_groups) {
      if (op_group.size() <= 1) continue;
      Yield(op_group);
    }
    return adt::Ok{};
  }

  struct CustomOpNameStruct {
    std::string op_name;
    std::string custom_op_name;

    bool operator==(const CustomOpNameStruct& other) const {
      return this->op_name == other.op_name &&
             this->custom_op_name == other.custom_op_name;
    }
  };

  using TrivialOpNameImpl = std::variant<std::string, CustomOpNameStruct>;
  struct TrivialOpName : public TrivialOpNameImpl {
    using TrivialOpNameImpl::TrivialOpNameImpl;
    ADT_DEFINE_VARIANT_METHODS(TrivialOpNameImpl);

    const std::string& op_name() const {
      return Match(
          [&](const std::string& op_name) -> const std::string& {
            return op_name;
          },
          [&](const CustomOpNameStruct& custom_op_name_struct)
              -> const std::string& { return custom_op_name_struct.op_name; });
    }

    bool MatchOp(pir::Operation* op) const {
      return Match(
          [&](const std::string& op_name) -> bool {
            return op->name() == op_name;
          },
          [&](const CustomOpNameStruct& custom_op_name_struct) -> bool {
            if (op->name() != custom_op_name_struct.op_name) return false;
            const auto& attrs = op->attributes();
            const auto& iter = attrs.find("custom_op_name");
            if (iter == attrs.end()) return false;
            if (!iter->second.isa<pir::StrAttribute>()) return false;
            const auto& custom_op_name =
                iter->second.dyn_cast<pir::StrAttribute>().AsString();
            return custom_op_name == custom_op_name_struct.custom_op_name;
          });
    }
  };

  adt::Result<adt::Ok> MakePredicatorIsTrivialOp(
      ::pir::ModuleOp module_op,
      std::function<adt::Result<bool>(pir::Operation*)>* IsTrivialOp) const {
    ADT_LET_CONST_REF(trivial_op_names, GetTrivialOpNames());
    std::unordered_map<std::string, std::list<TrivialOpName>>
        op_name2trivial_op_name{};
    for (const auto& trivial_op_name : trivial_op_names) {
      op_name2trivial_op_name[trivial_op_name.op_name()].push_back(
          trivial_op_name);
    }
    using RetT = adt::Result<bool>;
    *IsTrivialOp =
        [map = move(op_name2trivial_op_name)](pir::Operation* op) -> RetT {
      const auto& iter = map.find(op->name());
      if (iter == map.end()) return false;
      const auto& trivial_op_names = iter->second;
      for (const auto& trivial_op_name : trivial_op_names) {
        if (trivial_op_name.MatchOp(op)) return true;
      }
      return false;
    };
    return adt::Ok{};
  }

  adt::Result<std::list<TrivialOpName>> GetTrivialOpNames() const {
    ADT_LET_CONST_REF(axpr_value, GetTrivialOpNamesAxprValue());
    ADT_LET_CONST_REF(lst,
                      axpr_value.template CastTo<adt::List<axpr::Value>>());
    std::list<TrivialOpName> ret_trivial_op_names;
    for (const auto& lst_elt : *lst) {
      ADT_LET_CONST_REF(trivial_op_name, ConvertToTrivialOpName(lst_elt));
      ret_trivial_op_names.push_back(trivial_op_name);
    }
    return ret_trivial_op_names;
  }

  adt::Result<TrivialOpName> ConvertToTrivialOpName(
      const axpr::Value& value) const {
    return value.Match(
        [&](const std::string& op_name) -> adt::Result<TrivialOpName> {
          return TrivialOpName{op_name};
        },
        [&](const adt::List<axpr::Value>& lst) -> adt::Result<TrivialOpName> {
          ADT_CHECK(lst->size() == 2);
          ADT_LET_CONST_REF(op_name, lst->at(0).template CastTo<std::string>());
          ADT_LET_CONST_REF(custom_op_name,
                            lst->at(1).template CastTo<std::string>());
          return TrivialOpName{CustomOpNameStruct{op_name, custom_op_name}};
        },
        [&](const auto&) -> adt::Result<TrivialOpName> {
          return adt::errors::TypeError{
              "configuration error. trivial_op_name should be of type str or "
              "tuple of str like [str, str]"};
        });
  }

  adt::Result<axpr::Value> GetTrivialOpNamesAxprValue() const {
    static axpr::Lambda<axpr::CoreExpr> lambda([] {
      ap::axpr::LambdaExprBuilder lmd{};
      const ap::axpr::AnfExpr anf_expr = lmd.Lambda({}, [&](auto& ctx) {
        auto& registry_hook_module =
            ctx.Var("import").Call(ctx.String("__builtin_trivial_op_names__"));
        return registry_hook_module.Attr("GetGroupedTrivialOpNames").Call();
      });
      const auto& core_expr = ap::axpr::ConvertAnfExprToCoreExpr(anf_expr);
      const auto& atomic =
          core_expr.Get<ap::axpr::Atomic<ap::axpr::CoreExpr>>();
      return atomic.Get<ap::axpr::Lambda<ap::axpr::CoreExpr>>();
    }());

    static ap::memory::Guard guard{};
    ap::axpr::Interpreter interpreter(
        ap::axpr::MakeBuiltinFrameAttrMap<ap::axpr::Value>(),
        guard.circlable_ref_list());
    return interpreter.Interpret(lambda, std::vector<axpr::Value>{});
  }
};

class PutApTrivialOpsInRangePass : public pir::Pass {
 public:
  PutApTrivialOpsInRangePass()
      : pir::Pass("put_ap_trivial_ops_in_range_pass", 1) {}

  void Run(pir::Operation* op) override {
    auto module_op = op->dyn_cast<pir::ModuleOp>();
    PutApTrivialOpsInRangeHelper{}.MatchAndRewrite(module_op);
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
  }
};

}  // namespace

std::unique_ptr<pir::Pass> CreatePutApTrivialOpsInRangePass() {
  return std::make_unique<PutApTrivialOpsInRangePass>();
}

}  // namespace ap::paddle
