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

#include "paddle/cinn/hlir/dialect/operator/transforms/pir_to_py_code_converter.h"
#include <atomic>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <unordered_set>
#include <variant>
#include "paddle/cinn/hlir/dialect/operator/ir/op_attribute.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/attr_adt_type_id.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/type_adt_type_id.h"
#include "paddle/common/adt_type_id.h"
#include "paddle/common/ddim.h"
#include "paddle/common/flags.h"
#include "paddle/common/overloaded.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/pir/include/core/ir_printer.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/include/dialect/shape/ir/shape_attribute.h"
#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"

COMMON_DECLARE_string(logging_pir_py_code_dir);
COMMON_DECLARE_bool(logging_trunc_pir_py_code);

namespace cinn::dialect::ir {

namespace {

class Indentation {};

template <typename T0, typename T1>
using Cons = std::pair<T0, std::shared_ptr<T1>>;

template <typename T0, typename T1>
Cons<T0, T1> MakeCons(const T0& first, const T1& second) {
  return Cons<T0, T1>{first, std::make_shared<T1>(second)};
}

template <typename T>
using IStringBase = std::variant<std::string, Cons<Indentation, T>>;

struct IString final : public IStringBase<IString> {
  explicit IString(const std::string& str) : IStringBase<IString>(str) {}

  IString(const Indentation& indent, const IString& istr)
      : IStringBase<IString>(MakeCons<Indentation, IString>(indent, istr)) {}

  const IStringBase<IString>& variant() const {
    return static_cast<const IStringBase<IString>&>(*this);
  }

  template <typename... Args>
  decltype(auto) Match(Args&&... args) const {
    return std::visit(::common::Overloaded{std::forward<Args>(args)...},
                      variant());
  }
};

IString Indent(const IString& istr) { return IString(Indentation{}, istr); }

IString Indent(const std::string& str) { return Indent(IString{str}); }

using IStrings = std::list<IString>;

struct OpPyCode {
  IStrings defines;
  std::string op_expr;
};

constexpr int kDefaultIndentSize = 2;

namespace {

int64_t GetAutoIncrementalId() {
  static std::atomic<int64_t> seq_no(0);
  return seq_no++;
}

using ShapeAnalysisGetterT =
    std::function<std::optional<pir::ShapeConstraintIRAnalysis*>(
        const pir::Program*)>;

}  // namespace

struct PirToPyCodeConverterHelper {
  explicit PirToPyCodeConverterHelper(
      const pir::Program* program,
      const ShapeAnalysisGetterT& ShapeAnalysisGetter)
      : program_(program),
        indent_size_(kDefaultIndentSize),
        seq_no_(GetAutoIncrementalId()),
        ShapeAnalysisGetter_(ShapeAnalysisGetter) {}

  std::string Convert() { return Convert(*program_); }

 private:
  const pir::Program* program_;
  const int indent_size_;
  int64_t seq_no_;
  ShapeAnalysisGetterT ShapeAnalysisGetter_;

  std::string Convert(const pir::Program& program) {
    auto istrings = ConvertMethodsToPyClass(program.module_op(), [&]() {
      IStrings all_defines = DefineInit(program.module_op());
      IStrings defines = ConvertModuleOp(program.module_op());
      all_defines.insert(all_defines.end(), defines.begin(), defines.end());
      return all_defines;
    });
    return ConvertIStringsToString(istrings);
  }

  IStrings DefineInit(const pir::ModuleOp& module) {
    IStrings def_init;
    def_init.push_back(IString("def __init__(self):"));
    VisitEachConstraintBuildStmt(
        [&](const std::string& stmt) { def_init.push_back(Indent(stmt)); });
    const auto* module_op = static_cast<const pir::Operation*>(module);
    auto* mut_module = const_cast<pir::Operation*>(module_op);
    mut_module->Walk(
        [&](pir::Operation* op) { def_init.push_back(Indent(DefineOp(op))); });
    def_init.push_back(Indent(""));
    return def_init;
  }

  template <typename DoEachConstraintBuildStmtT>
  void VisitEachConstraintBuildStmt(
      const DoEachConstraintBuildStmtT& DoEachConstraintBuildStmt) {
    VisitEachEQCstr([&](const auto& lhs, const auto& rhs) {
      DoEachConstraintBuildStmt(ConvertEQCstrBuildStmt(lhs, rhs));
    });
    VisitEachGtOneCstr([&](const auto& dim_expr) {
      DoEachConstraintBuildStmt(ConvertGtOneCstrBuildStmt(dim_expr));
    });
    VisitEachBroadcastableCstr([&](const auto& lhs, const auto& rhs) {
      DoEachConstraintBuildStmt(ConvertBroadcastableCstrBuildStmt(lhs, rhs));
    });
  }

  template <typename DoEachEQCstrT>
  void VisitEachEQCstr(const DoEachEQCstrT& DoEachEQCstr) {
    const auto& constraints_mgr = GetConstraintsMgr();
    if (!constraints_mgr.has_value()) return;
    for (const auto& [lhs, rhs] : constraints_mgr.value()->equals().GetMap()) {
      if (lhs == rhs) continue;
      DoEachEQCstr(lhs, rhs);
    }
  }

  std::string ConvertEQCstrBuildStmt(const symbol::DimExpr& lhs,
                                     const symbol::DimExpr& rhs) {
    const std::string& lhs_str = ConvertDimExpr(lhs);
    const std::string& rhs_str = ConvertDimExpr(rhs);
    std::ostringstream ss;
    ss << "self.add_eq_cstr(" << lhs_str << ", " << rhs_str << ")";
    return ss.str();
  }

  template <typename DoEachGtOneCstrT>
  void VisitEachGtOneCstr(const DoEachGtOneCstrT& DoEachGtOneCstr) {
    const auto& constraints_mgr = GetConstraintsMgr();
    if (!constraints_mgr.has_value()) return;
    for (const auto& dim_expr : constraints_mgr.value()->gtones()) {
      DoEachGtOneCstr(dim_expr);
    }
  }

  std::string ConvertGtOneCstrBuildStmt(const symbol::DimExpr& dim_expr) {
    const std::string& dim_expr_str = ConvertDimExpr(dim_expr);
    std::ostringstream ss;
    ss << "self.add_gt_one_cstr(" << dim_expr_str << ")";
    return ss.str();
  }

  template <typename DoEachBroadcastableCstrT>
  void VisitEachBroadcastableCstr(
      const DoEachBroadcastableCstrT& DoEachBroadcastableCstr) {
    const auto& constraints_mgr = GetConstraintsMgr();
    if (!constraints_mgr.has_value()) return;
    const auto& broadcastables = constraints_mgr.value()->broadcastables();
    for (const auto& broadcastable : broadcastables) {
      const auto& [lhs, rhs] = *broadcastable;
      if (lhs == rhs) continue;
      DoEachBroadcastableCstr(lhs, rhs);
    }
  }

  std::string ConvertBroadcastableCstrBuildStmt(const symbol::DimExpr& lhs,
                                                const symbol::DimExpr& rhs) {
    const std::string& lhs_str = ConvertDimExpr(lhs);
    const std::string& rhs_str = ConvertDimExpr(rhs);
    std::ostringstream ss;
    ss << "self.add_broadcastable_cstr(" << lhs_str << ", " << rhs_str << ")";
    return ss.str();
  }

  std::optional<const symbol::ConstraintsManager*> GetConstraintsMgr() {
    const auto& shape_analysis = ShapeAnalysisGetter_(program_);
    if (!shape_analysis.has_value()) return std::nullopt;
    return &shape_analysis.value()->constraints_manager();
  }

  IStrings ConvertModuleOp(const pir::ModuleOp& module) {
    return ConvertToCallMethod([&]() { return ConvertOpCall(module); });
  }

  IStrings ConvertToCallMethod(const std::function<OpPyCode()>& GetOpPyCode) {
    auto [ret, op_py_code] = GetOpPyCode();
    ret.push_back(IString("def __call__(self, call, *args, **kwargs):"));
    ret.push_back(Indent("self.SetArgs(args)"));
    ret.push_back(Indent("self.SetKeywordArgs(kwargs)"));
    ret.push_back(Indent(std::string("return ") + op_py_code));
    return ret;
  }

  std::vector<pir::Value> GetFreeVars(const pir::Block& block) {
    std::vector<pir::Value> inputs;
    for (const auto& value : GetUsedExternalValue(block)) {
      if (!value) continue;
      if (std::find(inputs.begin(), inputs.end(), value) != inputs.end())
        continue;
      inputs.push_back(value);
    }
    return inputs;
  }

  std::string ConvertFreeVarsAsArgs(const pir::Block& block) {
    const std::vector<pir::Value> inputs = GetFreeVars(block);
    return ConvertValuesAsArgs(inputs);
  }

  std::string ConvertValuesAsArgs(const std::vector<pir::Value>& inputs) {
    std::stringstream ss;
    for (int i = 0; i < inputs.size(); ++i) {
      if (i > 0) {
        ss << ", ";
      }
      ss << ConvertValue(inputs.at(i));
    }
    return ss.str();
  }

  std::string ConvertKwargsToString(const pir::Block& block) {
    std::vector<pir::Value> values;
    for (const auto& [_, value] : block.kwargs()) {
      values.push_back(value);
    }
    return ConvertValuesAsArgs(values);
  }

  std::string ConvertValue(pir::Value value) {
    if (!value) {
      return "None";
    }
    const auto* op = value.defining_op();
    if (op == nullptr) {
      return std::string("arg_") +
             std::to_string(std::hash<pir::Value>()(value));
    }
    std::string op_unique_name = ConvertOpUniqueName(op);
    std::string idx = std::to_string(GetResultIdx(op, value));
    return op_unique_name + idx;
  }

  int GetResultIdx(const pir::Operation* op, pir::Value value) {
    for (int i = 0; i < op->num_results(); ++i) {
      if (op->result(i) == value) return i;
    }
    return -1;
  }

  std::string ConvertOpUniqueName(const pir::Operation* op) {
    std::string valid_var_name = ConvertOpNameToPythonValidVarName(op->name());
    return valid_var_name + "_" + std::to_string(op->id());
  }

  std::string ConvertOpNameToPythonValidVarName(const std::string& name) {
    const auto IsValidVarChar = [](char ch) {
      if (ch >= 'a' && ch <= 'z') return true;
      if (ch >= 'A' && ch <= 'Z') return true;
      if (ch >= '0' && ch <= '9') return true;
      if (ch == '_') return true;
      return false;
    };
    int i = name.size() - 1;
    for (; i >= 0; --i) {
      if (!IsValidVarChar(name.at(i))) break;
    }
    return name.substr(i + 1);
  }

  OpPyCode ConvertBlock(const pir::Block& block,
                        const std::string& func_op_name) {
    IStrings all_defines;
    IStrings block_body;
    const auto& IsReturnOp = [](const pir::Operation& op) {
      if (op.isa<::pir::YieldOp>()) return true;
      return false;
    };
    for (const auto& op : block) {
      const auto& [defines, py_expr] = ConvertOpCall(&op);
      all_defines.insert(all_defines.end(), defines.begin(), defines.end());
      block_body.push_back([&] {
        if (IsReturnOp(op)) {
          return IString{std::string("return ") + py_expr};
        } else {
          return IString{py_expr};
        }
      }());
    }
    const std::string ret_lambda_name = "ret_lambda";
    const auto GetRetLambda = [&]() {
      const auto& args_str = ConvertValuesAsArgs(block.args());
      const auto& kwargs_str = ConvertKwargsToString(block);
      IString ret_lambda_declare(
          std::string("def ") + ret_lambda_name + "(" + args_str +
          (kwargs_str.empty() ? "" : ", *, ") + kwargs_str + "):");
      IStrings return_lambda{ret_lambda_declare};
      PushBackIndented(&return_lambda, block_body);
      return return_lambda;
    };
    std::string free_vars_as_args = ConvertFreeVarsAsArgs(block);
    IStrings func = [&] {
      IString declare(std::string("def ") + func_op_name + "(self, call" +
                      (free_vars_as_args.empty() ? "" : ", ") +
                      free_vars_as_args + "):");
      IStrings block_func{declare};
      PushBackIndented(&block_func, GetRetLambda());
      block_func.push_back(Indent(std::string("return ") + ret_lambda_name));
      block_func.push_back(Indent(""));
      return block_func;
    }();
    all_defines.insert(all_defines.end(), func.begin(), func.end());
    const std::string block_lambda_and_free_vars =
        std::string("(self.") + func_op_name +
        (free_vars_as_args.empty() ? "," : ", ") + free_vars_as_args + ")";
    return OpPyCode{all_defines, block_lambda_and_free_vars};
  }

  OpPyCode ConvertRegions(const pir::Operation* op) {
    IStrings all_defines;
    std::stringstream ss;
    const std::string op_var_name = ConvertOpUniqueName(op);

    ss << "[";
    int i = 0;
    for (const auto& region : *op) {
      if (i > 0) {
        ss << ",";
      }
      int j = 0;
      ss << "[";
      for (const auto& block : region) {
        const std::string block_name =
            op_var_name + "_block" + std::to_string(i) + std::to_string(j);
        const auto& [defines, lambda] = ConvertBlock(block, block_name);
        all_defines.insert(all_defines.end(), defines.begin(), defines.end());
        if (j > 0) {
          ss << ",";
        }
        ss << lambda;
        ++j;
      }
      ss << "]";
      ++i;
    }
    ss << "]";
    return OpPyCode{all_defines, ss.str()};
  }

  std::string ConvertOperandsAsArgs(const pir::Operation* op) {
    std::stringstream ss;
    for (int i = 0; i < op->num_operands(); ++i) {
      if (i > 0) {
        ss << ", ";
      }
      ss << ConvertValue(op->operand_source(i));
    }
    return ss.str();
  }

  std::string ConvertResultAsTuple(const pir::Operation* op) {
    std::stringstream ss;
    for (int i = 0; i < op->num_results(); ++i) {
      if (i > 0) {
        ss << ", ";
      }
      ss << ConvertValue(op->result(i));
    }
    return ss.str();
  }

  std::string ConvertAttrsAsArgs(const pir::Operation* op) {
    std::stringstream ss;
    int i = 0;
    VisitAttr(op, [&](const auto& attr_name, const auto& attr) {
      if (i++ > 0) {
        ss << ", ";
      }
      ss << attr_name << "=" << ConvertAttr(attr);
    });
    VisitSymbolicAttrs(op, [&](const auto& attr_name, const auto& attrs) {
      if (i++ > 0) {
        ss << ", ";
      }
      ss << attr_name << "=" << ConvertSymbolicAttrs(attrs);
    });
    return ss.str();
  }

  static std::string ConvertAttr(const pir::Attribute& attr) {
    auto adt_type_id = GetAttrAdtTypeId(attr);
    return std::visit(AttrConverter{attr}, adt_type_id.variant());
  }

  static std::string ConvertSymbolicAttrs(
      const std::vector<std::optional<pir::Attribute>>& attrs) {
    std::ostringstream ss;
    ss << "self.a_array(";
    int i = 0;
    for (const auto& attr : attrs) {
      if (i++ > 0) {
        ss << ", ";
      }
      if (!attr.has_value()) {
        ss << "self.a_symbol(self.s_null())";
      } else {
        ss << ConvertAttr(attr.value());
      }
    }
    ss << ")";
    return ss.str();
  }

  static std::string ConvertShapeOrData(
      const symbol::ShapeOrDataDimExprs& shape_or_data) {
    return shape_or_data.Match(
        [](const symbol::TensorShapeOrDataDimExprs& impl) {
          return ConvertTensorShapeOrData(impl);
        },
        [](const symbol::TensorListShapeOrDataDimExprs& impl) {
          return ConvertTensorListShapeOrData(impl);
        });
  }

  static std::string ConvertTensorListShapeOrData(
      const symbol::TensorListShapeOrDataDimExprs& tensor_list_shape_or_data) {
    std::ostringstream ss;
    ss << "self.s_tensor_list_shape_or_data(";
    int i = 0;
    for (const auto& shape_or_data : tensor_list_shape_or_data) {
      if (i++ > 0) {
        ss << ", ";
      }
      ss << ConvertTensorShapeOrData(shape_or_data);
    }
    ss << ")";
    return ss.str();
  }

  static std::string ConvertTensorShapeOrData(
      const symbol::TensorShapeOrDataDimExprs& shape_or_data) {
    std::string shape = ConvertSymbolShape(shape_or_data);
    std::string data = ConvertSymbolData(shape_or_data);
    std::ostringstream ss;
    ss << "self.s_tensor_shape_or_data(" << shape << ", " << data << ")";
    return ss.str();
  }

  static std::string ConvertSymbolShape(
      const symbol::TensorShapeOrDataDimExprs& shape_or_data) {
    return ConvertDimExprs(shape_or_data.shape());
  }

  static std::string ConvertSymbolData(
      const symbol::TensorShapeOrDataDimExprs& shape_or_data) {
    const auto& data = shape_or_data.data();
    if (!data.has_value()) {
      return "None";
    } else {
      return ConvertDimExprs(data.value());
    }
  }

  static std::string ConvertDimExprs(
      const std::vector<symbol::DimExpr>& dim_exprs) {
    std::ostringstream ss;
    ss << "[";
    int i = 0;
    for (const auto& dim_expr : dim_exprs) {
      if (i++ > 0) {
        ss << ", ";
      }
      ss << ConvertDimExpr(dim_expr);
    }
    ss << "]";
    return ss.str();
  }

  static std::string ConvertDimExpr(const symbol::DimExpr& dim_expr) {
    return dim_expr.Match(
        [](int64_t constant) {
          std::ostringstream ss;
          ss << "self.s_int64(" << constant << ")";
          return ss.str();
        },
        [](const std::string& symbol) {
          std::ostringstream ss;
          ss << "self.s_str(" << std::quoted(symbol) << ")";
          return ss.str();
        },
        [](const symbol::Negative<symbol::DimExpr>& negative) {
          std::ostringstream ss;
          const auto& [operand] = *negative;
          ss << "self.s_negative(";
          ss << PirToPyCodeConverterHelper::ConvertDimExpr(operand);
          ss << ")";
          return ss.str();
        },
        [](const symbol::Reciprocal<symbol::DimExpr>& reciprocal) {
          std::ostringstream ss;
          const auto& [operand] = *reciprocal;
          ss << "self.s_reciprocal(";
          ss << PirToPyCodeConverterHelper::ConvertDimExpr(operand);
          ss << ")";
          return ss.str();
        },
        [](const symbol::Add<symbol::DimExpr>& add) {
          std::ostringstream ss;
          ss << "self.s_add(";
          const auto& operands = add.operands;
          int i = 0;
          for (const auto& operand : *operands) {
            if (i++ > 0) {
              ss << ", ";
            }
            ss << PirToPyCodeConverterHelper::ConvertDimExpr(operand);
          }
          ss << ")";
          return ss.str();
        },
        [](const symbol::Mul<symbol::DimExpr>& mul) {
          std::ostringstream ss;
          ss << "self.s_mul(";
          const auto& operands = mul.operands;
          int i = 0;
          for (const auto& operand : *operands) {
            if (i++ > 0) {
              ss << ", ";
            }
            ss << PirToPyCodeConverterHelper::ConvertDimExpr(operand);
          }
          ss << ")";
          return ss.str();
        },
        [](const symbol::Max<symbol::DimExpr>& max) {
          std::ostringstream ss;
          ss << "self.s_max(";
          const auto& operands = max.operands;
          int i = 0;
          for (const auto& operand : *operands) {
            if (i++ > 0) {
              ss << ", ";
            }
            ss << PirToPyCodeConverterHelper::ConvertDimExpr(operand);
          }
          ss << ")";
          return ss.str();
        },
        [](const symbol::Min<symbol::DimExpr>& min) {
          std::ostringstream ss;
          ss << "self.s_min(";
          const auto& operands = min.operands;
          int i = 0;
          for (const auto& operand : *operands) {
            if (i++ > 0) {
              ss << ", ";
            }
            ss << PirToPyCodeConverterHelper::ConvertDimExpr(operand);
          }
          ss << ")";
          return ss.str();
        },
        [](const symbol::Broadcast<symbol::DimExpr>& broadcast) {
          std::ostringstream ss;
          ss << "self.s_broadcast(";
          const auto& operands = broadcast.operands;
          int i = 0;
          for (const auto& operand : *operands) {
            if (i++ > 0) {
              ss << ", ";
            }
            ss << PirToPyCodeConverterHelper::ConvertDimExpr(operand);
          }
          ss << ")";
          return ss.str();
        });
  }

  struct AttrConverter {
    pir::Attribute attr_;

    template <typename T>
    using TypeId = ::common::AdtTypeId<T>;

    std::string operator()(TypeId<pir::BoolAttribute>) {
      const auto& name = pir::BoolAttribute::name();
      bool data = attr_.dyn_cast<pir::BoolAttribute>().data();
      std::stringstream ss;
      ss << "self." << name << "(";
      ss << (data ? "True" : "False");
      ss << ")";
      return ss.str();
    }
    std::string operator()(TypeId<pir::Complex64Attribute>) {
      const auto& name = pir::Complex64Attribute::name();
      const auto& data = attr_.dyn_cast<pir::Complex64Attribute>().data();
      std::stringstream ss;
      ss << "self." << name << "(\"" << data.real << "\", \"" << data.imag
         << "\")";
      return ss.str();
    }
    std::string operator()(TypeId<pir::Complex128Attribute>) {
      const auto& name = pir::Complex128Attribute::name();
      const auto& data = attr_.dyn_cast<pir::Complex128Attribute>().data();
      std::stringstream ss;
      ss << "self." << name << "(\"" << data.real << "\", \"" << data.imag
         << "\")";
      return ss.str();
    }
    std::string operator()(TypeId<pir::FloatAttribute>) {
      const auto& name = pir::FloatAttribute::name();
      const auto& data = attr_.dyn_cast<pir::FloatAttribute>().data();
      std::stringstream ss;
      ss << "self." << name << "(\"" << data << "\")";
      return ss.str();
    }
    std::string operator()(TypeId<pir::DoubleAttribute>) {
      const auto& name = pir::DoubleAttribute::name();
      const auto& data = attr_.dyn_cast<pir::DoubleAttribute>().data();
      std::stringstream ss;
      ss << "self." << name << "(\"" << data << "\")";
      return ss.str();
    }
    std::string operator()(TypeId<pir::Int32Attribute>) {
      const auto& name = pir::Int32Attribute::name();
      const auto& data = attr_.dyn_cast<pir::Int32Attribute>().data();
      std::stringstream ss;
      ss << "self." << name << "(" << data << ")";
      return ss.str();
    }
    std::string operator()(TypeId<pir::IndexAttribute>) {
      const auto& name = pir::IndexAttribute::name();
      const auto& data = attr_.dyn_cast<pir::IndexAttribute>().data();
      std::stringstream ss;
      ss << "self." << name << "(" << data << ")";
      return ss.str();
    }
    std::string operator()(TypeId<pir::Int64Attribute>) {
      const auto& name = pir::Int64Attribute::name();
      const auto& data = attr_.dyn_cast<pir::Int64Attribute>().data();
      std::stringstream ss;
      ss << "self." << name << "(" << data << ")";
      return ss.str();
    }
    std::string operator()(TypeId<pir::PointerAttribute>) {
      const auto& name = pir::PointerAttribute::name();
      void* data = attr_.dyn_cast<pir::PointerAttribute>().data();
      std::stringstream ss;
      ss << "self." << name << "(\"" << data << "\")";
      return ss.str();
    }
    std::string operator()(TypeId<pir::TypeAttribute>) {
      const auto& name = pir::TypeAttribute::name();
      const auto data = attr_.dyn_cast<pir::TypeAttribute>().data();
      std::stringstream ss;
      ss << "self." << name << "(\"";
      data.Print(ss);
      ss << "\")";
      return ss.str();
    }
    std::string operator()(TypeId<pir::StrAttribute>) {
      const auto& name = pir::StrAttribute::name();
      const auto& data = attr_.dyn_cast<pir::StrAttribute>().AsString();
      std::stringstream ss;
      ss << "self." << name << "(" << std::quoted(data) << ")";
      return ss.str();
    }
    std::string operator()(TypeId<pir::ArrayAttribute>) {
      const auto& name = pir::ArrayAttribute::name();
      const auto& array_attr = attr_.dyn_cast<pir::ArrayAttribute>();
      std::stringstream ss;
      ss << "self." << name << "(";
      for (int i = 0; i < array_attr.size(); ++i) {
        if (i > 0) {
          ss << ", ";
        }
        ss << PirToPyCodeConverterHelper::ConvertAttr(array_attr[i]);
      }
      ss << ")";
      return ss.str();
    }
    std::string operator()(TypeId<pir::TensorNameAttribute>) {
      const auto& name = pir::TensorNameAttribute::name();
      const auto& data = attr_.dyn_cast<pir::TensorNameAttribute>().data();
      std::stringstream ss;
      ss << "self." << name << "(" << std::quoted(data) << ")";
      return ss.str();
    }
    std::string operator()(TypeId<pir::shape::SymbolAttribute>) {
      const auto& name = pir::shape::SymbolAttribute::name();
      const auto& data = attr_.dyn_cast<pir::shape::SymbolAttribute>().data();
      const std::string& shape_or_data =
          PirToPyCodeConverterHelper::ConvertShapeOrData(data);
      return "self." + name + "(" + shape_or_data + ")";
    }
    std::string operator()(TypeId<paddle::dialect::KernelAttribute>) {
      const auto& name = paddle::dialect::KernelAttribute::name();
      return "self." + name + "()";
    }
    std::string operator()(TypeId<paddle::dialect::IntArrayAttribute>) {
      const auto& name = paddle::dialect::IntArrayAttribute::name();
      const auto& data =
          attr_.dyn_cast<paddle::dialect::IntArrayAttribute>().data();
      std::stringstream ss;
      ss << "self." << name << "(";
      for (int i = 0; i < data.size(); ++i) {
        if (i > 0) {
          ss << ", ";
        }
        ss << data[i];
      }
      ss << ")";
      return ss.str();
    }
    std::string operator()(TypeId<paddle::dialect::ScalarAttribute>) {
      const auto& name = paddle::dialect::ScalarAttribute::name();
      const auto& data =
          attr_.dyn_cast<paddle::dialect::ScalarAttribute>().data();
      pir::Type type = paddle::dialect::TransToIrDataType(data.dtype());
      std::stringstream ss;
      ss << "self." << name << "(" << std::quoted(data.ToRawString()) << ", "
         << PirToPyCodeConverterHelper::ConvertType(type) << ")";
      return ss.str();
    }
    std::string operator()(TypeId<paddle::dialect::DataTypeAttribute>) {
      const auto& name = paddle::dialect::DataTypeAttribute::name();
      const auto& data =
          attr_.dyn_cast<paddle::dialect::DataTypeAttribute>().data();
      std::stringstream ss;
      ss << "self." << name << "(\"" << data << "\")";
      return ss.str();
    }
    std::string operator()(TypeId<paddle::dialect::PlaceAttribute>) {
      const auto& name = paddle::dialect::PlaceAttribute::name();
      const auto& place =
          attr_.dyn_cast<paddle::dialect::PlaceAttribute>().data();
      std::stringstream ss;
      ss << "self." << name << "(";
      if (place.GetType() == phi::AllocationType::CUSTOM) {
        ss << std::quoted(place.GetDeviceType());
      } else {
        ss << std::quoted(phi::AllocationTypeStr(place.GetType()));
      }
      if (place.GetType() == phi::AllocationType::GPUPINNED ||
          place.GetType() == phi::AllocationType::CPU) {
        // Do nothing.
      } else {
        ss << ", " << static_cast<int64_t>(place.GetDeviceId());
      }
      ss << ")";
      return ss.str();
    }
    std::string operator()(TypeId<paddle::dialect::DataLayoutAttribute>) {
      const auto& name = paddle::dialect::DataLayoutAttribute::name();
      const auto& data =
          attr_.dyn_cast<paddle::dialect::DataLayoutAttribute>().data();
      std::stringstream ss;
      ss << "self." << name << "(\"" << data << "\")";
      return ss.str();
    }
    std::string operator()(TypeId<cinn::dialect::GroupInfoAttribute>) {
      const auto& name = cinn::dialect::GroupInfoAttribute::name();
      std::stringstream ss;
      ss << "self." << name << "()";
      return ss.str();
    }
    std::string operator()(TypeId<cinn::dialect::CINNKernelInfoAttribute>) {
      const auto& name = cinn::dialect::CINNKernelInfoAttribute::name();
      std::stringstream ss;
      ss << "self." << name << "()";
      return ss.str();
    }
    std::string operator()(TypeId<UnclassifiedAttribute>) {
      return "self.UnclassifiedAttribute()";
    }
  };

  template <typename DoEachAttrT>
  void VisitAttr(const pir::Operation* op, const DoEachAttrT& DoEachAttr) {
    for (const auto& [attr_name, attr] : op->attributes()) {
      if (attr_name == "op_callstack") continue;
      if (attr_name == "sym_shape_str") continue;
      DoEachAttr(attr_name, attr);
    }
  }

  template <typename DoEachAttrT>
  void VisitSymbolicAttrs(const pir::Operation* op,
                          const DoEachAttrT& DoEachAttr) {
    DoEachAttr("__operands_symbols_signature__",
               GetOpOperandsSymbolDimsAttributes(op));
    DoEachAttr("__results_symbols_signature__",
               GetOpResultsSymbolDimsAttributes(op));
  }

  std::vector<std::optional<pir::Attribute>> GetOpOperandsSymbolDimsAttributes(
      const pir::Operation* op) {
    std::vector<std::optional<pir::Attribute>> attrs;
    attrs.reserve(op->num_operands());
    for (int i = 0; i < op->num_operands(); ++i) {
      attrs.push_back(GetValueSymbolDimsAttribute(op->operand_source(i)));
    }
    return attrs;
  }

  std::vector<std::optional<pir::Attribute>> GetOpResultsSymbolDimsAttributes(
      const pir::Operation* op) {
    std::vector<std::optional<pir::Attribute>> attrs;
    attrs.reserve(op->num_results());
    for (int i = 0; i < op->num_results(); ++i) {
      attrs.push_back(GetValueSymbolDimsAttribute(op->result(i)));
    }
    return attrs;
  }

  std::optional<pir::Attribute> GetValueSymbolDimsAttribute(pir::Value value) {
    auto* ctx = pir::IrContext::Instance();
    using SymbolAttr = pir::shape::SymbolAttribute;
    if (!value) {
      return std::nullopt;
    }
    const auto& shape_or_data = GetShapeOrDataDimExprs(value);
    if (!shape_or_data.has_value()) return std::nullopt;
    return SymbolAttr::get(ctx, *shape_or_data.value());
  }

  std::optional<const symbol::ShapeOrDataDimExprs*> GetShapeOrDataDimExprs(
      pir::Value value) {
    const auto& shape_analysis = ShapeAnalysisGetter_(program_);
    if (!shape_analysis.has_value()) return std::nullopt;
    return &shape_analysis.value()->GetShapeOrDataForValue(value);
  }

  std::string ConvertInputTypes(const pir::Operation* op) {
    std::stringstream ss;
    ss << "[";
    for (int i = 0; i < op->num_operands(); ++i) {
      if (i > 0) {
        ss << ", ";
      }
      ss << ConvertType(op->operand_source(i).type());
    }
    ss << "]";
    return ss.str();
  }

  std::string ConvertOutputTypes(const pir::Operation* op) {
    std::stringstream ss;
    ss << "[";
    for (int i = 0; i < op->num_results(); ++i) {
      if (i > 0) {
        ss << ", ";
      }
      ss << ConvertType(op->result(i).type());
    }
    ss << "]";
    return ss.str();
  }

  static std::string ConvertType(const pir::Type& type) {
    auto adt_type_id = GetTypeAdtTypeId(type);
    return std::visit(TypeConverter{type}, adt_type_id.variant());
  }

  struct TypeConverter {
    pir::Type type;

    template <typename T>
    using AdtTypeId = ::common::AdtTypeId<T>;

    std::string operator()(AdtTypeId<cinn::dialect::ir::NullType>) {
      return "self.t_null";
    }

    std::string operator()(AdtTypeId<::pir::VectorType>) {
      std::stringstream ss;
      const auto& name = ::pir::VectorType::name();
      const auto& vec_type = type.dyn_cast<::pir::VectorType>();
      ss << "self." << name << "(";
      for (int i = 0; i < vec_type.size(); ++i) {
        if (i > 0) {
          ss << ", ";
        }
        ss << PirToPyCodeConverterHelper::ConvertType(vec_type[i]);
      }
      ss << ")";
      return ss.str();
    }

    std::string operator()(AdtTypeId<::pir::DenseTensorType>) {
      std::stringstream ss;
      const auto& name = ::pir::DenseTensorType::name();
      const auto& dens_type = type.dyn_cast<::pir::DenseTensorType>();
      ss << "self." << name << "(";
      ss << "[";
      int i = 0;
      for (int dim : ::common::vectorize<int>(dens_type.dims())) {
        if (i++ > 0) {
          ss << ", ";
        }
        ss << dim;
      }
      ss << "], ";
      ss << PirToPyCodeConverterHelper::ConvertType(dens_type.dtype());
      ss << ")";
      return ss.str();
    }

    std::string operator()(AdtTypeId<::pir::BFloat16Type>) {
      const auto& name = ::pir::BFloat16Type::name();
      return std::string("self.") + name + "()";
    }

    std::string operator()(AdtTypeId<::pir::Float16Type>) {
      const auto& name = ::pir::Float16Type::name();
      return std::string("self.") + name + "()";
    }

    std::string operator()(AdtTypeId<::pir::Float32Type>) {
      const auto& name = ::pir::Float32Type::name();
      return std::string("self.") + name + "()";
    }

    std::string operator()(AdtTypeId<::pir::Float64Type>) {
      const auto& name = ::pir::Float64Type::name();
      return std::string("self.") + name + "()";
    }

    std::string operator()(AdtTypeId<::pir::Int8Type>) {
      const auto& name = ::pir::Int8Type::name();
      return std::string("self.") + name + "()";
    }

    std::string operator()(AdtTypeId<::pir::UInt8Type>) {
      const auto& name = ::pir::UInt8Type::name();
      return std::string("self.") + name + "()";
    }

    std::string operator()(AdtTypeId<::pir::Int16Type>) {
      const auto& name = ::pir::Int16Type::name();
      return std::string("self.") + name + "()";
    }

    std::string operator()(AdtTypeId<::pir::Int32Type>) {
      const auto& name = ::pir::Int32Type::name();
      return std::string("self.") + name + "()";
    }

    std::string operator()(AdtTypeId<::pir::Int64Type>) {
      const auto& name = ::pir::Int64Type::name();
      return std::string("self.") + name + "()";
    }

    std::string operator()(AdtTypeId<::pir::IndexType>) {
      const auto& name = ::pir::IndexType::name();
      return std::string("self.") + name + "()";
    }

    std::string operator()(AdtTypeId<::pir::BoolType>) {
      const auto& name = ::pir::BoolType::name();
      return std::string("self.") + name + "()";
    }

    std::string operator()(AdtTypeId<::pir::Complex64Type>) {
      const auto& name = ::pir::Complex64Type::name();
      return std::string("self.") + name + "()";
    }

    std::string operator()(AdtTypeId<::pir::Complex128Type>) {
      const auto& name = ::pir::Complex128Type::name();
      return std::string("self.") + name + "()";
    }
    std::string operator()(AdtTypeId<UnclassifiedType>) {
      std::stringstream ss;
      ss << "self.UnclassifiedType(";
      ss << std::quoted([&] {
        std::stringstream type_ss;
        pir::IrPrinter printer(type_ss);
        printer.PrintType(type);
        return type_ss.str();
      }());
      ss << ")";
      return ss.str();
    }
  };

  std::string DefineOp(const pir::Operation* op) {
    const std::string& id = std::to_string(op->id());
    const std::string& input_types_str = ConvertInputTypes(op);
    const std::string& output_types_str = ConvertOutputTypes(op);
    const std::string& attrs_as_args = ConvertAttrsAsArgs(op);
    const std::string& block_signature = ConvertBlockSignatureAsArgs(op);
    std::stringstream ss;
    ss << "self." << ConvertOpUniqueName(op) << " = self.Op("
       << std::quoted(op->name()) << ", " << id << ", "
       << "input_types=" << input_types_str
       << ", output_types=" << output_types_str << ", attrs=dict("
       << attrs_as_args << ")";
    if (!block_signature.empty()) {
      ss << ", " << block_signature;
    }
    ss << ")";
    return ss.str();
  }

  std::string ConvertBlockSignatureAsArgs(const pir::Operation* op) {
    if (op->num_regions() == 0) return "";
    std::stringstream ss;
    const auto& ConvertPostionalArgsAsQuotedString = [&](const auto& block) {
      std::stringstream ss;
      int idx = 0;
      for (const auto& value : block.args()) {
        if (idx++ > 0) {
          ss << ", ";
        }
        ss << std::quoted(ConvertValue(value));
      }
      return ss.str();
    };
    {
      int i = 0;
      ss << "block_positional_arg_names=[";
      for (const auto& region : *op) {
        if (i++ > 0) {
          ss << ",";
        }
        int j = 0;
        ss << "[";
        for (const auto& block : region) {
          if (j++ > 0) {
            ss << ",";
          }
          ss << "[" << ConvertPostionalArgsAsQuotedString(block) << "]";
        }
        ss << "]";
      }
      ss << "], ";
    }
    const auto& ConvertKeywordArgsAsQuotedString = [&](const auto& block) {
      std::stringstream ss;
      int idx = 0;
      for (const auto& [key, value] : block.kwargs()) {
        if (idx++ > 0) {
          ss << ", ";
        }
        ss << std::quoted(key) << ": " << std::quoted(ConvertValue(value));
      }
      return ss.str();
    };
    {
      int i = 0;
      ss << "block_keyword_arg_names=[";
      for (const auto& region : *op) {
        if (i++ > 0) {
          ss << ",";
        }
        int j = 0;
        ss << "[";
        for (const auto& block : region) {
          if (j++ > 0) {
            ss << ",";
          }
          ss << "{" << ConvertKeywordArgsAsQuotedString(block) << "}";
        }
        ss << "]";
      }
      ss << "]";
    }
    return ss.str();
  }

  OpPyCode ConvertOpCall(const pir::Operation* op) {
    const std::string name = op->name();
    const std::string& id = std::to_string(op->id());
    auto [defines, regions] = ConvertRegions(op);
    const std::string& operands_as_args = ConvertOperandsAsArgs(op);
    const std::string& results_as_tuple_str = ConvertResultAsTuple(op);
    std::stringstream ss;
    if (!results_as_tuple_str.empty()) {
      ss << results_as_tuple_str << ", = ";
    }
    ss << "call(self." << ConvertOpUniqueName(op);
    if (!operands_as_args.empty()) {
      ss << ", " << operands_as_args;
    }
    if (regions != "[]") {
      ss << ", blocks=" << regions;
    }
    ss << ")";
    return OpPyCode{defines, ss.str()};
  }

  IStrings ConvertMethodsToPyClass(const pir::ModuleOp& module,
                                   const std::function<IStrings()>& GetBody) {
    IStrings ret;
    {
      std::stringstream ss;
      ss << "class " << GetPyClassName() << ":";
      ret.push_back(IString(ss.str()));
    }
    PushBackIndented(&ret, GetBody());
    return ret;
  }

  std::string GetPyClassName() {
    return std::string("PirProgram_") + std::to_string(seq_no_);
  }

  std::string ConvertIStringsToString(const IStrings& istrings) {
    std::stringstream ss;
    for (const auto& istring : istrings) {
      ss << ConvertIStringToString(istring) << std::endl << std::endl;
    }
    return ss.str();
  }

  std::string ConvertIStringToString(const IString& istring) {
    return istring.Match([](const std::string& str) { return str; },
                         [this](const Cons<Indentation, IString>& cons) {
                           std::string ret;
                           for (int i = 0; i < indent_size_; ++i) {
                             ret += " ";
                           }
                           ret += ConvertIStringToString(*cons.second);
                           return ret;
                         });
  }

  void PushBackIndented(IStrings* ret, const IStrings& istrings) {
    for (const auto& istring : istrings) {
      ret->push_back(Indent(istring));
    }
  }
};

std::optional<pir::ShapeConstraintIRAnalysis*> GetShapeAnalysisFromManager(
    const pir::Program* program) {
  return &pir::ShapeAnalysisManager::Instance().Get(program);
}

std::optional<pir::ShapeConstraintIRAnalysis*> GetNullShapeAnalysis(
    const pir::Program* program) {
  return std::nullopt;
}

}  // namespace

void PirToPyCodeConverter::SaveIfFlagEnabled() const {
  if (program_ == nullptr) return;
  if (file_name_.empty()) return;
  if (FLAGS_logging_pir_py_code_dir == "") return;
  const std::string file_path =
      FLAGS_logging_pir_py_code_dir + "/" + file_name_;
  ShapeAnalysisGetterT ShapeAnalysisGetter =
      (dump_symbolic_shape_ ? GetShapeAnalysisFromManager
                            : GetNullShapeAnalysis);
  PirToPyCodeConverterHelper converter_helper(program_, ShapeAnalysisGetter);
  const std::string content = converter_helper.Convert();
  static std::mutex mutex;
  std::unique_lock<std::mutex> lock(mutex);
  if (FLAGS_logging_trunc_pir_py_code) {
    static std::unordered_map<std::string, std::once_flag> once_flags;
    std::call_once(once_flags[file_path], [&] {
      std::ofstream ofs;
      ofs.open(file_path.c_str(), std::ios::out | std::ios::trunc);
      ofs.close();
    });
  }
  std::ofstream ofs;
  ofs.open(file_path.c_str(), std::ios::out | std::ios::app);
  if (!ofs.is_open()) return;
  ofs << content << std::endl;
  ofs.close();
}

}  // namespace cinn::dialect::ir
