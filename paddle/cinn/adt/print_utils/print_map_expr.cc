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

#include <string>

#include "paddle/cinn/adt/map_expr.h"
#include "paddle/cinn/adt/print_utils/print_equations.h"
#include "paddle/cinn/adt/print_utils/print_map_expr.h"
#include "paddle/cinn/adt/print_utils/print_schedule_descriptor.h"
#include "paddle/cinn/adt/print_utils/print_schedule_mesh.h"
#include "paddle/cinn/adt/print_utils/print_value.h"
#include "paddle/cinn/adt/schedule_descriptor.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/cinn/runtime/flags.h"

PD_DECLARE_bool(cinn_enable_map_expr_index_detail);

namespace cinn::adt {

constexpr std::size_t kIndentSpaceSize = 2;

namespace {

std::string GetIndentString(std::size_t space_size) {
  std::string ret{};
  for (std::size_t i = 0; i < space_size; ++i) {
    ret += std::string{" "};
  }
  return ret;
}

}  // namespace

template <typename DoEachT>
void VisitEachArg(const List<Arg>& out_args,
                  const List<Arg>& in_args,
                  const DoEachT& DoEach) {
  for (const auto& out_arg : *out_args) {
    DoEach(out_arg, tOut<bool>{true});
  }
  for (const auto& in_arg : *in_args) {
    DoEach(in_arg, tOut<bool>{false});
  }
}

namespace {

std::string ToTxtStringImpl(const adapter::Tensor& tensor) {
  std::string ret;
  ret += "t_";
  ret += hlir::framework::pir::CompatibleInfo::ValueName(tensor.node_data);
  return ret;
}

std::string ToTxtStringImpl(const adapter::DynamicTensor& tensor) {
  std::string ret;
  ret += "t_";
  ret += hlir::framework::pir::CompatibleInfo::ValueName(tensor.node_data);
  return ret;
}

std::string ToTxtStringImpl(const TempStorage& tensor) {
  PADDLE_THROW(::common::errors::Unimplemented("Not supported yet"));
}

}  // namespace

std::string ToTxtString(const Tensor& tensor) {
  return std::visit([&](const auto& impl) { return ToTxtStringImpl(impl); },
                    tensor.variant());
}

std::string ArgsToTxtString(
    const List<Arg>& out_args,
    const List<Arg>& in_args,
    const std::function<std::string(Arg)>& GetPrevArgStr,
    const std::function<std::string(Arg)>& GetPostArgStr) {
  std::string ret;
  ret += "(";
  std::size_t count = 0;
  VisitEachArg(out_args, in_args, [&](const auto& arg, const auto& as_output) {
    if (count++ > 0) {
      ret += ", ";
    }
    ret += GetPrevArgStr(arg);
    if (as_output.value()) {
      ret += "&";
    }
    ret += ToTxtString(arg);
    ret += GetPostArgStr(arg);
  });
  ret += ")";
  return ret;
}

std::string ArgsToTxtString(const List<Arg>& out_args,
                            const List<Arg>& in_args) {
  const auto& GetEmptyStr = [&](Arg) -> std::string { return ""; };
  return ArgsToTxtString(out_args, in_args, GetEmptyStr, GetEmptyStr);
}

std::string ToTxtStringOpImpl(const ::pir::Operation* op) {
  return hlir::framework::pir::CompatibleInfo::OpName(*op);
}

std::string ToTxtStringOpImpl(const tReduceInit<const ::pir::Operation*>& op) {
  return ToTxtStringOpImpl(op.value()) + "_init";
}

std::string ToTxtStringOpImpl(const tReduceAcc<const ::pir::Operation*>& op) {
  return ToTxtStringOpImpl(op.value()) + "_acc";
}

std::string ToTxtString(const Op& op) {
  return std::visit([&](const auto& impl) { return ToTxtStringOpImpl(impl); },
                    op.variant());
}

std::string ToTxtStringImpl(const OpStmt& op_stmt,
                            std::size_t indent_size,
                            const AnchoredMapStmt* anchored_map_stmt) {
  std::string ret;
  const auto& [op, in_args, out_args] = op_stmt.tuple();
  ret += GetIndentString(indent_size * kIndentSpaceSize);
  ret += ToTxtString(op);

  const auto& GetPrevArgStr = [&](Arg) -> std::string {
    if (FLAGS_cinn_enable_map_expr_index_detail) {
      if (anchored_map_stmt != nullptr) {
        return std::string("\n") +
               GetIndentString((indent_size + 2) * kIndentSpaceSize);
      }
    }
    return "";
  };

  const auto& GetPostArgStr = [&](Arg arg) -> std::string {
    std::string tmp;
    if (FLAGS_cinn_enable_map_expr_index_detail) {
      if (anchored_map_stmt != nullptr) {
        tmp += "[";
        tmp += ToTxtString(anchored_map_stmt->GetTensorIndexExpr(arg));
        tmp += "]";
      }
    }
    return tmp;
  };

  ret += ArgsToTxtString(
      out_args.value(), in_args.value(), GetPrevArgStr, GetPostArgStr);
  ret += ";\n";
  return ret;
}

std::string ToTxtString(const OpStmt& op_stmt) {
  return ToTxtStringImpl(op_stmt, 0, nullptr);
}

std::string ToTxtString(const LoopDescriptors& schedule_descriptor) {
  std::string ret;
  std::size_t count = 0;
  for (const auto& loop_descriptor : *schedule_descriptor) {
    if (count++ > 0) {
      ret += ", ";
    }
    ret += ToTxtString(loop_descriptor);
  }
  return ret;
}

std::string ToTxtStringImpl(const MapStmt<Stmt>& map_stmt,
                            std::size_t indent_size,
                            const AnchoredMapStmt* anchored_map_stmt);

std::string ToTxtString(const Stmt& stmt,
                        std::size_t indent_size,
                        const AnchoredMapStmt* anchored_map_stmt) {
  std::string ret{""};
  ret += std::visit(
      [&](const auto& impl) {
        return ToTxtStringImpl(impl, indent_size, anchored_map_stmt);
      },
      stmt.variant());
  return ret;
}

std::string ToTxtString(const Stmt& stmt) {
  return ToTxtString(stmt, 0, nullptr);
}

std::string ToTxtStringImpl(const MapStmt<Stmt>& map_stmt,
                            std::size_t indent_size,
                            const AnchoredMapStmt* anchored_map_stmt) {
  std::string ret;
  const auto& [loop_iterators, stmts] = map_stmt.tuple();
  ret += GetIndentString(indent_size * kIndentSpaceSize) + "MapStmt(";
  ret += ToTxtString(loop_iterators);
  ret += ") {\n";
  for (const auto& stmt : *stmts) {
    ret += ToTxtString(stmt, indent_size + 1, anchored_map_stmt);
  }
  ret += GetIndentString(indent_size * kIndentSpaceSize) + "}\n";
  return ret;
}

std::string ToTxtString(const AnchoredMapStmt& anchored_map_stmt,
                        std::size_t indent_size) {
  std::string ret;
  const auto& [map_stmt, schedule_mesh, anchor_tensor, _0, _1, _2] =
      anchored_map_stmt.tuple();
  ret += GetIndentString(indent_size * kIndentSpaceSize) + "AnchoredMapStmt(";
  ret += ToTxtString(anchor_tensor.value());
  ret += ") {\n";
  ret += ToTxtString(map_stmt, indent_size + 1, &anchored_map_stmt);
  ret += GetIndentString(indent_size * kIndentSpaceSize) + "}\n";
  return ret;
}

std::string ToTxtString(const std::string& group_id, const MapExpr& map_expr) {
  std::string ret;
  const auto& [anchored_map_stmts, inputs, outputs] = map_expr.tuple();
  ret += "\n" + group_id;
  ret += ArgsToTxtString(outputs.value(), inputs.value());

  ret += " {\n";
  for (const auto& anchored_map_stmt : *anchored_map_stmts) {
    ret += ToTxtString(anchored_map_stmt, 1);
  }
  ret += "}\n";
  return ret;
}

std::string ToTxtString(const MapExpr& map_expr, const std::string& group_id) {
  std::string ret;
  ret += ToTxtString(group_id, map_expr);
  return ret;
}

}  // namespace cinn::adt
