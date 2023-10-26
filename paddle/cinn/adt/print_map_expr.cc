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

#include "paddle/cinn/adt/m_expr.h"
#include "paddle/cinn/adt/print_equations.h"
#include "paddle/cinn/adt/print_map_expr.h"
#include "paddle/cinn/adt/print_schedule_descriptor.h"
#include "paddle/cinn/adt/print_schedule_mesh.h"
#include "paddle/cinn/adt/print_value.h"
#include "paddle/cinn/adt/schedule_descriptor.h"

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

std::string ToTxtString(const Tensor& tensor) {
  CHECK(tensor.Has<adapter::Tensor>());
  std::string ret;
  ret += "t_";
  ret += tensor.Get<adapter::Tensor>().node_data->id();
  return ret;
}

std::string ToTxtString(const List<Arg>& out_args,
                        const List<Arg>& in_args,
                        bool with_semicolon,
                        const AnchoredMapStmt* anchored_map_stmt) {
  std::string ret;
  ret += "(";
  std::size_t count = 0;
  VisitEachArg(out_args, in_args, [&](const auto& arg, const auto& as_output) {
    if (count++ > 0) {
      ret += ", ";
    }
    if (as_output.value()) {
      ret += "&";
    }
    ret += ToTxtString(arg);
    ret += "[";
    if (anchored_map_stmt != nullptr) {
      ret += ToTxtString(anchored_map_stmt->GetTensorIndexExpr(arg));
    }
    ret += "]";
  });
  ret += ")";
  if (with_semicolon) {
    ret += ";\n";
  }
  return ret;
}

std::string ToTxtStringOpImpl(const hlir::framework::Node* op) {
  return op->op()->name;
}

std::string ToTxtStringOpImpl(
    const tReduceInit<const hlir::framework::Node*>& op) {
  return op.value()->op()->name + "_init";
}

std::string ToTxtStringOpImpl(
    const tReduceAcc<const hlir::framework::Node*>& op) {
  return op.value()->op()->name + "_acc";
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
  ret +=
      ToTxtString(out_args.value(), in_args.value(), true, anchored_map_stmt);
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
  ret += ToTxtString(outputs.value(), inputs.value(), false, nullptr);

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
