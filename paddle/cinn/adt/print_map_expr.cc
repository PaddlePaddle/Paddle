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
#include "paddle/cinn/adt/print_map_expr.h"
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

void ToTensorTxtString(const Tensor& tensor, std::string* string) {
  CHECK(tensor.Has<adapter::Tensor>());
  *string += "t_";
  *string += tensor.Get<adapter::Tensor>().node_data->id();
}

void ToTxtString(const List<Arg>& out_args,
                 const List<Arg>& in_args,
                 std::string* string,
                 bool with_semicolon) {
  *string += "(";
  std::size_t count = 0;
  VisitEachArg(out_args, in_args, [&](const auto& arg, const auto& as_output) {
    if (count++ > 0) {
      *string += ", ";
    }
    if (as_output.value()) {
      *string += "&";
    }
    ToTensorTxtString(arg, string);
  });
  *string += ")";
  if (with_semicolon) {
    *string += ";\n";
  }
}

void ToTextStringImpl(const OpStmt& op_stmt,
                      std::size_t indent_size,
                      std::string* string) {
  const auto& [op, in_args, out_args] = op_stmt.tuple();
  CHECK(op.Has<const hlir::framework::Node*>());
  *string += GetIndentString(indent_size * kIndentSpaceSize);
  *string += op.Get<const hlir::framework::Node*>()->op()->name;
  ToTxtString(out_args.value(), in_args.value(), string, true);
}

void ToTextString(const LoopDescriptor& loop_descriptor,
                  std::size_t indent_size,
                  std::string* string) {
  *string += DebugString(loop_descriptor);
}

void ToTextString(const ScheduleDescriptor& schedule_descriptor,
                  std::size_t indent_size,
                  std::string* string) {
  std::size_t count = 0;
  for (const auto& loop_descriptor : *schedule_descriptor) {
    if (count++ > 0) {
      *string += ", ";
    }
    ToTextString(loop_descriptor, indent_size, string);
  }
}

void ToTextStringImpl(const MapStmt<Stmt>& map_stmt,
                      std::size_t indent_size,
                      std::string* string);

void ToTextString(const Stmt& stmt,
                  std::size_t indent_size,
                  std::string* string) {
  std::visit(
      [&](const auto& impl) { ToTextStringImpl(impl, indent_size, string); },
      stmt.variant());
}

void ToTextStringImpl(const MapStmt<Stmt>& map_stmt,
                      std::size_t indent_size,
                      std::string* string) {
  const auto& [schedule_descriptor, stmts] = map_stmt.tuple();
  *string += GetIndentString(indent_size * kIndentSpaceSize) + "MapStmt(";
  ToTextString(schedule_descriptor, indent_size, string);
  *string += ") {\n";
  for (const auto& stmt : *stmts) {
    ToTextString(stmt, indent_size + 1, string);
  }
  *string += GetIndentString(indent_size * kIndentSpaceSize) + "}\n";
}

void ToTextString(const AnchoredMapStmt& anchored_map_stmt,
                  std::size_t indent_size,
                  std::string* string) {
  const auto& [map_stmt, anchor_tensor, _] = anchored_map_stmt.tuple();
  *string +=
      GetIndentString(indent_size * kIndentSpaceSize) + "AnchoredMapStmt(";
  ToTensorTxtString(anchor_tensor.value(), string);
  *string += ") {\n";
  ToTextString(map_stmt, indent_size + 1, string);
  *string += GetIndentString(indent_size * kIndentSpaceSize) + "}\n";
}

void ToTextString(const std::string& group_id,
                  const MapExpr& map_expr,
                  std::size_t indent_size,
                  std::string* txt_string) {
  const auto& [anchored_map_stmts, inputs, outputs] = map_expr.tuple();

  *txt_string += "\n" + group_id;
  ToTxtString(outputs.value(), inputs.value(), txt_string, false);
  *txt_string += " {\n";
  for (const auto& anchored_map_stmt : *anchored_map_stmts) {
    ToTextString(anchored_map_stmt, 1, txt_string);
  }
  *txt_string += "}\n";
}

void PrintMapExpr(const MapExpr& map_expr, const std::string& group_id) {
  std::string txt_string{};
  ToTextString(group_id, map_expr, 0, &txt_string);
  LOG(ERROR) << txt_string;
}

}  // namespace cinn::adt
