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

#include "paddle/cinn/adt/print_map_expr.h"

namespace cinn::adt {

namespace {

// void ToTxtString(const OpStmt& op_stmt, std::size_t indent_size, std::string*
// string);

void ToTextString(const LoopType& loop_type,
                  std::size_t indent_size,
                  std::string* string) {
  loop_type >> match {
    [&](const S0x&) { *string += "blockIdx.x"; },
        [&](const S0y&) { *string += "blockIdx.y"; },
        [&](const S0z&) { *string += "blockIdx.z"; },
        [&](const S1x&) { *string += "threadIdx.x"; },
        [&](const S1y&) { *string += "threadIdx.y"; },
        [&](const S1z&) { *string += "threadIdx.z"; },
        [&](const Temporal& temporal) { *string += temporal.iter_var_name(); },
        [&](const Vectorize& vectorize) {
          *string += vectorize.iter_var_name();
        },
        [&](const Unroll& unroll) { *string += unroll.iter_var_name(); },
  }
}

void ToTextString(const ScheduleDescriptor& schedule_descriptor,
                  std::size_t indent_size,
                  std::string* string) {
  *string += std::string(" ", indent_size * kIndentSpaceSize);
  for (const auto& loop_descriptor : *schedule_descriptor) {
    const auto& [loop_type, loop_size] = loop_descriptor.tuple();
    ToTextString(loop_type, indent_size, string);
    CHECK(loop_size.Has<std::int64_t>());
    *string += ", " + std::to_string(loop_size.Get<std::int64_t>());
  }
}

void ToTextString(const MapStmt<Stmt>& map_stmt,
                  std::size_t indent_size,
                  std::string* string);

void ToTextString(const Stmt& stmt,
                  std::size_t indent_size,
                  std::string* string) {
  std::visit([&](const auto& impl) { ToTextString(stmt, indent_size, string); },
             stmt.variant());
}

void ToTextString(const MapStmt<Stmt>& map_stmt,
                  std::size_t indent_size,
                  std::string* string) {
  const auto& [schedule_descriptor, stmts] = map_stmt.tuple();
  *string += std::string(" ", indent_size * kIndentSpaceSize) + "{\n";
  ToTextString(schedule_descriptor, indent_size + 1, string);
  for (const auto& stmt : *stmts) {
    ToTextString(stmt, indent_size + 1, string);
  }
  *string += std::string(" ", indent_size * kIndentSpaceSize) + "}\n";
}

void ToTextString(const AnchoredMapStmt& anchored_map_stmt,
                  std::size_t indent_size,
                  std::string* string) {
  const auto& [map_stmt, anchor_tensor, _] = anchored_map_stmt.tuple();
  ToTextString(map_stmt, indent_size, string);
}

void ToTextString(const List<Tensor>& tensors,
                  std::size_t indent_size,
                  std::string* string) {}

void ToTextString(const MapExpr& map_expr,
                  std::size_t indent_size,
                  std::string* txt_string) {
  const auto& [anchored_map_stmts, inputs, outputs] = map_expr.tuple();

  *txt_string += "Input tensors: \n";
  ToTextString(inputs.value(), 0, &txt_string);
  *txt_string += "Output tensors: \n";
  ToTextString(outputs.value(), 0, &txt_string);

  for (const auto& anchored_map_stmt : *anchored_map_stmts) {
    *txt_string += "\n";
    ToTextString(anchored_map_stmts, 0, &txt_string);
  }
}

}  // namespace

void PrintMapExpr(const MapExpr& map_expr) {
  std::string txt_string{};
  ToTextString(map_expr, 0, &txt_string);
}

}  // namespace cinn::adt
