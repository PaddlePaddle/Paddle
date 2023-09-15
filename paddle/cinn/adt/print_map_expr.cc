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

#include "paddle/cinn/adt/print_map_expr.h"

namespace cinn::adt {

namespace {

constexpr std::size_t kIndentSpaceSize = 2;

inline void AppendIndentSpace(std::size_t indent_size, std::string* string) {
  for (std::size_t i = 0; i < indent_size; ++i) {
    for (std::size_t j = 0; j < kIndentSpaceSize; ++j) {
      *string += ' ';
    }
  }

}

template<typename DoEachT>
void VisitEachArg(const List<Arg>& out_args, const List<Arg>& in_args, const DoEachT& DoEach) {
  for (const auto& out_arg : *out_args) {
    DoEach(out_arg, kOut<bool>{true});
  }
  for (const auto& in_arg : *in_args) {
    DoEach(out_arg, kOut<bool>{false});
  }
}

}

void ToTxtString(const Tensor& tensor, std::string* string) {
  CHECK(tensor.Has<adapter::Tensor>());
  *string += "t_";
  *string += tensor.Get<adapter::Tensor>().node_data->id();
}

void ToTxtString(const OpStmt& op_stmt, std::size_t indent_size, std::string* string) {
  const auto& [op, in_args, out_args] = op_stmt.tuple();
  CHECK(op->Has<const hlir::framework::Node*>()); 
  AppendIndentSpace(indent_size, &string);
  *string += op->Get<const hlir::framework::Node*>()->op()->name;
  *string += "("
  std::size_t count = 0;
  VisitEachArg(out_args.value(), in_args.value(), [&](const auto& arg, const auto& as_output){
    if (count++ > 0) { *string += ", "; }
    if (as_output.value()) { *string += "&"; }
    ToTxtString(arg, string);
  });
  *string += ")";
}

}

namespace cinn::adt {

void PrintMapExpr(const MapExpr& map_expr) {}

}  // namespace cinn::adt
