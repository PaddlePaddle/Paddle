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

#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"

#include <vector>
#include "paddle/pir/core/builtin_type.h"
#include "paddle/pir/core/op_base.h"

namespace cinn {
namespace dialect {

const char *GroupOp::attributes_name[GroupOp::attributes_num] = {"group_info"};

// TODO(Aurlius84): Need to figure out how to rebuild relation info of ops outer
// GroupOp
void GroupOp::Build(pir::Builder &builder,
                    pir::OperationArgument &argument,
                    const std::vector<pir::Type> &output_types) {
  argument.AddRegion(nullptr);
  argument.output_types = output_types;
}

pir::Block *GroupOp::Block() {
  pir::Region &region = (*this)->region(0);
  if (region.empty()) region.emplace_back();
  return region.front();
}

std::vector<pir::Operation *> GroupOp::Ops() {
  auto *block = this->Block();
  return std::vector<pir::Operation *>(block->begin(), block->end());
}

void GroupOp::Verify() {}

void GroupOp::Print(pir::IrPrinter &printer) {
  auto &os = printer.os;
  auto op = operation();
  printer.PrintOpResult(op);
  os << " = " << name();
  printer.PrintOpOperands(op);
  os << " -> ";
  printer.PrintOpReturnType(op);
  os << " {";
  for (auto &sub_op : Ops()) {
    os << "\n";
    printer.PrintOperation(sub_op);
  }
  os << " \n }";
}

}  // namespace dialect
}  // namespace cinn

IR_DEFINE_EXPLICIT_TYPE_ID(cinn::dialect::GroupOp)
