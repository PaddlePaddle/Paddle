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

#include <list>
#include <ostream>
#include <string>
#include <unordered_map>

#include "paddle/ir/core/block.h"
#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/ir/core/builtin_type.h"
#include "paddle/ir/core/dialect.h"
#include "paddle/ir/core/ir_printer.h"
#include "paddle/ir/core/operation.h"
#include "paddle/ir/core/program.h"
#include "paddle/ir/core/utils.h"
#include "paddle/ir/core/value.h"

namespace ir {

namespace {
constexpr char newline[] = "\n";
}  // namespace

void BasicIrPrinter::PrintType(Type type) {
  if (!type) {
    os << "<<NULL TYPE>>";
    return;
  }

  if (type.isa<Float16Type>()) {
    os << "f16";
  } else if (type.isa<Float32Type>()) {
    os << "f32";
  } else if (type.isa<Float64Type>()) {
    os << "f64";
  } else if (type.isa<Int16Type>()) {
    os << "i16";
  } else if (type.isa<Int32Type>()) {
    os << "i32";
  } else if (type.isa<Int64Type>()) {
    os << "i64";
  } else if (type.isa<VectorType>()) {
    os << "vec[";
    auto inner_types = type.dyn_cast<VectorType>().data();
    PrintInterleave(
        inner_types.begin(),
        inner_types.end(),
        [this](Type v) { this->PrintType(v); },
        [this]() { this->os << ","; });
    os << "]";
  } else {
    auto& dialect = type.dialect();
    dialect.PrintType(type, os);
  }
}

void BasicIrPrinter::PrintAttribute(const Attribute& attr) {
  if (!attr) {
    os << "<#AttrNull>";
    return;
  }

  if (auto s = attr.dyn_cast<StrAttribute>()) {
    os << s.data();
  } else if (auto b = attr.dyn_cast<BoolAttribute>()) {
    os << b.data();
  } else if (auto f = attr.dyn_cast<FloatAttribute>()) {
    os << f.data();
  } else if (auto d = attr.dyn_cast<DoubleAttribute>()) {
    os << d.data();
  } else if (auto i = attr.dyn_cast<Int32_tAttribute>()) {
    os << i.data();
  } else if (auto i = attr.dyn_cast<Int64_tAttribute>()) {
    os << i.data();
  } else if (auto p = attr.dyn_cast<PointerAttribute>()) {
    os << p.data();
  } else if (auto arr = attr.dyn_cast<ArrayAttribute>()) {
    const auto& vec = arr.data();
    os << "array[";
    PrintInterleave(
        vec.begin(),
        vec.end(),
        [this](Attribute v) { this->PrintAttribute(v); },
        [this]() { this->os << ","; });
    os << "]";
  } else {
    auto& dialect = attr.dialect();
    dialect.PrintAttribute(attr, os);
  }
}

void IrPrinter::PrintProgram(Program* program) {
  auto top_level_op = program->module_op();
  for (size_t i = 0; i < top_level_op->num_regions(); ++i) {
    auto& region = top_level_op->GetRegion(i);
    for (auto it = region.begin(); it != region.end(); ++it) {
      auto* block = *it;
      os << "{\n";
      for (auto it = block->begin(); it != block->end(); ++it) {
        PrintOperation(*it);
        os << newline;
      }
      os << "}\n";
    }
  }
}

void IrPrinter::PrintOperation(Operation* op) {
  if (auto* dialect = op->dialect()) {
    if (auto print_fn = dialect->OperationPrinter()) {
      print_fn(op, *this);
      return;
    }
  }

  PrintGeneralOperation(op);
}

void IrPrinter::PrintGeneralOperation(Operation* op) {
  // TODO(lyk): add API to get opresults directly
  PrintOpResult(op);
  os << " =";

  os << " \"" << op->name() << "\"";

  // TODO(lyk): add API to get operands directly
  PrintOpOperands(op);

  PrintAttributeMap(op);
  os << " :";

  // PrintOpSingature
  PrintOperandsType(op);
  os << " -> ";

  // TODO(lyk): add API to get opresults directly
  PrintOpReturnType(op);
}

void IrPrinter::PrintFullOperation(Operation* op) {
  PrintOperation(op);
  os << newline;
  for (size_t i = 0; i < op->num_regions(); ++i) {
    auto& region = op->GetRegion(i);
    for (auto it = region.begin(); it != region.end(); ++it) {
      auto* block = *it;
      os << "{\n";
      for (auto it = block->begin(); it != block->end(); ++it) {
        PrintOperation(*it);
        os << newline;
      }
      os << "}\n";
    }
  }
}

void IrPrinter::PrintValue(Value v) {
  if (!v) {
    os << "<<NULL VALUE>>";
    return;
  }
  const void* key = static_cast<const void*>(v.impl());
  auto ret = aliases_.find(key);
  if (ret != aliases_.end()) {
    os << ret->second;
    return;
  }

  std::string new_name = "%" + std::to_string(cur_var_number_);
  cur_var_number_++;
  aliases_[key] = new_name;
  os << new_name;
}

void IrPrinter::PrintOpResult(Operation* op) {
  os << " (";
  auto num_op_result = op->num_results();
  std::vector<OpResult> op_results;
  op_results.reserve(num_op_result);
  for (size_t idx = 0; idx < num_op_result; idx++) {
    op_results.push_back(op->GetResultByIndex(idx));
  }
  PrintInterleave(
      op_results.begin(),
      op_results.end(),
      [this](Value v) { this->PrintValue(v); },
      [this]() { this->os << ", "; });
  os << ")";
}

void IrPrinter::PrintAttributeMap(Operation* op) {
  os << " {";

  PrintInterleave(
      op->attributes().begin(),
      op->attributes().end(),
      [this](std::pair<std::string, Attribute> it) {
        this->os << it.first;
        this->os << ":";
        this->PrintAttribute(it.second);
      },
      [this]() { this->os << ","; });

  os << "}";
}

void IrPrinter::PrintOpOperands(Operation* op) {
  os << " (";
  auto num_op_operands = op->num_operands();
  std::vector<Value> op_operands;
  op_operands.reserve(num_op_operands);
  for (size_t idx = 0; idx < num_op_operands; idx++) {
    op_operands.push_back(op->GetOperandByIndex(idx).source());
  }
  PrintInterleave(
      op_operands.begin(),
      op_operands.end(),
      [this](Value v) { this->PrintValue(v); },
      [this]() { this->os << ", "; });
  os << ")";
}

void IrPrinter::PrintOperandsType(Operation* op) {
  auto num_op_operands = op->num_operands();
  std::vector<Type> op_operand_types;
  op_operand_types.reserve(num_op_operands);
  for (size_t idx = 0; idx < num_op_operands; idx++) {
    auto op_operand = op->GetOperandByIndex(idx);
    if (op_operand) {
      op_operand_types.push_back(op->GetOperandByIndex(idx).source().type());
    } else {
      op_operand_types.push_back(Type(nullptr));
    }
  }
  os << " (";
  PrintInterleave(
      op_operand_types.begin(),
      op_operand_types.end(),
      [this](Type t) { this->PrintType(t); },
      [this]() { this->os << ", "; });
  os << ")";
}

void IrPrinter::PrintOpReturnType(Operation* op) {
  auto num_op_result = op->num_results();
  std::vector<Type> op_result_types;
  op_result_types.reserve(num_op_result);
  for (size_t idx = 0; idx < num_op_result; idx++) {
    auto op_result = op->GetResultByIndex(idx);
    if (op_result) {
      op_result_types.push_back(op_result.type());
    } else {
      op_result_types.push_back(Type(nullptr));
    }
  }
  PrintInterleave(
      op_result_types.begin(),
      op_result_types.end(),
      [this](Type t) { this->PrintType(t); },
      [this]() { this->os << ", "; });
}

void Program::Print(std::ostream& os) {
  IrPrinter printer(os);
  printer.PrintProgram(this);
}

void Operation::Print(std::ostream& os) {
  IrPrinter printer(os);
  printer.PrintFullOperation(this);
}

void Type::Print(std::ostream& os) const {
  BasicIrPrinter printer(os);
  printer.PrintType(*this);
}

void Attribute::Print(std::ostream& os) const {
  BasicIrPrinter printer(os);
  printer.PrintAttribute(*this);
}

std::ostream& operator<<(std::ostream& os, Type type) {
  type.Print(os);
  return os;
}

std::ostream& operator<<(std::ostream& os, Attribute attr) {
  attr.Print(os);
  return os;
}

}  // namespace ir
