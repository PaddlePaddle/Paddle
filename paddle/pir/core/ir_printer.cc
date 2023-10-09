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

#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/pir/core/block.h"
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/builtin_type.h"
#include "paddle/pir/core/dialect.h"
#include "paddle/pir/core/ir_printer.h"
#include "paddle/pir/core/operation.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/core/utils.h"
#include "paddle/pir/core/value.h"

namespace pir {

namespace {
constexpr char newline[] = "\n";  // NOLINT
}  // namespace

void BasicIrPrinter::PrintType(Type type) {
  if (!type) {
    os_ << "<<NULL TYPE>>";
    return;
  }

  if (type.isa<BFloat16Type>()) {
    os_ << "bf16";
  } else if (type.isa<Float16Type>()) {
    os_ << "f16";
  } else if (type.isa<Float32Type>()) {
    os_ << "f32";
  } else if (type.isa<Float64Type>()) {
    os_ << "f64";
  } else if (type.isa<BoolType>()) {
    os_ << "b";
  } else if (type.isa<Int8Type>()) {
    os_ << "i8";
  } else if (type.isa<UInt8Type>()) {
    os_ << "u8";
  } else if (type.isa<Int16Type>()) {
    os_ << "i16";
  } else if (type.isa<Int32Type>()) {
    os_ << "i32";
  } else if (type.isa<Int64Type>()) {
    os_ << "i64";
  } else if (type.isa<IndexType>()) {
    os_ << "index";
  } else if (type.isa<Complex64Type>()) {
    os_ << "c64";
  } else if (type.isa<Complex128Type>()) {
    os_ << "c128";
  } else if (type.isa<VectorType>()) {
    os_ << "vec[";
    auto inner_types = type.dyn_cast<VectorType>().data();
    PrintInterleave(
        inner_types.begin(),
        inner_types.end(),
        [this](Type v) { this->PrintType(v); },
        [this]() { this->os_ << ","; });
    os_ << "]";
  } else {
    auto& dialect = type.dialect();
    dialect.PrintType(type, os_);
  }
}

void BasicIrPrinter::PrintAttribute(Attribute attr) {
  if (!attr) {
    os_ << "<#AttrNull>";
    return;
  }

  if (auto s = attr.dyn_cast<StrAttribute>()) {
    std::string s_val = s.AsString();
    std::string replacement = "\\\"";
    std::string search = "\"";
    size_t found = s_val.find(search);
    while (found != std::string::npos) {
      s_val.replace(found, search.length(), replacement);
      found = s_val.find(search, found + replacement.length());
    }
    os << "\"" << s_val << "\"";
  } else if (auto b = attr.dyn_cast<BoolAttribute>()) {
    if (b.data()) {
      os_ << "true";
    } else {
      os_ << "false";
    }
  } else if (auto f = attr.dyn_cast<FloatAttribute>()) {
    os_ << "(Float)" << f.data();
  } else if (auto d = attr.dyn_cast<DoubleAttribute>()) {
    os_ << "(Double)" << d.data();
  } else if (auto i = attr.dyn_cast<Int32Attribute>()) {
    os_ << "(Int32)" << i.data();
  } else if (auto i = attr.dyn_cast<Int64Attribute>()) {
    os_ << "(Int64)" << i.data();
  } else if (auto p = attr.dyn_cast<PointerAttribute>()) {
    os_ << "(Pointer)" << p.data();
  } else if (auto arr = attr.dyn_cast<ArrayAttribute>()) {
    const auto& vec = arr.AsVector();
    os << "[";
    PrintInterleave(
        vec.begin(),
        vec.end(),
        [this](Attribute v) { this->PrintAttribute(v); },
        [this]() { this->os_ << ","; });
    os_ << "]";
  } else if (auto type = attr.dyn_cast<TypeAttribute>()) {
    os_ << type.data();
  } else {
    auto& dialect = attr.dialect();
    dialect.PrintAttribute(attr, os_);
  }
}

void IrPrinter::PrintProgram(const Program* program) {
  auto top_level_op = program->module_op();
  for (size_t i = 0; i < top_level_op->num_regions(); ++i) {
    auto& region = top_level_op->region(i);
    PrintRegion(region);
  }
}

void IrPrinter::PrintOperation(Operation* op) {
  if (auto* dialect = op->dialect()) {
    dialect->PrintOperation(op, *this);
    return;
  }

  PrintGeneralOperation(op);
}

void IrPrinter::PrintOperationWithNoRegion(Operation* op) {
  // TODO(lyk): add API to get opresults directly
  PrintOpResult(op);
  os_ << " =";

  os_ << " \"" << op->name() << "\"";

  // TODO(lyk): add API to get operands directly
  PrintOpOperands(op);

  PrintAttributeMap(op);
  os_ << " :";

  // PrintOpSingature
  PrintOperandsType(op);
  os_ << " -> ";

  // TODO(lyk): add API to get opresults directly
  PrintOpReturnType(op);
}

void IrPrinter::PrintGeneralOperation(Operation* op) {
  PrintOperationWithNoRegion(op);
  if (!options_.print_regions) {
    return;
  }

  if (op->num_regions() > 0) {
    os_ << newline;
  }
  for (size_t i = 0; i < op->num_regions(); ++i) {
    auto& region = op->region(i);
    PrintRegion(region);
  }
}

void IrPrinter::PrintRegion(const Region& region) {
  for (auto block : region) {
    PrintBlock(block);
  }
}

void IrPrinter::PrintBlock(const Block* block) {
  os_ << "{\n";
  for (auto item : *block) {
    PrintOperation(item);
    os_ << newline;
  }
  os_ << "}\n";
}

void IrPrinter::PrintValue(Value v) {
  if (!v) {
    os_ << "<<NULL VALUE>>";
    return;
  }
  const void* key = static_cast<const void*>(v.impl());
  auto ret = aliases_.find(key);
  if (ret != aliases_.end()) {
    os_ << ret->second;
    return;
  }

  std::string new_name = "%" + std::to_string(cur_var_number_);
  cur_var_number_++;
  aliases_[key] = new_name;
  os_ << new_name;
}

void IrPrinter::PrintOpResult(Operation* op) {
  os_ << " (";
  auto num_op_result = op->num_results();
  std::vector<OpResult> op_results;
  op_results.reserve(num_op_result);
  for (size_t idx = 0; idx < num_op_result; idx++) {
    op_results.push_back(op->result(idx));
  }
  PrintInterleave(
      op_results.begin(),
      op_results.end(),
      [this](Value v) { this->PrintValue(v); },
      [this]() { this->os_ << ", "; });
  os_ << ")";
}

void IrPrinter::PrintAttributeMap(Operation* op) {
  AttributeMap attributes = op->attributes();

  std::vector<std::string> attribute_will_be_printed;
  for (size_t i = 0u; i < op->info().num_attributes(); i++) {
    attribute_will_be_printed.push_back(op->info().attribute_name(i));
  }
  for (const auto& attr_name : options_.custom_attrs_white_list) {
    if (attributes.count(attr_name) == 0) {
      continue;
    }
    attribute_will_be_printed.push_back(attr_name);
  }

  os_ << " {";

  PrintInterleave(
      attribute_will_be_printed.begin(),
      attribute_will_be_printed.end(),
      [this, &attributes](const std::string& attr_name) {
        this->os_ << attr_name;
        this->os_ << ":";
        this->PrintAttribute(attributes.at(attr_name));
      },
      [this]() { this->os_ << ","; });

  os_ << "}";
}

void IrPrinter::PrintOpOperands(Operation* op) {
  os_ << " (";
  auto num_op_operands = op->num_operands();
  std::vector<Value> op_operands;
  op_operands.reserve(num_op_operands);
  for (size_t idx = 0; idx < num_op_operands; idx++) {
    op_operands.push_back(op->operand_source(idx));
  }
  PrintInterleave(
      op_operands.begin(),
      op_operands.end(),
      [this](Value v) { this->PrintValue(v); },
      [this]() { this->os_ << ", "; });
  os_ << ")";
}

void IrPrinter::PrintOperandsType(Operation* op) {
  auto num_op_operands = op->num_operands();
  std::vector<Type> op_operand_types;
  op_operand_types.reserve(num_op_operands);
  for (size_t idx = 0; idx < num_op_operands; idx++) {
    auto op_operand = op->operand(idx);
    if (op_operand) {
      op_operand_types.push_back(op_operand.type());
    } else {
      op_operand_types.emplace_back();
    }
  }
  os_ << " (";
  PrintInterleave(
      op_operand_types.begin(),
      op_operand_types.end(),
      [this](Type t) { this->PrintType(t); },
      [this]() { this->os_ << ", "; });
  os_ << ")";
}

void IrPrinter::PrintOpReturnType(Operation* op) {
  auto num_op_result = op->num_results();
  std::vector<Type> op_result_types;
  op_result_types.reserve(num_op_result);
  for (size_t idx = 0; idx < num_op_result; idx++) {
    auto op_result = op->result(idx);
    if (op_result) {
      op_result_types.push_back(op_result.type());
    } else {
      op_result_types.emplace_back(nullptr);
    }
  }
  PrintInterleave(
      op_result_types.begin(),
      op_result_types.end(),
      [this](Type t) { this->PrintType(t); },
      [this]() { this->os_ << ", "; });
}

void Dialect::PrintOperation(Operation* op, IrPrinter& printer) const {
  printer.PrintGeneralOperation(op);
}

void Program::Print(std::ostream& os) const { Print(os, PrinterOptions()); }

void Program::Print(std::ostream& os, const PrinterOptions& options) const {
  IrPrinter printer(os, options);
  printer.PrintProgram(this);
}

void Operation::Print(std::ostream& os) { Print(os, PrinterOptions()); }

void Operation::Print(std::ostream& os, const PrinterOptions options) {
  IrPrinter printer(os, options);
  printer.PrintOperation(this);
}

void Value::Print(std::ostream& os) const {
  IrPrinter printer(os);
  printer.PrintValue(*this);
}

void Type::Print(std::ostream& os) const { Print(os, PrinterOptions()); }

void Type::Print(std::ostream& os, const PrinterOptions& options) const {
  BasicIrPrinter printer(os, options);
  printer.PrintType(*this);
}

void Attribute::Print(std::ostream& os) const { Print(os, PrinterOptions()); }

void Attribute::Print(std::ostream& os, const PrinterOptions& options) const {
  BasicIrPrinter printer(os, options);
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

std::ostream& operator<<(std::ostream& os, const Program& prog) {
  prog.Print(os);
  return os;
}

}  // namespace pir
