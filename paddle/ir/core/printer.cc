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

#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/ir/core/builtin_type.h"
#include "paddle/ir/core/dialect.h"
#include "paddle/ir/core/operation.h"
#include "paddle/ir/core/program.h"
#include "paddle/ir/core/value.h"

namespace ir {

namespace {
constexpr char newline[] = "\n";

template <typename ForwardIterator, typename UnaryFunctor, typename NullFunctor>
void PrintInterleave(ForwardIterator begin,
                     ForwardIterator end,
                     UnaryFunctor print_func,
                     NullFunctor between_func) {
  if (begin == end) return;
  print_func(*begin);
  begin++;
  for (; begin != end; begin++) {
    between_func();
    print_func(*begin);
  }
}

}  // namespace

class Printer {
 public:
  explicit Printer(std::ostream& os) : os(os) {}

  void PrintType(ir::Type type) {
    if (type.isa<ir::Float16Type>()) {
      os << "f16";
    } else if (type.isa<ir::Float32Type>()) {
      os << "f32";
    } else if (type.isa<ir::Float64Type>()) {
      os << "f64";
    } else if (type.isa<ir::Int16Type>()) {
      os << "i16";
    } else if (type.isa<ir::Int32Type>()) {
      os << "i32";
    } else if (type.isa<ir::Int64Type>()) {
      os << "i64";
    } else if (type.isa<ir::VectorType>()) {
      os << "vec<";
      auto inner_types = type.dyn_cast<ir::VectorType>().data();
      PrintInterleave(
          inner_types.begin(),
          inner_types.end(),
          [this](ir::Type v) { this->PrintType(v); },
          [this]() { this->os << ","; });
      os << ">";
    } else {
      auto& dialect = type.dialect();
      dialect.PrintType(type, os);
    }
  }

 public:
  std::ostream& os;
};

void Type::print(std::ostream& os) const {
  if (!*this) {
    os << "<!TypeNull>";
    return;
  }
  Printer p(os);
  p.PrintType(*this);
}

class ProgramPrinter : public Printer {
 public:
  explicit ProgramPrinter(std::ostream& os) : Printer(os), cur_var_number(0) {}

  void Print(ir::Program& program) {
    auto iterator = program.block()->begin();
    while (iterator != program.block()->end()) {
      PrintOperation(*iterator);
      os << newline;
      iterator++;
    }
  }

  void PrintValue(ir::Value v) {
    const void* key = static_cast<const void*>(v.impl());
    auto ret = aliases.find(key);
    if (ret != aliases.end()) {
      os << ret->second;
      return;
    }

    std::string new_name = "%" + std::to_string(cur_var_number);
    cur_var_number++;
    aliases[key] = new_name;
    os << new_name;
  }

  /// @brief print operation
  /// @param op
  /// @example
  void PrintOperation(ir::Operation* op) {
    PrintOpResult(op);  // TODO(lyk): add API to get opresults directly
    os << " = ";

    os << "\"" << op->op_name() << "\"";
    PrintOpOperands(op);  // TODO(lyk): add API to get operands directly

    PrintAttribute(op);
    os << " : ";

    // PrintOpSingature
    PrintOperandsType(op);
    os << " -> ";
    PrintOpReturnType(op);  // TODO(lyk): add API to get opresults directly
  }

  void PrintOpResult(ir::Operation* op) {
    os << " (";
    auto num_op_result = op->num_results();
    std::vector<ir::OpResult> op_results;
    op_results.reserve(num_op_result);
    for (size_t idx = 0; idx < num_op_result; idx++) {
      op_results.push_back(op->GetResultByIndex(idx));
    }
    PrintInterleave(
        op_results.begin(),
        op_results.end(),
        [this](ir::Value v) { this->PrintValue(v); },
        [this]() { this->os << ","; });
    os << ") ";
  }

  void PrintAttribute(ir::Operation* op) { os << " { ATTRIBUTE } "; }

  void PrintOpOperands(ir::Operation* op) {
    os << " (";
    auto num_op_operands = op->num_operands();
    std::vector<ir::Value> op_operands;
    op_operands.reserve(num_op_operands);
    for (size_t idx = 0; idx < num_op_operands; idx++) {
      op_operands.push_back(op->GetOperandByIndex(idx).impl()->source());
    }
    PrintInterleave(
        op_operands.begin(),
        op_operands.end(),
        [this](ir::Value v) { this->PrintValue(v); },
        [this]() { this->os << ","; });
    os << ") ";
  }

  void PrintOperandsType(ir::Operation* op) {
    auto num_op_operands = op->num_operands();
    std::vector<ir::Type> op_operand_types;
    op_operand_types.reserve(num_op_operands);
    for (size_t idx = 0; idx < num_op_operands; idx++) {
      op_operand_types.push_back(
          op->GetOperandByIndex(idx).impl()->source().type());
    }
    PrintInterleave(
        op_operand_types.begin(),
        op_operand_types.end(),
        [this](ir::Type t) { this->PrintType(t); },
        [this]() { this->os << ","; });
  }

  void PrintOpReturnType(ir::Operation* op) {
    auto num_op_result = op->num_results();
    std::vector<ir::Type> op_result_types;
    op_result_types.reserve(num_op_result);
    for (size_t idx = 0; idx < num_op_result; idx++) {
      op_result_types.push_back(op->GetResultByIndex(idx).type());
    }
    PrintInterleave(
        op_result_types.begin(),
        op_result_types.end(),
        [this](ir::Type t) { this->PrintType(t); },
        [this]() { this->os << ","; });
  }

 private:
  size_t cur_var_number;
  std::unordered_map<const void*, std::string> aliases;
};

std::ostream& operator<<(std::ostream& os, Program& program) {
  ProgramPrinter printer(os);
  printer.Print(program);
  return os;
}

}  // namespace ir
