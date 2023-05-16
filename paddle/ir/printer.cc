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

#include "paddle/ir/builtin_attribute.h"
#include "paddle/ir/operation.h"
#include "paddle/ir/program.h"
#include "paddle/ir/value.h"

namespace ir {

namespace {
constexpr char newline[] = "/n";
}  // namespace

class ProgramPrinter {
 public:
  explicit ProgramPrinter(std::ostream& os) : os(os), cur_var_number(0) {}

  void Print(ir::Program& program) {
    VLOG(0) << "ffff1 " << &program << " " << program.ops().size();
    for (auto* op : program.ops()) {
      VLOG(0) << "ffff1.1 " << op->op_name();
      PrintOperation(op);
      os << newline;
    }
  }

  template <typename ForwardIterator,
            typename UnaryFunctor,
            typename NullFunctor>
  void PrintInterleave(ForwardIterator begin,
                       ForwardIterator end,
                       UnaryFunctor print_func,
                       NullFunctor between_func) {
    print_func(*begin);
    begin++;
    for (; begin != end; begin++) {
      between_func();
      print_func(*begin);
    }
  }

  void PrintType(ir::Type type) {}

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
    VLOG(0) << "ffff2 " << op->op_name();
    PrintOpResult(op);  // TODO(lyk): add API to get opresults directly
    os << " = ";
    VLOG(0) << "ffff3";

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
    auto num_op_result = op->num_results();
    std::vector<ir::OpResult> op_results;
    op_results.reserve(num_op_result);
    for (size_t idx = 0; idx < num_op_result; idx++) {
      op_results.push_back(op->GetResultByIndex(idx));
    }
    VLOG(0) << "ffff5";
    PrintInterleave(
        op_results.begin(),
        op_results.end(),
        [this](ir::Value v) { this->PrintValue(v); },
        [this]() { this->os << ","; });
  }

  void PrintAttribute(ir::Operation* op) { os << "{Attribute PlaceHolder}"; }

  void PrintOpOperands(ir::Operation* op) { os << "Operands PlaceHolder"; }
  void PrintOperandsType(ir::Operation* op) {
    os << "OperandsType PlaceHolder";
  }

  void PrintOpReturnType(ir::Operation* op) {
    os << "OpReturnType PlaceHolder";
  }

 private:
  std::ostream& os;
  size_t cur_var_number;
  std::unordered_map<const void*, std::string> aliases;
};

std::ostream& operator<<(std::ostream& os, Program& program) {
  ProgramPrinter printer(os);
  printer.Print(program);
  return os;
}

}  // namespace ir
