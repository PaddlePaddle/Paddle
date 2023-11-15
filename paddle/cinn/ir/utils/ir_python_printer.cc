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
#include <algorithm>
#include <iomanip>
#include <limits>
#include <string>
#include <vector>

#include "paddle/cinn/common/type.h"
#include "paddle/cinn/ir/buffer.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/ir_visitor.h"
#include "paddle/cinn/ir/lowered_func.h"
#include "paddle/cinn/ir/module.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/ir/utils/ir_python_printer.h"
#include "paddle/cinn/optim/ir_simplify.h"
#include "paddle/cinn/runtime/intrinsic.h"
#include "paddle/cinn/utils/string.h"
namespace cinn {
namespace ir {
using common::bfloat16;
using common::float16;

void IrPythonPrinter::Print(const Expr &e) {
  IRVisitorRequireReImpl::Visit(&e);
  os_ << str_;
  str_ = "";
}
void IrPythonPrinter::Print(const std::vector<Expr> &exprs,
                            const std::string &splitter) {
  for (std::size_t i = 0; !exprs.empty() && i + 1 < exprs.size(); i++) {
    Visit(exprs[i]);
    str_ += splitter;
  }
  if (!exprs.empty()) Visit(exprs.back());
  os_ << str_;
  str_ = "";
}
void IrPythonPrinter::DoIndent() {
  str_ += "\n";
  str_ += std::string(indent_, ' ');
}
void IrPythonPrinter::IncIndent() { indent_ += indent_unit; }
void IrPythonPrinter::DecIndent() { indent_ -= indent_unit; }

// Type
void IrPythonPrinter::Visit(const IntImm *x) {
  str_ += std::to_string(x->value);
}

void IrPythonPrinter::Visit(const UIntImm *x) {
  if (x->type().is_uint(1)) {
    if (x->value) {
      str_ += "True";
    } else {
      str_ += "False";
    }
  } else {
    str_ += std::to_string(x->value);
  }
}
void IrPythonPrinter::Visit(const FloatImm *x) {
  std::ostringstream ss;
  ss << x->value;
  str_ += ss.str();
}
void IrPythonPrinter::Visit(const StringImm *x) {
  str_ += "\"";
  str_ += x->value;
  str_ += "\"";
}

// Var
void IrPythonPrinter::Visit(const _Var_ *x) { str_ += x->name; }

// Buffer
void IrPythonPrinter::Visit(const _Buffer_ *x) {
  std::vector<std::string> dim_names;
  std::transform(x->shape.begin(),
                 x->shape.end(),
                 std::back_inserter(dim_names),
                 [&](const Expr &x) { return utils::GetStreamCnt(x); });

  str_ += x->name;
}

// Expr
void IrPythonPrinter::Visit(const Cast *x) {
  str_ += x->type().to_string();
  str_ += "(";
  Visit(x->v());
  str_ += ")";
}

// Expr BinOp
void IrPythonPrinter::Visit(const Add *x) { PrintBinaryOp("+", x); }
void IrPythonPrinter::Visit(const Sub *x) { PrintBinaryOp("-", x); }
void IrPythonPrinter::Visit(const Mul *x) { PrintBinaryOp("*", x); }
void IrPythonPrinter::Visit(const Div *x) { PrintBinaryOp("/", x); }
void IrPythonPrinter::Visit(const Mod *x) { PrintBinaryOp("%", x); }
void IrPythonPrinter::Visit(const Min *x) {
  str_ += "cinn_min(";
  Visit(x->a());
  str_ += ", ";
  Visit(x->b());
  str_ += ")";
}
void IrPythonPrinter::Visit(const Max *x) {
  str_ += "cinn_max(";
  Visit(x->a());
  str_ += ", ";
  Visit(x->b());
  str_ += ")";
}

void IrPythonPrinter::Visit(const EQ *x) { PrintBinaryOp("==", x); }
void IrPythonPrinter::Visit(const NE *x) { PrintBinaryOp("!=", x); }
void IrPythonPrinter::Visit(const LT *x) { PrintBinaryOp("<", x); }
void IrPythonPrinter::Visit(const LE *x) { PrintBinaryOp("<=", x); }
void IrPythonPrinter::Visit(const GT *x) { PrintBinaryOp(">", x); }
void IrPythonPrinter::Visit(const GE *x) { PrintBinaryOp(">=", x); }
void IrPythonPrinter::Visit(const And *x) { PrintBinaryOp("and", x); }
void IrPythonPrinter::Visit(const Or *x) { PrintBinaryOp("or", x); }

// Expr Unary Op
void IrPythonPrinter::Visit(const Not *x) {
  str_ += "not ";
  Visit(x->v());
}

void IrPythonPrinter::Visit(const Minus *x) {
  str_ += "-(";
  Visit(x->v());
  str_ += ")";
}

// Expr
void IrPythonPrinter::Visit(const Select *x) {
  str_ += "ir.Select.make(";
  Visit(x->condition);
  str_ += ", ";
  Visit(x->true_value);
  str_ += ", ";
  Visit(x->false_value);
  str_ += ")";
}

void IrPythonPrinter::Visit(const Load *x) {
  if (x->is_addr_tensor()) {
    auto *tensor = x->tensor.As<ir::_Tensor_>();
    CHECK(tensor);
    str_ += tensor->name;
  } else if (x->is_addr_scalar()) {
    Visit(x->tensor);
  } else {
    CINN_NOT_IMPLEMENTED
  }

  str_ += "[";
  for (std::size_t i = 0; i + 1 < x->indices.size(); i++) {
    Visit(x->indices[i]);
    str_ += ", ";
  }
  if (!x->indices.empty()) Visit(x->indices.back());
  str_ += "]";
}
void IrPythonPrinter::Visit(const Broadcast *x) {
  str_ += "ir.Broadcast.make(";
  Visit(x->value);
  str_ += ", ";
  str_ += std::to_string(x->lanes);
  str_ += ")";
}

void IrPythonPrinter::Visit(const Ramp *x) {
  str_ += "ir.Ramp(";
  Visit(x->base);
  str_ += ", ";
  Visit(x->stride);
  str_ += ", ";
  str_ += std::to_string(x->lanes);
  str_ += ")";
}
void IrPythonPrinter::Visit(const ir::Call *op) {
  if (op->call_type == CallType::Extern) {
    str_ += "lang.call_extern(";
    str_ += "\"";
    str_ += op->name;
    str_ += "\", [";
    if (!op->read_args.empty()) {
      for (std::size_t i = 0; i + 1 < op->read_args.size(); i++) {
        IrPythonPrinter::Visit(op->read_args[i]);
        str_ += ", ";
      }
      IrPythonPrinter::Visit(op->read_args.back());
    }

    if (!op->write_args.empty()) {
      if (!op->read_args.empty()) str_ += ", ";

      for (std::size_t i = 0; i + 1 < op->write_args.size(); i++) {
        IrPythonPrinter::Visit(op->write_args[i]);
        str_ += ", ";
      }
      IrPythonPrinter::Visit(op->write_args.back());
    }
    str_ += "], {";

    for (const auto &kv : op->attrs) {
      str_ += kv.first;
      absl::visit(
          [this](auto &&arg) {
            std::ostringstream ss;
            ss << arg;
            this->str_ += ss.str();
          },
          kv.second);
    }
    str_ += "}";
    str_ += ")";
  } else {
    IrPythonPrinter::Visit(op);
  }
}
void IrPythonPrinter::Visit(const Let *f) {
  CHECK(f->type().valid());
  str_ += f->type().to_string();
  str_ += " ";
  Visit(f->symbol);
  if (f->body.defined()) {
    str_ += " = ";
    Visit(f->body);
  }
}
// stmt
void IrPythonPrinter::Visit(const For *x) {
  // TODO(6clc): support different types of for loop in python script,
  // and currently only support the python native range()
  DoIndent();
  str_ += "for ";
  IrPythonPrinter::Visit(x->loop_var);
  str_ += " in range(";
  IrPythonPrinter::Visit(x->min);
  str_ += ", ";
  IrPythonPrinter::Visit(x->extent);
  str_ += "):";

  IncIndent();
  IrPythonPrinter::Visit(x->body);
  DecIndent();
}
void IrPythonPrinter::Visit(const Store *x) {
  if (x->is_addr_tensor()) {
    auto *tensor_node = x->tensor.As<ir::_Tensor_>();
    CHECK(tensor_node);
    str_ += tensor_node->name;
  } else if (x->is_addr_scalar()) {
    Visit(x->tensor);
  } else {
    CINN_NOT_IMPLEMENTED
  }

  str_ += "[";
  for (std::size_t i = 0; i + 1 < x->indices.size(); i++) {
    Visit(x->indices[i]);
    str_ += ", ";
  }
  if (!x->indices.empty()) Visit(x->indices.back());
  str_ += "] = ";
  Visit(x->value);
}
void IrPythonPrinter::Visit(const Alloc *x) {
  auto *buffer = x->destination.As<ir::_Buffer_>();
  CHECK(buffer);
  str_ += "alloc(";
  str_ += buffer->name;
  str_ += ", ";
  Visit(x->extents);
  str_ += ")";
}

void IrPythonPrinter::Visit(const IfThenElse *x) {
  DoIndent();
  str_ += "if (";
  IrPythonPrinter::Visit(x->condition);
  str_ += "):";
  IncIndent();
  IrPythonPrinter::Visit(x->true_case);
  DecIndent();

  if (x->false_case.defined()) {
    DoIndent();
    str_ += "else:";
    IncIndent();
    IrPythonPrinter::Visit(x->false_case);
    DecIndent();
  }
}

void IrPythonPrinter::Visit(const Block *x) {
  for (std::size_t i = 0; !x->stmts.empty() && i < x->stmts.size(); i++) {
    // DoIndent();
    IrPythonPrinter::Visit(x->stmts[i]);
  }
}

void IrPythonPrinter::Visit(const ScheduleBlock *x) {}

void IrPythonPrinter::Visit(const ScheduleBlockRealize *x) {
  auto *schedule_block = x->schedule_block.As<ScheduleBlock>();
  if (schedule_block->name == "root") {
    IrPythonPrinter::Visit(schedule_block->body);
    return;
  }
  DoIndent();
  str_ += "with ir.ScheduleBlockContext(\"";
  str_ += schedule_block->name;
  str_ += "\") as ";
  str_ += schedule_block->name;
  str_ += "_block:";
  IncIndent();
  // print block vars and bindings
  auto iter_vars = schedule_block->iter_vars;
  auto iter_values = x->iter_values;
  CHECK_EQ(iter_vars.size(), iter_values.size());
  if (!iter_vars.empty()) {
    DoIndent();
  }
  for (std::size_t i = 0; i < iter_vars.size(); i++) {
    if (i) str_ += ", ";
    str_ += iter_vars[i]->name;
  }
  if (!iter_vars.empty()) str_ += " = ir.AxisMap(\"";

  for (const Var &iter_var : iter_vars) {
    if (iter_var->is_reduce_axis)
      str_ += "R";
    else
      str_ += "S";
  }
  str_ += "\", [";
  for (std::size_t i = 0; i < iter_values.size(); i++) {
    if (i) str_ += ", ";
    IrPythonPrinter::Visit(iter_values[i]);
  }
  str_ += "]";
  if (!iter_vars.empty()) str_ += ")";
  DoIndent();
  IrPythonPrinter::Visit(schedule_block->body);
  DecIndent();
}
void IrPythonPrinter::Visit(const ir::_LoweredFunc_ *op) {
  str_ += "# from cinn import ir, lang, to_cinn_llir\n";
  str_ += "# from cinn.runtime.data_array import DataArray\n";
  str_ += "# from cinn.schedule import IRSchedule as sch\n";
  str_ += "def fn_";
  str_ += op->name;
  str_ += "(";

  std::vector<std::string> arg_names;
  auto &func_args = op->args;
  for (int i = 0; i < func_args.size(); i++) {
    if (i) {
      str_ += ", ";
    }
    if (func_args[i].is_buffer()) {
      str_ += func_args[i].name().substr(1);
      str_ += ": DataArray((";
      PrintShape(func_args[i].buffer_arg()->shape);
      str_ += "))";
    } else {
      str_ += func_args[i].name();
      str_ += ": ";
      str_ += "";
    }
  }
  str_ += "):";

  IncIndent();
  ir::IrPythonPrinter::Visit(op->body);
  DecIndent();
}
void IrPythonPrinter::Visit(const _Tensor_ *x) {
  str_ += x->name;
  str_ += ", ";
  str_ += "[";
  if (!x->shape.empty()) {
    for (std::size_t i = 0; i + 1 < x->shape.size(); i++) {
      Visit(x->shape[i]);
      str_ += ",";
    }
    Visit(x->shape.back());
  }
  str_ += "]";
}
void IrPythonPrinter::Visit(const PrimitiveNode *x) {
  str_ += x->name;
  str_ += "(";
  std::vector<std::string> args_repr;
  for (auto &args : x->arguments) {
    std::vector<std::string> arg_repr;
    for (auto &arg : args) {
      arg_repr.push_back(utils::GetStreamCnt(arg));
    }
    args_repr.push_back(utils::Join(arg_repr, ","));
  }

  str_ += utils::Join(args_repr, ",");
  str_ += ")";
}
void IrPythonPrinter::Visit(const IntrinsicOp *x) {
  switch (x->getKind()) {
#define __(op__)                                \
  case IntrinsicKind::k##op__:                  \
    Visit(llvm::dyn_cast<intrinsics::op__>(x)); \
    break;

    INTRINSIC_KIND_FOR_EACH(__)
#undef __
  }
}
void IrPythonPrinter::Visit(const Free *x) {
  DoIndent();
  auto *buffer = x->destination.As<ir::_Buffer_>();
  CHECK(buffer);
  str_ += "free(";
  str_ += buffer->name;
  str_ += ")";
}
void IrPythonPrinter::Visit(const PolyFor *x) {
  DoIndent();
  if (x->is_parallel()) {
    str_ += "parallel poly_for (";
  } else {
    str_ += "poly_for (";
  }
  Visit(x->iterator);
  str_ += ", ";
  Visit(x->init);
  str_ += ", ";
  Visit(x->condition);
  str_ += ", ";
  Visit(x->inc);
  DoIndent();
  Visit(x->body);
}
void IrPythonPrinter::Visit(const _Module_ *x) {}
void IrPythonPrinter::Visit(const _BufferRange_ *x) {
  auto *buffer = x->buffer.As<ir::_Buffer_>();
  CHECK(buffer);
  str_ += buffer->name;
  str_ += "[";
  for (std::size_t i = 0; i < x->ranges.size(); i++) {
    if (i) str_ += ", ";
    auto &range = x->ranges[i];
    str_ += range->name;
    str_ += "(";
    if (range->lower_bound.defined()) {
      Visit(range->lower_bound);
      str_ += ":";
    } else {
      str_ += "undefined:";
    }

    if (range->upper_bound.defined()) {
      Visit(range->upper_bound);
    } else {
      str_ += "undefined";
    }
    str_ += ")";
  }
  str_ += "]";
}
void IrPythonPrinter::Visit(const Reduce *f) {
  str_ += "Reduce(";
  switch (f->reduce_type) {
    case Reduce::ReduceType::kSum:
      str_ += "sum";
      break;
    case Reduce::ReduceType::kSub:
      str_ += "sub";
      break;
    case Reduce::ReduceType::kDiv:
      str_ += "Div";
      break;
    case Reduce::ReduceType::kMul:
      str_ += "Mul";
      break;
    case Reduce::ReduceType::kMax:
      str_ += "Max";
      break;
    case Reduce::ReduceType::kMin:
      str_ += "Min";
      break;
    case Reduce::ReduceType::kAll:
      str_ += "&&";
      break;
    case Reduce::ReduceType::kAny:
      str_ += "||";
      break;
  }
  str_ += ", ";
  Visit(f->body);
  str_ += ",";
  Visit(f->init);
  str_ += ")";
}
void IrPythonPrinter::Visit(const Sum *x) {
  str_ += "(";
  for (std::size_t i = 0; i + 1 < x->operands().size(); i++) {
    Visit(x->operand(i));
    str_ += " + ";
  }
  if (!x->operands().empty()) Visit(x->operands().back());
  str_ += ")";
}
void IrPythonPrinter::Visit(const FracOp *x) {
  str_ += "(";
  Visit(x->a());
  str_ += " / ";
  Visit(x->b());
  str_ += ")";
}

void IrPythonPrinter::Visit(const Product *x) {
  str_ += "(";
  for (std::size_t i = 0; i + 1 < x->operands().size(); i++) {
    Visit(x->operand(i));
    str_ += " * ";
  }
  if (!x->operands().empty()) Visit(x->operands().back());
  str_ += ")";
}

// intrinsics op
void IrPythonPrinter::Visit(const intrinsics::BufferGetDataHandle *x) {
  str_ += runtime::intrinsic::buffer_get_data_handle;
  Visit(x->buffer);
  str_ += ")";
}
void IrPythonPrinter::Visit(const intrinsics::BufferGetDataConstHandle *x) {
  str_ += runtime::intrinsic::buffer_get_data_const_handle;
  Visit(x->buffer);
  str_ += ")";
}
void IrPythonPrinter::Visit(const intrinsics::PodValueToX *x) {
  str_ += "pod_value_to_";
  str_ += x->GetOutputType(0).to_string();
  str_ += "(";
  Visit(x->pod_value_ptr);
  str_ += ")";
}
void IrPythonPrinter::Visit(const intrinsics::BufferCreate *x) {
  str_ += runtime::intrinsic::buffer_create;
  str_ += "()";
}
void IrPythonPrinter::Visit(const intrinsics::GetAddr *x) {
  str_ += "get_addr(";
  Visit(x->data);
  str_ += ")";
}
void IrPythonPrinter::Visit(const intrinsics::ArgsConstruct *x) {
  str_ += runtime::intrinsic::args_construct_repr;
  str_ += "(";
  Visit(std::vector<Expr>(x->args.begin(), x->args.end()));
  str_ += ")";
}

void IrPythonPrinter::Visit(const intrinsics::BuiltinIntrin *x) {
  str_ += runtime::intrinsic::builtin_intrin_repr;
  str_ += "_";
  str_ += x->name;
  str_ += "(";
  if (!x->args.empty()) {
    for (std::size_t i = 0; i + 1 < x->args.size(); i++) {
      Visit(x->args[i]);
      str_ += ", ";
    }
    Visit(x->args.back());
  }

  str_ += ")";
}
void IrPythonPrinter::Print(const common::Type &type) {
  str_ += "cinn.common.type_of(\"";
  str_ += common::Type2Str(type);
  str_ += "\")";
  os_ << str_;
  str_ = "";
}
void IrPythonPrinter::PrintShape(const std::vector<Expr> &shape) {
  for (int i = 0; i < shape.size() - 1; i++) {
    IrPythonPrinter::Visit(shape[i]);
    str_ += ", ";
  }
  if (shape.size() > 1) IrPythonPrinter::Visit(shape.back());
}
}  // namespace ir

}  // namespace cinn
