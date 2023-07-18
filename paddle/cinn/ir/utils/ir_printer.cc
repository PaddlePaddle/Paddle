// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/ir/utils/ir_printer.h"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <vector>

#include "paddle/cinn/ir/lowered_func.h"
#include "paddle/cinn/ir/module.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/optim/ir_simplify.h"
#include "paddle/cinn/runtime/intrinsic.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace ir {

using common::bfloat16;
using common::float16;

void IrPrinter::Print(Expr e) { IRVisitorRequireReImpl::Visit(&e); }
void IrPrinter::Print(const std::vector<Expr> &exprs,
                      const std::string &splitter) {
  for (std::size_t i = 0; !exprs.empty() && i + 1 < exprs.size(); i++) {
    Print(exprs[i]);
    os_ << splitter;
  }
  if (!exprs.empty()) Print(exprs.back());
}

void IrPrinter::Visit(const IntImm *x) {
  if (x->type().is_int(64)) {
    os_ << x->value << "ll";
  } else if (x->type().is_int(32)) {
    os_ << x->value;
  } else if (x->type().is_int(16)) {
    os_ << "(int16_t)" << x->value;
  } else if (x->type().is_int(8)) {
    os_ << "(int8_t)" << x->value;
  } else {
    LOG(FATAL) << "Not support int type: " << x->type();
  }
}
void IrPrinter::Visit(const UIntImm *x) {
  if (x->type().is_uint(64)) {
    os_ << x->value << "ull";
  } else if (x->type().is_uint(32)) {
    os_ << x->value;
  } else if (x->type().is_uint(16)) {
    os_ << "(uint16_t)" << x->value;
  } else if (x->type().is_uint(8)) {
    os_ << "(uint8_t)" << x->value;
  } else if (x->type().is_uint(1)) {
    if (x->value) {
      os_ << "true";
    } else {
      os_ << "false";
    }
  } else {
    LOG(FATAL) << "Not support uint type: " << x->type();
  }
}
void IrPrinter::Visit(const FloatImm *x) {
  if (x->type().is_float16()) {
    if (std::isinf(x->value)) {
      os_ << "cinn::common::raw_uint16_to_float16(0x7c00)";
    } else if (std::isnan(x->value)) {
      os_ << "cinn::common::raw_uint16_to_float16(0x7e00)";
    } else {
      os_ << "(float16)"
          << std::setprecision(std::numeric_limits<float16>::max_digits10)
          << static_cast<float16>(x->value) << "f";
    }
  } else if (x->type().is_bfloat16()) {
    if (std::isinf(x->value)) {
      os_ << "cinn::common::raw_uint16_to_bfloat16(0x7F80)";
    } else if (std::isnan(x->value)) {
      os_ << "cinn::common::raw_uint16_to_bfloat16(0x7FC0)";
    } else {
      os_ << "(bfloat16)"
          << std::setprecision(std::numeric_limits<bfloat16>::max_digits10)
          << static_cast<bfloat16>(x->value) << "f";
    }
  } else if (x->type().is_float(32)) {
    os_ << std::setprecision(std::numeric_limits<float>::max_digits10)
        << std::showpoint << x->value;
    if (std::isfinite(x->value)) {
      os_ << "f";
    }
  } else if (x->type().is_float(64)) {
    os_ << std::setprecision(std::numeric_limits<double>::max_digits10)
        << std::showpoint << x->value;
  } else {
    LOG(FATAL) << "Not support float type: " << x->type();
  }
}
void IrPrinter::Visit(const StringImm *x) { os_ << "\"" << x->value << "\""; }
void IrPrinter::Visit(const Add *x) { PrintBinaryOp("+", x); }
void IrPrinter::Visit(const Sub *x) { PrintBinaryOp("-", x); }
void IrPrinter::Visit(const Mul *x) { PrintBinaryOp("*", x); }
void IrPrinter::Visit(const Div *x) { PrintBinaryOp("/", x); }
void IrPrinter::Visit(const Mod *x) { PrintBinaryOp("%", x); }
void IrPrinter::Visit(const EQ *x) { PrintBinaryOp("==", x); }
void IrPrinter::Visit(const NE *x) { PrintBinaryOp("!=", x); }
void IrPrinter::Visit(const LT *x) { PrintBinaryOp("<", x); }
void IrPrinter::Visit(const LE *x) { PrintBinaryOp("<=", x); }
void IrPrinter::Visit(const GT *x) { PrintBinaryOp(">", x); }
void IrPrinter::Visit(const GE *x) { PrintBinaryOp(">=", x); }
void IrPrinter::Visit(const And *x) { PrintBinaryOp("and", x); }
void IrPrinter::Visit(const Or *x) { PrintBinaryOp("or", x); }
void IrPrinter::Visit(const Not *x) {
  os_ << "!";
  Print(x->v());
}
void IrPrinter::Visit(const Min *x) {
  os_ << "cinn_min(";
  Print(x->a());
  os_ << ", ";
  Print(x->b());
  os_ << ")";
}
void IrPrinter::Visit(const Max *x) {
  os_ << "cinn_max(";
  Print(x->a());
  os_ << ", ";
  Print(x->b());
  os_ << ")";
}
void IrPrinter::Visit(const Minus *x) {
  os_ << "-(";
  Print(x->v());
  os_ << ")";
}
void IrPrinter::Visit(const For *x) {
  if (x->is_parallel()) {
    os() << "parallel for (";
  } else if (x->is_unrolled()) {
    os() << "unroll for (";
  } else if (x->is_vectorized()) {
    int factor = x->vectorize_info().factor;
    os() << "vectorize[" << factor << "] for (";
  } else if (x->is_binded()) {
    auto &bind_info = x->bind_info();
    if (bind_info.valid()) {
      char axis_name = 'x' + bind_info.offset;
      auto for_type = bind_info.for_type;
      std::string prefix =
          for_type == ForType::GPUBlock ? "blockIdx." : "threadIdx.";
      os() << "thread_bind[" << prefix << axis_name << "] for (";
    } else {
      os() << "thread_bind[invalid info] for (";
    }
  } else if (x->is_serial()) {
    os() << "serial for (";
  } else if (x->is_default()) {
    os() << "default for (";
  } else {
    os() << "for (";
  }
  Print(x->loop_var);
  os_ << ", ";
  Print(x->min);
  os_ << ", ";
  Print(x->extent);
  os_ << ")\n";

  DoIndent();
  Print(x->body);
}

void IrPrinter::Visit(const PolyFor *x) {
  if (x->is_parallel()) {
    os() << "parallel poly_for (";
  } else {
    os() << "poly_for (";
  }
  Print(x->iterator);
  os_ << ", ";
  Print(x->init);
  os_ << ", ";
  Print(x->condition);
  os_ << ", ";
  Print(x->inc);
  os_ << ")\n";

  DoIndent();
  Print(x->body);
}
void IrPrinter::Visit(const IfThenElse *x) {
  os_ << "if (";
  Print(x->condition);
  os_ << ") {\n";
  IncIndent();
  DoIndent();
  Print(x->true_case);
  DecIndent();
  os() << "\n";
  DoIndent();
  os() << "}";

  if (x->false_case.defined()) {
    os_ << " else {\n";
    IncIndent();

    DoIndent();
    Print(x->false_case);
    os() << "\n";

    DecIndent();
    DoIndent();
    os_ << "}";
  }
}
void IrPrinter::Visit(const Block *x) {
  os_ << "{\n";

  IncIndent();
  for (std::size_t i = 0; !x->stmts.empty() && i + 1 < x->stmts.size(); i++) {
    DoIndent();
    Print(x->stmts[i]);
    os_ << "\n";
  }
  if (!x->stmts.empty()) {
    DoIndent();
    Print(x->stmts.back());
  }
  DecIndent();
  os_ << "\n";
  DoIndent();
  os_ << "}";
}
void IrPrinter::Visit(const Call *x) {
  os_ << x->name << "(";
  if (!x->read_args.empty()) {
    for (std::size_t i = 0; i + 1 < x->read_args.size(); i++) {
      Print(x->read_args[i]);
      os_ << ", ";
    }
    Print(x->read_args.back());
  }

  if (!x->write_args.empty()) {
    if (!x->read_args.empty()) os() << ", ";

    for (std::size_t i = 0; i + 1 < x->write_args.size(); i++) {
      Print(x->write_args[i]);
      os_ << ", ";
    }
    Print(x->write_args.back());
  }

  os_ << ")";
}
void IrPrinter::Visit(const Cast *x) {
  os() << x->type();
  os() << "(";
  os() << x->v();
  os() << ")";
}
void IrPrinter::Visit(const _Module_ *x) {}
void IrPrinter::Visit(const _Var_ *x) { os_ << x->name; }
void IrPrinter::Visit(const Alloc *x) {
  auto *buffer = x->destination.As<ir::_Buffer_>();
  CHECK(buffer);
  os_ << "alloc(" << buffer->name << ", ";
  Print(x->extents);
  os_ << ")";
}
void IrPrinter::Visit(const Select *x) {
  os_ << "select(";
  Print(x->condition);
  os_ << ", ";
  Print(x->true_value);
  os_ << ", ";
  Print(x->false_value);
  os_ << ")";
}
void IrPrinter::Visit(const Load *x) {
  if (x->is_addr_tensor()) {
    auto *tensor = x->tensor.As<ir::_Tensor_>();
    CHECK(tensor);
    os_ << tensor->name;
  } else if (x->is_addr_scalar()) {
    Print(x->tensor);
  } else {
    CINN_NOT_IMPLEMENTED
  }

  os_ << "[";
  for (std::size_t i = 0; i + 1 < x->indices.size(); i++) {
    Print(x->indices[i]);
    os() << ", ";
  }
  if (!x->indices.empty()) Print(x->indices.back());
  os_ << "]";
}
void IrPrinter::Visit(const Store *x) {
  if (x->is_addr_tensor()) {
    auto *tensor_node = x->tensor.As<ir::_Tensor_>();
    CHECK(tensor_node);
    os_ << tensor_node->name;
  } else if (x->is_addr_scalar()) {
    Print(x->tensor);
  } else {
    CINN_NOT_IMPLEMENTED
  }

  os_ << "[";
  for (std::size_t i = 0; i + 1 < x->indices.size(); i++) {
    Print(x->indices[i]);
    os() << ", ";
  }
  if (!x->indices.empty()) Print(x->indices.back());
  os_ << "] = ";
  Print(x->value);
}
void IrPrinter::Visit(const Free *x) {
  auto *buffer = x->destination.As<ir::_Buffer_>();
  CHECK(buffer);
  os_ << "free(" << buffer->name << ")";
}

void IrPrinter::DoIndent() { os_ << std::string(indent_, ' '); }
void IrPrinter::IncIndent() { indent_ += indent_unit; }
void IrPrinter::DecIndent() { indent_ -= indent_unit; }

void IrPrinter::Visit(const _Buffer_ *x) {
  std::vector<std::string> dim_names;
  std::transform(x->shape.begin(),
                 x->shape.end(),
                 std::back_inserter(dim_names),
                 [&](const Expr &x) { return utils::GetStreamCnt(x); });

  os_ << "_Buffer_<" << x->type() << ": " << utils::Join(dim_names, ",") << ">("
      << x->name << ")";
}
void IrPrinter::Visit(const _Tensor_ *x) {
  os_ << "Tensor(";
  os() << x->name << ", ";
  os() << "[";
  if (!x->shape.empty()) {
    for (std::size_t i = 0; i + 1 < x->shape.size(); i++) {
      Print(x->shape[i]);
      os() << ",";
    }
    Print(x->shape.back());
  }
  os_ << "])";
}
void IrPrinter::Visit(const _LoweredFunc_ *f) {
  os_ << "function " << f->name << " ";

  std::vector<std::string> arg_names;
  for (auto &arg : f->args) {
    arg_names.push_back(arg.name());
  }
  os_ << "(" << utils::Join(arg_names, ", ") << ")\n";

  Print(f->body);
}
void IrPrinter::Visit(const Let *f) {
  CHECK(f->type().valid());
  os() << f->type() << " ";
  Print(f->symbol);
  if (f->body.defined()) {
    os() << " = ";
    Print(f->body);
  }
}

void IrPrinter::Visit(const Reduce *f) {
  os() << "Reduce(";
  switch (f->reduce_type) {
    case Reduce::ReduceType::kSum:
      os() << "sum";
      break;
    case Reduce::ReduceType::kSub:
      os() << "sub";
      break;
    case Reduce::ReduceType::kDiv:
      os() << "Div";
      break;
    case Reduce::ReduceType::kMul:
      os() << "Mul";
      break;
    case Reduce::ReduceType::kMax:
      os() << "Max";
      break;
    case Reduce::ReduceType::kMin:
      os() << "Min";
      break;
    case Reduce::ReduceType::kAll:
      os() << "&&";
      break;
    case Reduce::ReduceType::kAny:
      os() << "||";
      break;
  }
  os() << ", ";
  Print(f->body);
  os() << ",";
  Print(f->init);
  os() << ")";
}

void IrPrinter::Visit(const Ramp *x) {
  os() << "Ramp(";
  Print(x->base);
  os() << ",";
  Print(x->stride);
  os() << ",";
  os() << x->lanes;
  os() << ")";
}

void IrPrinter::Visit(const Broadcast *x) {
  os() << "Broadcast(";
  Print(x->value);
  os() << ",";
  os() << x->lanes;
  os() << ")";
}

void IrPrinter::Visit(const FracOp *x) {
  os() << "(";
  Print(x->a());
  os() << " / ";
  Print(x->b());
  os() << ")";
}

void IrPrinter::Visit(const Product *x) {
  os() << "(";
  for (std::size_t i = 0; i + 1 < x->operands().size(); i++) {
    Print(x->operand(i));
    os() << " * ";
  }
  if (!x->operands().empty()) Print(x->operands().back());
  os() << ")";
}

void IrPrinter::Visit(const Sum *x) {
  os() << "(";
  for (std::size_t i = 0; i + 1 < x->operands().size(); i++) {
    Print(x->operand(i));
    os() << " + ";
  }
  if (!x->operands().empty()) Print(x->operands().back());
  os() << ")";
}

void IrPrinter::Visit(const PrimitiveNode *x) {
  os() << x->name << "(";
  std::vector<std::string> args_repr;
  for (auto &args : x->arguments) {
    std::vector<std::string> arg_repr;
    for (auto &arg : args) {
      arg_repr.push_back(utils::GetStreamCnt(arg));
    }
    args_repr.push_back(utils::Join(arg_repr, ","));
  }

  os() << utils::Join(args_repr, ",");
  os() << ")";
}

void IrPrinter::Visit(const _BufferRange_ *x) {
  auto *buffer = x->buffer.As<ir::_Buffer_>();
  CHECK(buffer);
  os() << buffer->name << "[";
  for (std::size_t i = 0; i < x->ranges.size(); i++) {
    if (i) os() << ", ";
    auto &range = x->ranges[i];
    os() << range->name << "(";
    if (range->lower_bound.defined()) {
      os() << range->lower_bound << ":";
    } else {
      os() << "undefined:";
    }

    if (range->upper_bound.defined()) {
      os() << range->upper_bound;
    } else {
      os() << "undefined";
    }
    os() << ")";
  }
  os() << "]";
}

void IrPrinter::Visit(const ScheduleBlock *x) {}

void IrPrinter::Visit(const ScheduleBlockRealize *x) {
  auto *schedule_block = x->schedule_block.As<ScheduleBlock>();
  os() << "ScheduleBlock(" << schedule_block->name << ")\n";
  DoIndent();
  os() << "{\n";
  // print block vars and bindings
  auto iter_vars = schedule_block->iter_vars;
  auto iter_values = x->iter_values;
  CHECK_EQ(iter_vars.size(), iter_values.size());
  IncIndent();
  if (!iter_vars.empty()) DoIndent();
  for (std::size_t i = 0; i < iter_vars.size(); i++) {
    if (i) os() << ", ";
    os() << iter_vars[i]->name;
  }
  if (!iter_vars.empty()) os() << " = axis.bind(";
  for (std::size_t i = 0; i < iter_values.size(); i++) {
    if (i) os() << ", ";
    os() << iter_values[i];
  }
  if (!iter_vars.empty()) os() << ")\n";
  // print block body
  if (!schedule_block->read_buffers.empty()) {
    DoIndent();
    os() << "read_buffers(";
    auto &read_buffers = schedule_block->read_buffers;
    for (std::size_t i = 0; i < read_buffers.size(); i++) {
      if (i) os() << ", ";
      Print(read_buffers[i]);
    }
    os() << ")\n";
  }
  if (!schedule_block->write_buffers.empty()) {
    DoIndent();
    os() << "write_buffers(";
    auto &write_buffers = schedule_block->write_buffers;
    for (std::size_t i = 0; i < write_buffers.size(); i++) {
      if (i) os() << ", ";
      Print(write_buffers[i]);
    }
    os() << ")\n";
  }
  if (!schedule_block->attrs.empty()) {
    DoIndent();
    os() << "attrs(";
    bool comma = false;
    for (auto &&kv : schedule_block->attrs) {
      if (comma) os() << ", ";
      os() << kv.first << ":";
      absl::visit([this](auto &&arg) { this->os() << arg; }, kv.second);
      comma = true;
    }
    os() << ")\n";
  }
  DoIndent();
  Print(schedule_block->body);
  os() << "\n";
  DecIndent();
  DoIndent();
  os() << "}";
}

void IrPrinter::Visit(const IntrinsicOp *x) {
  switch (x->getKind()) {
#define __(op__)                                \
  case IntrinsicKind::k##op__:                  \
    Visit(llvm::dyn_cast<intrinsics::op__>(x)); \
    break;

    INTRINSIC_KIND_FOR_EACH(__)
#undef __
  }
}
void IrPrinter::Visit(const intrinsics::BufferGetDataHandle *x) {
  os() << runtime::intrinsic::buffer_get_data_handle;
  Print(x->buffer);
  os() << ")";
}
void IrPrinter::Visit(const intrinsics::BufferGetDataConstHandle *x) {
  os() << runtime::intrinsic::buffer_get_data_const_handle;
  Print(x->buffer);
  os() << ")";
}
void IrPrinter::Visit(const intrinsics::PodValueToX *x) {
  os() << "pod_value_to_";
  os() << x->GetOutputType(0);
  os() << "(";
  Print(x->pod_value_ptr);
  os() << ")";
}
void IrPrinter::Visit(const intrinsics::BufferCreate *x) {
  os() << runtime::intrinsic::buffer_create;
  os() << "()";
}
void IrPrinter::Visit(const intrinsics::GetAddr *x) {
  os() << "get_addr(";
  Print(x->data);
  os() << ")";
}
void IrPrinter::Visit(const intrinsics::ArgsConstruct *x) {
  os() << runtime::intrinsic::args_construct_repr;
  os() << "(";
  Print(std::vector<Expr>(x->args.begin(), x->args.end()));
  os() << ")";
}

void IrPrinter::Visit(const intrinsics::BuiltinIntrin *x) {
  os_ << runtime::intrinsic::builtin_intrin_repr << "_";
  os_ << x->name << "(";
  if (!x->args.empty()) {
    for (std::size_t i = 0; i + 1 < x->args.size(); i++) {
      Print(x->args[i]);
      os_ << ", ";
    }
    Print(x->args.back());
  }

  os_ << ")";
}

std::ostream &operator<<(std::ostream &os, Expr a) {
  std::stringstream ss;
  IrPrinter printer(ss);
  printer.Print(a);
  os << ss.str();
  return os;
}

std::ostream &operator<<(std::ostream &os, const std::vector<Expr> &a) {
  std::stringstream ss;
  IrPrinter printer(ss);
  printer.Print(a);
  os << ss.str();
  return os;
}

std::ostream &operator<<(std::ostream &os, const ir::Module &m) {
  os << "Module " << m->name << " {\n\n";
  for (auto &fn : m->functions) {
    os << fn << '\n';
  }
  os << "\n\n}";
  return os;
}

}  // namespace ir
}  // namespace cinn
