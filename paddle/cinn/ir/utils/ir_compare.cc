// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/ir/utils/ir_compare.h"

#include <regex>

#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/module.h"

namespace cinn {
namespace ir {

namespace ir_utils {
bool IrEqualVisitor::Compare(const Expr& lhs, const Expr& rhs) {
  if (lhs.get() == rhs.get()) {  // the same object, including both are null
    return true;
  }

  if (only_compare_structure_ && !lhs.defined() && !rhs.defined()) {
    return true;
  }

  if (!lhs.defined() || !rhs.defined()) {  // someone invalid
    return false;
    VLOG(7) << "Not equal on Expr, someone not defined";
  }
  bool equal = lhs->node_type() == rhs->node_type();
  if (lhs.is_index() && rhs.is_index())
    equal = equal && lhs.as_index() == rhs.as_index();
  else
    equal =
        equal && IRVisitorRequireReImpl<bool, const Expr*>::Visit(&lhs, &rhs);
  if (!equal) {
    VLOG(7) << "Not equal on Expr, lhs:[type:"
            << kIrNodeTyReprs[static_cast<int>(lhs->node_type())] << "]\n"
            << lhs << ", \nrhs[type:"
            << kIrNodeTyReprs[static_cast<int>(rhs->node_type())] << "]\n"
            << rhs;
  }
  return equal;
}

bool IrEqualVisitor::Compare(const Module& lhs, const Module& rhs) {
  return Visit(lhs.As<_Module_>(), rhs.As<_Module_>());
}

bool IrEqualVisitor::Compare(const LoweredFunc& lhs, const LoweredFunc& rhs) {
  return Visit(lhs.As<_LoweredFunc_>(), rhs.As<_LoweredFunc_>());
}

bool IrEqualVisitor::Compare(const std::string& lhs, const std::string& rhs) {
  // if allow_name_suffix_diff_=true then just compare the name prefix before
  // the
  // "_[0-9]+"
  auto common_len = 0;
  for (; common_len < lhs.size() && common_len < rhs.size(); ++common_len) {
    if (lhs[common_len] != rhs[common_len]) break;
  }

  auto is_endswith_index = [&common_len](const std::string& name) {
    const std::regex txt_regex("_\\d+");
    return common_len == name.size() ||
           std::regex_match(name.substr(common_len), txt_regex);
  };

  bool equal = false;
  if (common_len == lhs.size() && common_len == rhs.size()) {
    equal = true;
  } else {
    equal = false;
    if (allow_name_suffix_diff_) {
      equal = is_endswith_index(lhs) && is_endswith_index(rhs);
    }
  }

  if (!equal) {
    VLOG(5) << "Not equal on name, lhs=" << lhs << ", rhs=" << rhs;
  }

  return equal;
}

bool IrEqualVisitor::Compare(const std::map<std::string, attr_t>& lhs,
                             const std::map<std::string, attr_t>& rhs) {
  if (lhs.size() != rhs.size()) {
    VLOG(6) << "Not equal on attrs, lhs size=" << lhs.size()
            << ", rhs size=" << rhs.size();
    return false;
  }
  for (auto&& kv : lhs) {
    auto opposite = rhs.find(kv.first);
    if (opposite == rhs.end() || kv.second != opposite->second) {
      VLOG(6) << "Not equal at attr key=" << kv.first;
      return false;
    }
  }
  return true;
}

template <typename T>
bool IrEqualVisitor::Compare(const std::vector<T>& lhs,
                             const std::vector<T>& rhs) {
  if (lhs.size() != rhs.size()) {
    VLOG(6) << "Not equal on repeated fields, lhs size=" << lhs.size()
            << ", rhs size=" << rhs.size();
    return false;
  }
  for (auto i = 0; i < lhs.size(); ++i) {
    if (!Compare(lhs.at(i), rhs.at(i))) {
      VLOG(6) << "Not equal on repeated fields at index=" << i;
      return false;
    }
  }
  return true;
}

#define PRIMITIVE_TYPE_IMPL(op__)                                  \
  bool IrEqualVisitor::Visit(const op__* lhs, const Expr* other) { \
    auto* rhs = other->As<op__>();                                 \
    return lhs->value == rhs->value;                               \
  }

#define UNARY_OP_IMPL(op__)                                        \
  bool IrEqualVisitor::Visit(const op__* lhs, const Expr* other) { \
    auto* rhs = other->As<op__>();                                 \
    return Compare(lhs->v(), rhs->v());                            \
  }

#define BINARY_OP_IMPL(op__)                                           \
  bool IrEqualVisitor::Visit(const op__* lhs, const Expr* other) {     \
    auto* rhs = other->As<op__>();                                     \
    return Compare(lhs->a(), rhs->a()) && Compare(lhs->b(), rhs->b()); \
  }

NODETY_PRIMITIVE_TYPE_FOR_EACH(PRIMITIVE_TYPE_IMPL)
NODETY_UNARY_OP_FOR_EACH(UNARY_OP_IMPL)
NODETY_BINARY_OP_FOR_EACH(BINARY_OP_IMPL)

#undef PRIMITIVE_TYPE_IMPL
#undef UNARY_OP_IMPL
#undef BINARY_OP_IMPL

bool IrEqualVisitor::Visit(const Cast* lhs, const Expr* other) {
  auto* rhs = other->As<Cast>();
  return lhs->type() == rhs->type() && Compare(lhs->v(), rhs->v());
}

bool IrEqualVisitor::Visit(const For* lhs, const Expr* other) {
  auto* rhs = other->As<For>();
  return lhs->for_type() == rhs->for_type() &&
         Compare(lhs->loop_var, rhs->loop_var) && Compare(lhs->min, rhs->min) &&
         Compare(lhs->extent, rhs->extent) && Compare(lhs->body, rhs->body);
}

bool IrEqualVisitor::Visit(const PolyFor* lhs, const Expr* other) {
  auto* rhs = other->As<PolyFor>();
  return lhs->for_type() == rhs->for_type() &&
         Compare(lhs->iterator, rhs->iterator) &&
         Compare(lhs->init, rhs->init) &&
         Compare(lhs->condition, rhs->condition) &&
         Compare(lhs->inc, rhs->inc) && Compare(lhs->body, rhs->body);
}

bool IrEqualVisitor::Visit(const Select* lhs, const Expr* other) {
  auto* rhs = other->As<Select>();
  return Compare(lhs->condition, rhs->condition) &&
         Compare(lhs->true_value, rhs->true_value) &&
         Compare(lhs->false_value, rhs->false_value);
}

bool IrEqualVisitor::Visit(const IfThenElse* lhs, const Expr* other) {
  auto* rhs = other->As<IfThenElse>();
  return Compare(lhs->condition, rhs->condition) &&
         Compare(lhs->true_case, rhs->true_case) &&
         Compare(lhs->false_case, rhs->false_case);
}

bool IrEqualVisitor::Visit(const Block* lhs, const Expr* other) {
  auto* rhs = other->As<Block>();
  return Compare(lhs->stmts, rhs->stmts);
}

bool IrEqualVisitor::Visit(const Call* lhs, const Expr* other) {
  auto* rhs = other->As<Call>();
  bool flag = Compare(lhs->read_args, rhs->read_args) &&
              Compare(lhs->write_args, rhs->write_args) &&
              Compare(lhs->attrs, rhs->attrs) &&
              lhs->call_type == rhs->call_type;
  if (only_compare_structure_) {
    return flag;
  }
  return lhs->name == rhs->name && flag;
  // TODO(CtfGo): Compare `func` field
}

bool IrEqualVisitor::Visit(const _Var_* lhs, const Expr* other) {
  auto* rhs = other->As<_Var_>();
  bool flag = Compare(lhs->lower_bound, rhs->lower_bound) &&
              Compare(lhs->upper_bound, rhs->upper_bound) &&
              lhs->tag == rhs->tag;
  if (only_compare_structure_) {
    return flag;
  }
  return lhs->name == rhs->name && flag;
}

bool IrEqualVisitor::Visit(const Load* lhs, const Expr* other) {
  auto* rhs = other->As<Load>();
  return Compare(lhs->tensor, rhs->tensor) &&
         Compare(lhs->indices, rhs->indices);
}

bool IrEqualVisitor::Visit(const Store* lhs, const Expr* other) {
  auto* rhs = other->As<Store>();
  return Compare(lhs->tensor, rhs->tensor) &&
         Compare(lhs->indices, rhs->indices);
}

bool IrEqualVisitor::Visit(const Alloc* lhs, const Expr* other) {
  auto* rhs = other->As<Alloc>();
  return Compare(lhs->destination, rhs->destination) &&
         Compare(lhs->extents, rhs->extents) &&
         Compare(lhs->condition, rhs->condition) &&
         Compare(lhs->body, rhs->body);
}

bool IrEqualVisitor::Visit(const Free* lhs, const Expr* other) {
  auto* rhs = other->As<Free>();
  return Compare(lhs->destination, rhs->destination);
}

bool IrEqualVisitor::Visit(const _Buffer_* lhs, const Expr* other) {
  auto* rhs = other->As<_Buffer_>();
  bool flag =
      Compare(lhs->shape, rhs->shape) && Compare(lhs->strides, rhs->strides) &&
      lhs->scope == rhs->scope && Compare(lhs->elem_offset, rhs->elem_offset) &&
      lhs->offset_factor == rhs->offset_factor && lhs->target == rhs->target &&
      lhs->data_alignment == rhs->data_alignment &&
      lhs->memory_type == rhs->memory_type && lhs->dtype == rhs->dtype;
  if (only_compare_structure_) {
    return flag;
  }
  return flag && lhs->name == rhs->name;
}

bool IrEqualVisitor::Visit(const _Tensor_* lhs, const Expr* other) {
  auto* rhs = other->As<_Tensor_>();
  bool flag = Compare(lhs->shape, rhs->shape);
  if (only_compare_structure_) {
    return flag;
  }
  return flag && Compare(lhs->name, rhs->name);
}

bool IrEqualVisitor::Visit(const _LoweredFunc_* lhs, const _LoweredFunc_* rhs) {
  if (lhs->name != rhs->name) {
    VLOG(6) << "Not equal, lhs name=" << lhs->name
            << ", rhs name=" << rhs->name;
    return false;
  }

  auto compare_args_fn = [this](const std::vector<Argument>& largs,
                                const std::vector<Argument>& rargs) -> bool {
    if (largs.size() != rargs.size()) {
      VLOG(6) << "Not equal, lhs args size=" << largs.size()
              << ", rhs args size=" << rargs.size();
      return false;
    }
    for (auto i = 0; i < largs.size(); ++i) {
      const Argument& a = largs.at(i);
      const Argument& b = rargs.at(i);
      bool equal = a.io == b.io;
      equal = equal &&
              (!a.is_var() && !b.is_var() ||
               a.is_var() && b.is_var() && Compare(a.var_arg(), b.var_arg()));
      equal = equal && (!a.is_buffer() && !b.is_buffer() ||
                        a.is_buffer() && b.is_buffer() &&
                            Compare(a.buffer_arg(), b.buffer_arg()));
      if (!equal) {
        VLOG(6) << "Not equal at Argument index=" << i;
        return false;
      }
    }
    return true;
  };

  return compare_args_fn(lhs->args, rhs->args) &&
         Compare(lhs->temp_bufs, rhs->temp_bufs) &&
         Compare(lhs->body, rhs->body) && lhs->device_api == rhs->device_api &&
         Compare(lhs->alloc_output_buffer_exprs,
                 rhs->alloc_output_buffer_exprs) &&
         Compare(lhs->dealloc_output_buffer_exprs,
                 rhs->dealloc_output_buffer_exprs) &&
         Compare(lhs->buffer_data_cast_exprs, rhs->buffer_data_cast_exprs) &&
         Compare(lhs->argument_prepare_exprs, rhs->argument_prepare_exprs);
}

bool IrEqualVisitor::Visit(const _Module_* lhs, const _Module_* rhs) {
  bool flag = Compare(lhs->buffers, rhs->buffers) &&
              Compare(lhs->functions, rhs->functions) &&
              Compare(lhs->submodules, rhs->submodules);

  if (only_compare_structure_) {
    return flag;
  }

  return flag && lhs->name == rhs->name;
}

bool IrEqualVisitor::Visit(const Let* lhs, const Expr* other) {
  auto* rhs = other->As<Let>();
  return Compare(lhs->symbol, rhs->symbol) && Compare(lhs->body, rhs->body);
}

bool IrEqualVisitor::Visit(const Reduce* lhs, const Expr* other) {
  auto* rhs = other->As<Reduce>();
  return Compare(lhs->init, rhs->init) && Compare(lhs->body, rhs->body) &&
         lhs->reduce_type == rhs->reduce_type;
  // TODO(CtfGo): compare `reduce_axis` field
}

bool IrEqualVisitor::Visit(const Ramp* lhs, const Expr* other) {
  auto* rhs = other->As<Ramp>();
  return Compare(lhs->base, rhs->base) && Compare(lhs->stride, rhs->stride) &&
         lhs->lanes == rhs->lanes;
}

bool IrEqualVisitor::Visit(const Broadcast* lhs, const Expr* other) {
  auto* rhs = other->As<Broadcast>();
  return Compare(lhs->value, rhs->value) && lhs->lanes == rhs->lanes;
}

bool IrEqualVisitor::Visit(const FracOp* lhs, const Expr* other) {
  auto* rhs = other->As<FracOp>();
  return Compare(lhs->a(), rhs->a()) && Compare(lhs->b(), rhs->b());
}

bool IrEqualVisitor::Visit(const Product* lhs, const Expr* other) {
  auto* rhs = other->As<Product>();
  return Compare(lhs->operands(), rhs->operands());
}

bool IrEqualVisitor::Visit(const Sum* lhs, const Expr* other) {
  auto* rhs = other->As<Sum>();
  return Compare(lhs->operands(), rhs->operands());
}

bool IrEqualVisitor::Visit(const PrimitiveNode* lhs, const Expr* other) {
  auto* rhs = other->As<PrimitiveNode>();
  return lhs->name == rhs->name && Compare(lhs->arguments, rhs->arguments) &&
         Compare(lhs->attrs, rhs->attrs);
}

bool IrEqualVisitor::Visit(const IntrinsicOp* lhs, const Expr* other) {
  auto* rhs = other->As<IntrinsicOp>();
  return lhs->getKind() == rhs->getKind() &&
         lhs->input_types() == rhs->input_types() &&
         lhs->output_types() == rhs->output_types();
  // TODO(CtfGo): Compare every derived class of IntrinsicOp separately
}

bool IrEqualVisitor::Visit(const _BufferRange_* lhs, const Expr* other) {
  auto* rhs = other->As<_BufferRange_>();
  return Compare(lhs->buffer, rhs->buffer) && Compare(lhs->ranges, rhs->ranges);
}

bool IrEqualVisitor::Visit(const ScheduleBlock* lhs, const Expr* other) {
  auto* rhs = other->As<ScheduleBlock>();
  bool flag = Compare(lhs->iter_vars, rhs->iter_vars) &&
              Compare(lhs->read_buffers, rhs->read_buffers) &&
              Compare(lhs->write_buffers, rhs->write_buffers) &&
              Compare(lhs->body, rhs->body);

  if (only_compare_structure_) {
    return flag;
  }
  return flag && Compare(lhs->attrs, rhs->attrs) &&
         Compare(lhs->name, rhs->name);
}

bool IrEqualVisitor::Visit(const ScheduleBlockRealize* lhs, const Expr* other) {
  auto* rhs = other->As<ScheduleBlockRealize>();
  return Compare(lhs->iter_values, rhs->iter_values) &&
         Compare(lhs->schedule_block, rhs->schedule_block);
}

bool IrEqualVisitor::Visit(const _Dim_* lhs, const Expr* other) {
  auto* rhs = other->As<_Dim_>();
  return lhs->name == rhs->name && lhs->ToString() == rhs->ToString();
}

bool IrEqualVisitor::Visit(const IterMark* lhs, const Expr* other) {
  auto* rhs = other->As<IterMark>();
  if (!Compare(lhs->source, rhs->source)) return false;
  return lhs->extent == rhs->extent;
}

bool IrEqualVisitor::Visit(const IterSplit* lhs, const Expr* other) {
  auto* rhs = other->As<IterSplit>();
  if (!Compare(lhs->source, rhs->source)) return false;
  return lhs->extent == rhs->extent && lhs->lower_factor == rhs->lower_factor &&
         lhs->scale == rhs->scale;
}

bool IrEqualVisitor::Visit(const IterSum* lhs, const Expr* other) {
  auto* rhs = other->As<IterSum>();
  for (size_t i = 0; i < lhs->args.size(); ++i) {
    if (!Compare(lhs->args.at(i), rhs->args.at(i))) return false;
  }
  return lhs->base == rhs->base;
}

bool IRCompare(const Expr& lhs, const Expr& rhs, bool allow_name_suffix_diff) {
  IrEqualVisitor ir_equal_visitor(allow_name_suffix_diff);
  return ir_equal_visitor.Compare(lhs, rhs);
}

}  // namespace ir_utils
}  // namespace ir
}  // namespace cinn
