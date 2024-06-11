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

#include "paddle/cinn/ir/ir.h"

#include <map>
#include <string>
#include <vector>
#include "paddle/cinn/common/cinn_value.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/ir_utils.h"
#include "paddle/cinn/ir/ir_visitor.h"
#include "paddle/cinn/ir/module.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/optim/ir_simplify.h"
#include "paddle/common/enforce.h"
#include "paddle/common/errors.h"

namespace cinn {
namespace ir {

using cinn::common::make_shared;

Expr Cast::Make(Type t, Expr v) {
  CHECK(!t.is_unk());
  CHECK(!(t.is_void() && !t.is_cpp_handle())) << "Void is not allowed to cast";
  CHECK(v.defined());

  auto node = make_shared<Cast>();
  node->v() = v;
  node->set_type(t);
  return Expr(node);
}

void Cast::Verify() const {
  if (v().type() == type())
    LOG(WARNING) << "Found a Cast Node casting a value to the same type, this "
                    "is not reasonable";
}

Expr Add::Make(Expr a, Expr b) {
  auto node = make_shared<Add>(a, b);
  return Expr(node);
}

Add::Add(Expr a, Expr b) : BinaryOpNode<Add>(a.type(), a, b) {}

void BinaryNodeVerify(const Expr &a, const Expr &b, absl::string_view ir_name) {
  CHECK(a.defined());
  CHECK(b.defined());
  TryElevateInt32ToInt64({a, b});
  PADDLE_ENFORCE_EQ(
      a.type(),
      b.type(),
      phi::errors::InvalidArgument(
          "The operands' types of the node [%s] don't match", ir_name));
}

void Add::Verify() const { BinaryNodeVerify(a(), b(), "Add"); }

Expr Sub::Make(Expr a, Expr b) {
  auto node = make_shared<Sub>(a, b);
  return Expr(node);
}

void Sub::Verify() const { BinaryNodeVerify(a(), b(), "Sub"); }

Expr Mul::Make(Expr a, Expr b) {
  BinaryNodeVerify(a, b, "Mul");
  auto node = make_shared<Mul>(a, b);
  return Expr(node);
}

void Max::Verify() const { BinaryNodeVerify(a(), b(), "Max"); }

Expr Div::Make(Expr a, Expr b) {
  auto node = make_shared<Div>(a, b);
  return Expr(node);
}

void Div::Verify() const { BinaryNodeVerify(a(), b(), "Div"); }

Expr Mod::Make(Expr a, Expr b) {
  auto node = make_shared<Mod>(a, b);
  return Expr(node);
}

void Mod::Verify() const { BinaryNodeVerify(a(), b(), "Mod"); }

Expr Min::Make(Expr a, Expr b) {
  auto node = make_shared<Min>(a, b);
  return Expr(node);
}

void Min::Verify() const { BinaryNodeVerify(a(), b(), "Min"); }

Expr Max::Make(Expr a, Expr b) {
  auto node = make_shared<Max>(a, b);
  return Expr(node);
}

Expr Minus::Make(Expr a) {
  auto node = make_shared<Minus>(a);
  return Expr(node);
}

void Minus::Verify() const { CHECK(v().defined()); }

Expr EQ::Make(Expr a, Expr b) {
  auto node = make_shared<EQ>(a, b);
  return Expr(node);
}

void EQ::Verify() const { BinaryNodeVerify(a(), b(), "EQ"); }

Expr NE::Make(Expr a, Expr b) {
  auto node = make_shared<NE>(a, b);
  return Expr(node);
}

void NE::Verify() const { BinaryNodeVerify(a(), b(), "NE"); }

Expr LT::Make(Expr a, Expr b) {
  auto node = make_shared<LT>(a, b);
  return Expr(node);
}

void LT::Verify() const { BinaryNodeVerify(a(), b(), "LT"); }

Expr LE::Make(Expr a, Expr b) {
  auto node = make_shared<LE>(a, b);
  return Expr(node);
}

void LE::Verify() const { BinaryNodeVerify(a(), b(), "LE"); }

Expr GT::Make(Expr a, Expr b) {
  auto node = make_shared<GT>(a, b);
  return Expr(node);
}

void GT::Verify() const { BinaryNodeVerify(a(), b(), "GT"); }

Expr GE::Make(Expr a, Expr b) {
  auto node = make_shared<GE>(a, b);
  return Expr(node);
}

void GE::Verify() const { BinaryNodeVerify(a(), b(), "GE"); }

Expr And::Make(Expr a, Expr b) {
  auto node = make_shared<And>(a, b);
  return Expr(node);
}

void And::Verify() const {
  BinaryNodeVerify(a(), b(), "And");
  PADDLE_ENFORCE_EQ(
      a().type(),
      type_of<bool>(),
      phi::errors::InvalidArgument(
          "The type of the operands of the node [And] should be bool"));
}

Expr Or::Make(Expr a, Expr b) {
  auto node = make_shared<Or>(a, b);
  return Expr(node);
}

void Or::Verify() const {
  BinaryNodeVerify(a(), b(), "Or");
  PADDLE_ENFORCE_EQ(
      a().type(),
      type_of<bool>(),
      phi::errors::InvalidArgument(
          "The type of the operands of the node [Or] should be bool"));
}

Type Or::type() const { return type_; }

Expr Not::Make(Expr v) {
  auto node = make_shared<Not>(v);
  return Expr(node);
}

void Not::Verify() const {
  PADDLE_ENFORCE_EQ(
      v().type(),
      type_of<bool>(),
      phi::errors::InvalidArgument(
          "The type of the operand of the node [Not] should be bool"));
}

Type Not::type() const { return type_; }

Expr Let::Make(Expr symbol, Expr body) {
  auto *n = make_shared<Let>();
  CHECK(symbol.type().valid());
  if (body.defined()) {
    CHECK(body.type().valid());
  }
  n->symbol = symbol;
  n->body = body;
  n->set_type(n->symbol->type());
  return Expr(n);
}

void Let::Verify() const {
  CHECK(symbol.defined());
  // The default value(contained in body) is not required.
  if (body.defined()) {
    TryElevateInt32ToInt64({symbol, body});
    PADDLE_ENFORCE_EQ(
        symbol.type(),
        body.type(),
        phi::errors::InvalidArgument("The type of the symbol and the body of "
                                     "the node [Let] should be the same"));
  }
}

Type Let::type() const { return symbol.type(); }

Expr _Var_::Make(const std::string &name, const Type &type) {
  auto node = new _Var_(name, type);
  return Expr(node);
}

Expr _Var_::Make(Expr lower_bound,
                 Expr upper_bound,
                 const std::string &name,
                 bool is_reduce_axis,
                 bool is_symbolic_constant,
                 bool is_keepdim) {
  auto *n = make_shared<_Var_>();
  n->lower_bound = lower_bound;
  n->upper_bound = upper_bound;
  n->is_reduce_axis = is_reduce_axis;
  n->is_keepdim = is_keepdim;
  n->is_symbolic_constant = is_symbolic_constant;
  n->name = name;
  n->set_type(lower_bound.type());
  return Expr(n);
}

Expr _Var_::Copy() const {
  auto *n = make_shared<_Var_>();
  n->name = name;
  n->is_reduce_axis = is_reduce_axis;
  n->is_keepdim = is_keepdim;
  n->lower_bound = lower_bound;
  n->upper_bound = upper_bound;
  n->set_type(type());
  return Expr(n);
}

void _Var_::Verify() const { CHECK(!name.empty()) << "Var should have a name"; }

void Mul::Verify() const { BinaryNodeVerify(a(), b(), "Mul"); }

Expr For::Make(Var loop_var,
               Expr min,
               Expr extent,
               ForType for_type,
               DeviceAPI device_api,
               Expr body,
               VectorizeInfo vector_info,
               BindInfo bind_info) {
  ir::TryElevateInt32ToInt64({loop_var, min, extent});
  auto node = make_shared<For>();
  CHECK(loop_var.defined());
  CHECK(min.defined());
  CHECK(extent.defined());
  node->loop_var = loop_var;
  node->min = min;
  node->extent = extent;
  node->device_api = device_api;
  node->body = body.As<ir::Block>() ? body : ir::Block::Make({body});
  node->set_for_type(for_type);
  node->set_vectorize_info(vector_info);
  node->set_bind_info(bind_info);

  if (node->is_vectorized()) CHECK(node->vectorize_info().valid());
  if (node->is_binded() && bind_info.offset >= 0)
    CHECK(node->bind_info().valid());

  return Expr(node);
}

std::vector<Expr *> For::expr_fields() { return {&min, &extent, &body}; }
std::vector<const Expr *> For::expr_fields() const {
  return {&min, &extent, &body};
}

Expr Block::Make(const std::vector<Expr> &stmts) {
  auto node = make_shared<Block>();
  node->stmts = stmts;
  return Expr(node);
}
std::vector<Expr *> Block::expr_fields() {
  std::vector<Expr *> res;
  for (auto &x : stmts) res.push_back(&x);
  return res;
}
std::vector<const Expr *> Block::expr_fields() const {
  std::vector<const Expr *> res;
  for (auto &x : stmts) res.push_back(&x);
  return res;
}

Expr ScheduleBlock::Make(const std::vector<Var> &iter_vars,
                         const std::vector<Expr> &read_buffers,
                         const std::vector<Expr> &write_buffers,
                         const std::string &name,
                         Expr body) {
  auto node = make_shared<ScheduleBlock>();
  node->iter_vars = iter_vars;
  node->read_buffers = read_buffers;
  node->write_buffers = write_buffers;
  node->name = name;
  node->body = body;
  return Expr(node);
}
void ScheduleBlock::Verify() const {
  CHECK(!name.empty());
  CHECK(body.defined());
}
std::vector<Expr *> ScheduleBlock::expr_fields() {
  std::vector<Expr *> res;
  res.push_back(&body);
  return res;
}
std::vector<const Expr *> ScheduleBlock::expr_fields() const {
  std::vector<const Expr *> res;
  res.push_back(&body);
  return res;
}

Expr ScheduleBlockRealize::Make(const std::vector<Expr> &iter_values,
                                const Expr &schedule_block) {
  auto node = make_shared<ScheduleBlockRealize>();
  node->iter_values = iter_values;
  node->schedule_block = schedule_block;
  return Expr(node);
}
void ScheduleBlockRealize::Verify() const {
  auto *schedule_block_ptr = schedule_block.As<ScheduleBlock>();
  CHECK(schedule_block_ptr);
  PADDLE_ENFORCE_EQ(
      schedule_block_ptr->iter_vars.size(),
      iter_values.size(),
      phi::errors::InvalidArgument(
          "The size of iter_values should be equal to the size of iter_vars"));
}
std::vector<Expr *> ScheduleBlockRealize::expr_fields() {
  std::vector<Expr *> res;
  auto *schedule_block_ptr = schedule_block.As<ScheduleBlock>();
  CHECK(schedule_block_ptr);
  res.push_back(&schedule_block_ptr->body);
  return res;
}
std::vector<const Expr *> ScheduleBlockRealize::expr_fields() const {
  std::vector<const Expr *> res;
  auto *schedule_block_ptr = schedule_block.As<ScheduleBlock>();
  CHECK(schedule_block_ptr);
  res.push_back(&schedule_block_ptr->body);
  return res;
}

Expr IfThenElse::Make(Expr condition, Expr true_case, Expr false_case) {
  if (true_case.defined() && (!true_case.As<Block>()))
    true_case = ir::Block::Make({true_case});
  if (false_case.defined() && (!false_case.As<Block>()))
    false_case = ir::Block::Make({false_case});
  auto node = make_shared<IfThenElse>(condition, true_case, false_case);
  return Expr(node);
}

IfThenElse::IfThenElse(Expr condition, Expr true_case, Expr false_case)
    : ExprNode(Type()),
      condition(condition),
      true_case(true_case),
      false_case(false_case) {
  CHECK(condition.defined());
  CHECK(true_case.defined());
}
std::vector<Expr *> IfThenElse::expr_fields() {
  return {&condition, &true_case, &false_case};
}
std::vector<const Expr *> IfThenElse::expr_fields() const {
  return {&condition, &true_case, &false_case};
}

Expr Store::Make(Expr tensor, Expr value, const std::vector<Expr> &indices) {
  CHECK(tensor.As<_Tensor_>()) << "tensor should be _Tensor_ type";
  auto node = make_shared<Store>();
  node->tensor = tensor;
  node->value = value;
  node->indices =
      utils::GetCompitableStoreLoadIndices(tensor.as_tensor_ref(), indices);

  if (tensor->type() != Void()) {
    node->set_type(
        tensor->type().ElementOf().with_lanes(node->index().type().lanes()));
  }
  return Expr(node);
}

Expr Store::index() const {
  auto *tensor_n = tensor.As<ir::_Tensor_>();
  CHECK(tensor_n);
  if (indices.size() == 1) {
    return indices[0];
  }
  Expr res = cinn::common::IndiceToAbsOffset(tensor_n->shape, indices);
  return res;
}

void Store::replace(Expr old_op, Expr new_op) {
  if (value == old_op) {
    value = new_op;
  }
  if (tensor == old_op) {
    tensor = new_op;
  }
  for (int i = 0; i < indices.size(); i++) {
    if (indices[i] == old_op) {
      indices[i] = new_op;
    }
  }
}

void Select::replace(Expr old_op, Expr new_op) {
  if (condition == old_op) {
    condition = new_op;
  }
  if (true_value == old_op) {
    true_value = new_op;
  }
  if (false_value == old_op) {
    false_value = new_op;
  }
}

void Cast::replace(Expr old_op, Expr new_op) {
  if (v() == old_op) {
    v() = new_op;
  }
}

const std::string &Store::name() const {
  auto *t = tensor.As<ir::_Tensor_>();
  CHECK(t);
  return t->name;
}

Type Store::type() const { return value.type(); }

std::vector<Expr *> Store::expr_fields() {
  std::vector<Expr *> exprs({&tensor, &value});
  for (auto &idx : indices) exprs.push_back(&idx);
  return exprs;
}

std::vector<const Expr *> Store::expr_fields() const {
  std::vector<const Expr *> exprs({&tensor, &value});
  for (auto &idx : indices) exprs.push_back(&idx);
  return exprs;
}

void Store::Verify() const { CHECK(tensor.defined()); }

Expr Alloc::Make(Expr dest,
                 Type type,
                 const std::vector<Expr> &extents,
                 Expr condition,
                 Expr body) {
  auto node = make_shared<Alloc>();
  CHECK(dest.As<_Buffer_>()) << "Alloc destination only supports Buffer";
  node->destination = dest;
  node->extents = extents;
  node->condition = condition;
  node->body = body;
  node->set_type(type);
  return Expr(node);
}

int32_t Alloc::ConstantAllocationSize() const {
  return ConstantAllocationSize(extents);
}

int32_t Alloc::ConstantAllocationSize(const std::vector<Expr> &extents) {
  int32_t res{1};
  for (auto &e : extents) {
    auto *p = e.As<IntImm>();
    CHECK(p) << "extent should be IntImm";
    res *= p->value;
  }
  return res;
}
std::vector<Expr *> Alloc::expr_fields() {
  std::vector<Expr *> res;
  for (auto &x : extents) res.push_back(&x);
  res.push_back(&condition);
  res.push_back(&body);
  return res;
}
std::vector<const Expr *> Alloc::expr_fields() const {
  std::vector<const Expr *> res;
  for (auto &x : extents) res.push_back(&x);
  res.push_back(&condition);
  res.push_back(&body);
  return res;
}

Expr Free::Make(Expr dest) {
  auto node = make_shared<Free>();
  CHECK(dest.As<_Buffer_>()) << "Free destination only supports Buffer";
  node->destination = dest;
  return Expr(node);
}

Expr Call::Make(Type type,
                const std::string &name,
                const std::vector<Expr> &read_args,
                const std::vector<Expr> &write_args,
                CallType call_type,
                FunctionRef func,
                int value_index,
                const std::map<std::string, attr_t> &attrs) {
  for (size_t i = 0; i < read_args.size(); ++i) {
    CHECK(read_args[i].defined());
  }

  auto node = cinn::common::make_shared<Call>(type);
  node->name = name;
  node->read_args = read_args;
  node->write_args = write_args;
  node->call_type = call_type;
  node->func = func;
  node->value_index = value_index;
  node->set_type(type);
  node->attrs = attrs;
  return Expr(node);
}

void Call::replace(Expr old_op, Expr new_op) {
  for (int i = 0; i < read_args.size(); i++) {
    if (read_args[i] == old_op) {
      read_args[i] = new_op;
    }
  }
  for (int i = 0; i < write_args.size(); i++) {
    if (read_args[i] == old_op) {
      read_args[i] = new_op;
    }
  }
}

std::vector<Expr *> Call::expr_fields() {
  std::vector<Expr *> res;
  for (auto &x : read_args) res.push_back(&x);
  for (auto &x : write_args) res.push_back(&x);
  return res;
}
std::vector<const Expr *> Call::expr_fields() const {
  std::vector<const Expr *> res;
  for (auto &x : read_args) res.push_back(&x);
  for (auto &x : write_args) res.push_back(&x);
  return res;
}
void Call::Verify() const {}

Expr PolyFor::Make(Var iterator,
                   Expr init_val,
                   Expr condition,
                   Expr inc,
                   ForType for_type,
                   DeviceAPI device_api,
                   Expr body,
                   VectorizeInfo vectorize_info,
                   BindInfo bind_info) {
  auto n = make_shared<PolyFor>();
  n->iterator = iterator;
  n->init = init_val;
  n->condition = condition;
  n->inc = inc;
  n->device_api = device_api;
  n->body = body.As<ir::Block>() ? body : ir::Block::Make({body});
  n->set_for_type(for_type);
  n->set_vectorize_info(vectorize_info);
  n->set_bind_info(bind_info);

  if (n->is_vectorized()) CHECK(n->vectorize_info().valid());
  if (n->is_binded() && bind_info.offset >= 0) CHECK(n->bind_info().valid());

  return Expr(n);
}
std::vector<Expr *> PolyFor::expr_fields() {
  return {&init, &condition, &inc, &body};
}
std::vector<const Expr *> PolyFor::expr_fields() const {
  return {&init, &condition, &inc, &body};
}

Expr PolyFor::ExtractExtent() const {
  auto nodes = ir::ir_utils::CollectIRNodes(condition, [&](const Expr *e) {
    return e->As<NE>() ||   //
           e->As<EQ>() ||   //
           e->As<Min>() ||  //
           e->As<Max>();
  });

  if (!nodes.empty()) {
    return Expr();
  }

  auto *le_n = condition.As<LE>();
  auto *lt_n = condition.As<LT>();
  if (!(le_n || lt_n)) return Expr();

  if (le_n) {
    if (le_n->a() != Expr(iterator)) return Expr();
    auto *le_b_int = le_n->b().As<IntImm>();
    if (le_b_int)
      return Expr(make_shared<IntImm>(Int(32), le_b_int->value + 1));
    return Add::Make(le_n->b(), Expr(1));
  }

  if (lt_n) {
    if (lt_n->a() != Expr(iterator)) return Expr();
    return lt_n->b();
  }
  return Expr();
}

bool Var::operator==(const Var &o) const {
  return o->name == operator->()->name;
}
bool Var::operator!=(const Var &o) const { return !(*this == o); }

Var &Var::operator=(_Var_ *x) {
  *this = Var(x);
  return *this;
}

Var &Var::operator=(const _Var_ *x) {
  *this = x->Copy();
  return *this;
}

Expr Load::Make(Expr tensor, const std::vector<Expr> &origin_indices) {
  CHECK(tensor->type().valid());
  const auto indices = utils::GetCompitableStoreLoadIndices(
      tensor.as_tensor_ref(), origin_indices);
  CHECK(!indices.empty());
  TryElevateInt32ToInt64(indices);
  for (auto &idx : indices) {
    CHECK(idx.type().ElementOf() == Int(64) ||
          idx.type().ElementOf() == Int(32));
  }
  auto node = make_shared<Load>();
  node->tensor = tensor;
  node->indices = indices;
  node->set_type(node->type());
  return Expr(node);
}

void Load::convert_int32_to_int64() {
  IrNode::convert_int32_to_int64();
  tensor->convert_int32_to_int64();
}

Type Load::type() const {
  CHECK(tensor.defined());
  CHECK(tensor.type().valid());

  int lanes = 0;
  for (auto &idx : indices) lanes = std::max(lanes, idx.type().lanes());
  auto type = tensor.type().ElementOf().with_lanes(lanes);
  if (type.is_cpp_handle()) return type.set_cpp_handle(false);
  if (type.is_cpp_handle2()) return type.set_cpp_handle(true);
  return type;
}

std::vector<Expr *> Load::expr_fields() {
  std::vector<Expr *> exprs({&tensor});
  for (auto &idx : indices) exprs.push_back(&idx);
  return exprs;
}

std::vector<const Expr *> Load::expr_fields() const {
  std::vector<const Expr *> exprs({&tensor});
  for (auto &idx : indices) exprs.push_back(&idx);
  return exprs;
}

Expr Load::index() const {
  if (is_addr_tensor()) {
    auto *tensor_n = tensor.As<_Tensor_>();
    CHECK(tensor_n);
    VLOG(3) << "Begin Load::index IndiceToAbsOffset of tensor: "
            << this->name();
    if (indices.size() == 1) {
      return indices[0];
    }
    Expr res = cinn::common::IndiceToAbsOffset(tensor_n->shape, indices);
    return res;
  } else {
    PADDLE_ENFORCE_EQ(indices.size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "The indices size of Load node should be 1"));
    return indices[0];
  }
}

const std::string &Load::name() const {
  auto *t = tensor.As<ir::_Tensor_>();
  CHECK(t);
  return t->name;
}

void Load::Verify() const {
  CHECK(tensor.defined());
  CHECK(!indices.empty()) << "At least one indice is needed";
  for (auto &indice : indices) {
    CHECK(indice.defined());
    CHECK(indice.type().ElementOf() == type_of<int32_t>() ||
          indice.type().ElementOf() == type_of<int64_t>())
        << "get type " << indice.type() << " vs (int64 or int32)";
  }
}

bool LoadStoreAddrMnger::is_addr_tensor() const {
  return tensor.As<_Tensor_>();
}
bool LoadStoreAddrMnger::is_addr_scalar() const { return !is_addr_tensor(); }

Expr Ramp::Make(Expr base, Expr stride, int lanes) {
  CHECK(base.defined());
  CHECK(stride.defined());
  CHECK(base.type().valid());
  CHECK(stride.type().valid());
  PADDLE_ENFORCE_EQ(stride.type(),
                    Int(32),
                    phi::errors::InvalidArgument(
                        "The stride of the node [Ramp] should be int32"));
  PADDLE_ENFORCE_GT(
      lanes,
      0,
      phi::errors::InvalidArgument(
          "The lanes of the node [Ramp] should be greater than 0"));

  auto *n = make_shared<Ramp>();
  n->base = base;
  n->stride = stride;
  n->lanes = lanes;
  Type type(base.type().type(), base.type().bits(), lanes);
  n->set_type(type);
  return Expr(n);
}

Expr Broadcast::Make(Expr value, int lanes) {
  CHECK(value.defined());
  CHECK(value.type().valid());

  auto *n = make_shared<Broadcast>();
  n->value = value;
  n->lanes = lanes;

  Type type(value.type().type(), value.type().bits(), lanes);
  n->set_type(type);

  return Expr(n);
}

Type Broadcast::type() const {
  return value.type().ElementOf().with_lanes(lanes);
}

Expr Sum::Make(const std::vector<Expr> &vs) {
  CHECK(!vs.empty());
  if (vs.size() == 1) return vs.front();

  auto *n = make_shared<Sum>();
  TryElevateInt32ToInt64(vs);
  auto type = vs.front().type();
  for (auto &v : vs) {
    PADDLE_ENFORCE_EQ(v.type(),
                      type,
                      phi::errors::InvalidArgument(
                          "The operands' types of the node [Sum] don't match"));
  }

  n->operands() = vs;

  n->set_type(vs.front()->type());

  return Expr(n);
}

Expr Product::Make(const std::vector<Expr> &vs) {
  PADDLE_ENFORCE_GE(
      vs.size(),
      1,
      phi::errors::InvalidArgument("The operands of the node [Product] "
                                   "should have at least one element"));

  auto *n = make_shared<Product>();
  TryElevateInt32ToInt64(vs);
  auto type = vs.front().type();
  for (auto &v : vs)
    PADDLE_ENFORCE_EQ(
        v.type(),
        type,
        phi::errors::InvalidArgument("The operands' types of the node "
                                     "[Product] don't match"));

  n->operands() = vs;

  n->set_type(vs.front()->type());

  return Expr(n);
}

Expr FracOp::Make(Expr n, Expr d) {
  auto *node = make_shared<FracOp>();
  node->a() = n;
  node->b() = d;
  return Expr(node);
}

ir::Module _Module_::Make(const std::string &name, Target target) {
  auto n = make_shared<_Module_>();
  n->name = name;
  n->target = target;
  return ir::Module(n);
}

Expr PrimitiveNode::Make(const std::string &name,
                         const std::map<std::string, attr_t> &attrs) {
  auto *n = make_shared<PrimitiveNode>();
  n->name = name;
  n->attrs = attrs;
  return Expr(n);
}

Expr Reduce::Make(Reduce::ReduceType reduce_type,
                  Expr init,
                  Expr body,
                  const std::vector<Var> &reduce_axis) {
  CHECK(body.defined());
  CHECK(init.defined());
  auto n = cinn::common::make_shared<Reduce>();
  n->init = init;
  n->body = body;
  n->reduce_type = reduce_type;
  n->reduce_axis.append(reduce_axis.begin(), reduce_axis.end());
  CHECK(body.type().valid());
  if (init.defined()) {
    CHECK(init.type().valid());
    PADDLE_ENFORCE_EQ(
        init.type(),
        body.type(),
        phi::errors::InvalidArgument(
            "The type of the init and the body of the node [Reduce] "
            "should be the same"));
  }
  n->set_type(body.type());
  return Expr(n);
}

Type Reduce::type() const { return body.type().ElementOf(); }

std::vector<Expr *> Reduce::expr_fields() {
  std::vector<Expr *> res;
  if (init.defined()) {
    res.push_back(&init);
  }
  CHECK(body.defined());
  res.push_back(&body);
  return res;
}
std::vector<const Expr *> Reduce::expr_fields() const {
  std::vector<const Expr *> res;
  if (init.defined()) {
    res.push_back(&init);
  }
  CHECK(body.defined());
  res.push_back(&body);
  return res;
}

void Reduce::Verify() const {
  CHECK(init.defined());
  CHECK(body.defined());
  CHECK(!reduce_axis.empty()) << "At least one reduce axis is needed";
  PADDLE_ENFORCE_EQ(
      init.type(),
      body.type(),
      phi::errors::InvalidArgument(
          "The type of the init and the body of the node [Reduce] "
          "should be the same"));
}

Type Select::type() const {
  PADDLE_ENFORCE_EQ(
      true_value.type(), false_value.type(), "Type of Select must be same");
  return type_;
}

void Select::Verify() const {
  CHECK(condition.defined());
  CHECK(true_value.defined());
  CHECK(false_value.defined());
  CHECK(condition.type().is_bool())
      << "Select Node's condition should be a boolean";
  PADDLE_ENFORCE_EQ(true_value.type(),
                    false_value.type(),
                    phi::errors::InvalidArgument(
                        "Select Node's true_value and false_value should have "
                        "the same type"));
}

void Free::Verify() const { CHECK(destination.defined()); }

void Alloc::Verify() const { CHECK(destination.defined()); }

void For::Verify() const {
  CHECK(loop_var.defined());
  CHECK(min.defined());
  CHECK(extent.defined());
  CHECK(body.defined());

  PADDLE_ENFORCE_EQ((loop_var->type() == type_of<int32_t>()) ||
                        (loop_var->type() == type_of<int64_t>()),
                    true,
                    ::common::errors::InvalidArgument(
                        "loop var's type must be int32 or int64"));
  PADDLE_ENFORCE_EQ((min->type() == type_of<int32_t>()) ||
                        (min->type() == type_of<int64_t>()),
                    true,
                    ::common::errors::InvalidArgument(
                        "loop min's type must be int32 or int64"));
  PADDLE_ENFORCE_EQ((extent->type() == type_of<int32_t>()) ||
                        (extent->type() == type_of<int64_t>()),
                    true,
                    ::common::errors::InvalidArgument(
                        "loop extent's type must be int32 or int64"));
}

void PolyFor::Verify() const {
  CHECK(iterator.defined());
  CHECK(init.defined());
  CHECK(condition.defined());
  CHECK(inc.defined());
  CHECK(body.defined());

  PADDLE_ENFORCE_EQ(
      iterator->type(),
      type_of<int32_t>(),
      phi::errors::InvalidArgument("iterator's type must be int32"));
  PADDLE_ENFORCE_EQ(init.type(),
                    type_of<int32_t>(),
                    phi::errors::InvalidArgument("init's type must be int32"));
  PADDLE_ENFORCE_EQ(
      condition.type(),
      type_of<bool>(),
      phi::errors::InvalidArgument("condition's type must be bool"));
  PADDLE_ENFORCE_EQ(inc.type(),
                    type_of<int32_t>(),
                    phi::errors::InvalidArgument("inc's type must be int32"));
}

void Ramp::Verify() const {
  CHECK(base.defined());
  CHECK(stride.defined());
}

void FracOp::Verify() const {
  CHECK(a().defined());
  CHECK(b().defined());
  PADDLE_ENFORCE_EQ(
      a().type(),
      b().type(),
      phi::errors::InvalidArgument(
          "The type of the operands of the node [FracOp] should be "
          "the same"));
}

void Broadcast::Verify() const { CHECK(value.defined()); }

void MultiOperandVerify(llvm::ArrayRef<Expr> operands) {
  Type operand_type = operands.front().type();
  CHECK(operand_type.valid());
  for (int i = 1; i < operands.size(); i++) {
    CHECK(operands[i].defined());
    PADDLE_ENFORCE_EQ(operands[i].type(),
                      operand_type,
                      phi::errors::InvalidArgument(
                          "The operands' types of the node don't match"));
  }
}

Type Product::type() const { return operands().front().type(); }

void Product::Verify() const {
  PADDLE_ENFORCE_GT(operands().size(),
                    1UL,
                    phi::errors::InvalidArgument(
                        "Product node should have more than 1 operands"));
  MultiOperandVerify(operands());
}

Type Sum::type() const { return operands().front().type(); }

void Sum::Verify() const {
  PADDLE_ENFORCE_GT(operands().size(),
                    1UL,
                    phi::errors::InvalidArgument(
                        "Sum node should have more than 1 operands"));
  MultiOperandVerify(operands());
}

void Block::Verify() const {}

void PrimitiveNode::Verify() const {}

}  // namespace ir
}  // namespace cinn
