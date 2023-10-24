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

/**
 * This file contains all the internal representations used in CINN project.
 */
#pragma once

#include <absl/types/variant.h>

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "paddle/cinn/common/shared.h"
#include "paddle/cinn/common/type.h"
#include "paddle/cinn/ir/function_base.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/utils/small_vector.h"

namespace cinn {

namespace poly {
class Stage;
}  // namespace poly

namespace ir {
class Buffer;
class BufferRange;
struct LoweredFunc;
class Module;

using common::Object;
using common::Shared;
// NOTE attr_t only support POD, can not contain Expr or other IR nodes, or the
// IRVisitor or IRCopy on PrimitiveNode will result in undefined behavior.
using attr_t = absl::variant<int, float, bool, std::string>;

/**
 * Cast a node to another type, can't change the width.
 */
struct Cast : public ExprNode<Cast> {
  Cast() : ExprNode(1) {}

  static Expr Make(Type t, Expr v);

  template <typename T>
  static Expr Make(Type t, T v) {
    return Make(t, Expr(v));
  }

  Expr& v() { return operand(0); }
  const Expr& v() const { return operand(0); }

  void Verify() const override;

  static const IrNodeTy _node_type_ = IrNodeTy::Cast;

  std::vector<Expr*> expr_fields() override { return {&operand(0)}; }
  std::vector<const Expr*> expr_fields() const override {
    return {&operand(0)};
  }
};

/**
 * The sum of two expressions.
 */
struct Add : public BinaryOpNode<Add> {
  Add(Expr a, Expr b);

  static Expr Make(Expr a, Expr b);

  void Verify() const override;

  static const IrNodeTy _node_type_ = IrNodeTy::Add;
};

/**
 * The difference of two expressions.
 */
struct Sub : public BinaryOpNode<Sub> {
  Sub(Expr a, Expr b) : BinaryOpNode<Sub>(a.type(), a, b) {}

  static Expr Make(Expr a, Expr b);

  void Verify() const override;

  static const IrNodeTy _node_type_ = IrNodeTy::Sub;
};

/**
 * The product of two expressions.
 */
struct Mul : public BinaryOpNode<Mul> {
  Mul(Expr a, Expr b) : BinaryOpNode<Mul>(a.type(), a, b) {}

  static Expr Make(Expr a, Expr b);

  void Verify() const override;

  static const IrNodeTy _node_type_ = IrNodeTy::Mul;
};

/**
 * The ratio of two expressions.
 */
struct Div : public BinaryOpNode<Div> {
  Div(Expr a, Expr b) : BinaryOpNode<Div>(a.type(), a, b) {}

  static Expr Make(Expr a, Expr b);
  void Verify() const override;
  static const IrNodeTy _node_type_ = IrNodeTy::Div;
};

/**
 * The mod of two expressions.
 */
struct Mod : public BinaryOpNode<Mod> {
  Mod(Expr a, Expr b) : BinaryOpNode<Mod>(a.type(), a, b) {}

  static Expr Make(Expr a, Expr b);
  void Verify() const override;
  static const IrNodeTy _node_type_ = IrNodeTy::Mod;
};

/**
 * The lesser of two expressions.
 */
struct Min : public BinaryOpNode<Min> {
  Min(Expr a, Expr b) : BinaryOpNode<Min>(a.type(), a, b) {}

  static Expr Make(Expr a, Expr b);
  void Verify() const override;
  static const IrNodeTy _node_type_ = IrNodeTy::Min;
};

/**
 * The larger of two expressions.
 */
struct Max : public BinaryOpNode<Max> {
  Max(Expr a, Expr b) : BinaryOpNode<Max>(a.type(), a, b) {}

  static Expr Make(Expr a, Expr b);

  void Verify() const override;

  static const IrNodeTy _node_type_ = IrNodeTy::Max;
};

/**
 * Tell whether the first expression equals to the second expression.
 */
struct EQ : public BinaryOpNode<EQ> {
  EQ(Expr a, Expr b) : BinaryOpNode<EQ>(a.type(), a, b) {}

  Type type() const { return Bool(a()->type().lanes()); }

  static Expr Make(Expr a, Expr b);
  void Verify() const override;
  static const IrNodeTy _node_type_ = IrNodeTy::EQ;
};

/**
 * Tell whether the first expression not equals to the second expression.
 */
struct NE : public BinaryOpNode<NE> {
  NE(Expr a, Expr b) : BinaryOpNode<NE>(a.type(), a, b) {}

  Type type() const { return Bool(a()->type().lanes()); }

  static Expr Make(Expr a, Expr b);
  void Verify() const override;
  static const IrNodeTy _node_type_ = IrNodeTy::NE;
};

/**
 * Tell whether the first expression is lower than the second expression.
 */
struct LT : public BinaryOpNode<LT> {
  LT(Expr a, Expr b) : BinaryOpNode<LT>(a.type(), a, b) {}

  Type type() const { return Bool(a()->type().lanes()); }

  static Expr Make(Expr a, Expr b);
  void Verify() const override;
  static const IrNodeTy _node_type_ = IrNodeTy::LT;
};

/**
 * Tell whether the first expression is no larger than the second expression.
 */
struct LE : public BinaryOpNode<LE> {
  LE(Expr a, Expr b) : BinaryOpNode<LE>(a.type(), a, b) {}

  Type type() const { return Bool(a()->type().lanes()); }

  static Expr Make(Expr a, Expr b);
  void Verify() const override;
  static const IrNodeTy _node_type_ = IrNodeTy::LE;
};

/**
 * Tell whether the first expression is larger than the second expression.
 */
struct GT : public BinaryOpNode<GT> {
  GT(Expr a, Expr b) : BinaryOpNode<GT>(a.type(), a, b) {}

  Type type() const { return Bool(a()->type().lanes()); }

  static Expr Make(Expr a, Expr b);
  void Verify() const override;
  static const IrNodeTy _node_type_ = IrNodeTy::GT;
};

/**
 * Tell whether the first expression is not less than the second expression.
 */
struct GE : public BinaryOpNode<GE> {
  GE(Expr a, Expr b) : BinaryOpNode<GE>(a.type(), a, b) {}

  Type type() const { return Bool(a()->type().lanes()); }

  static Expr Make(Expr a, Expr b);
  void Verify() const override;
  static const IrNodeTy _node_type_ = IrNodeTy::GE;
};

/**
 * Logical and.
 */
struct And : public BinaryOpNode<And> {
  And(Expr a, Expr b) : BinaryOpNode<And>(a.type(), a, b) {
    CHECK(a->type().is_bool());
    CHECK(b->type().is_bool());
  }

  Type type() const { return Bool(a()->type().lanes()); }

  static Expr Make(Expr a, Expr b);
  void Verify() const override;
  static const IrNodeTy _node_type_ = IrNodeTy::And;
};

/**
 * -x
 */
struct Minus : public UnaryOpNode<Minus> {
  explicit Minus(Expr x) : UnaryOpNode<Minus>(x.type(), x) {}

  static Expr Make(Expr a);
  void Verify() const override;
  static const IrNodeTy _node_type_ = IrNodeTy::Minus;
};

/**
 * Logical or.
 */
struct Or : public BinaryOpNode<Or> {
  Or(Expr a, Expr b) : BinaryOpNode<Or>(Bool(), a, b) {
    CHECK(a->type().is_bool());
    CHECK(b->type().is_bool());
  }

  static Expr Make(Expr a, Expr b);

  Type type() const override;
  void Verify() const override;

  static const IrNodeTy _node_type_ = IrNodeTy::Or;
};

/**
 * Logical not.
 */
struct Not : public UnaryOpNode<Not> {
  explicit Not(Expr v) : UnaryOpNode<Not>(Bool(), v) {}

  static Expr Make(Expr v);

  Type type() const override;
  void Verify() const override;

  static const IrNodeTy _node_type_ = IrNodeTy::Not;
};

struct Let : public ExprNode<Let> {
  Expr symbol;
  Expr body;

  static Expr Make(Expr symbol, Expr body);

  Type type() const override;

  void Verify() const override;

  static const IrNodeTy _node_type_ = IrNodeTy::Let;

  std::vector<Expr*> expr_fields() override {
    if (!body.defined()) return {&symbol};
    return {&symbol, &body};
  }
  std::vector<const Expr*> expr_fields() const override {
    if (!body.defined()) return {&symbol};
    return {&symbol, &body};
  }
};

enum CallType : int {
  //! Extern "C" function.
  Extern = 0,
  //! CINN-style call, call a CINN function.
  CINN,
  //! Intrinsic functions.
  Intrinsic,
  //! Generated from ISL Ast.
  ISL,
};
struct Call : public ExprNode<Call> {
  explicit Call(Type t) : ExprNode<Call>(t) {}

  //! The name of the function/intrinsic.
  std::string name;
  //! The arguments.
  std::vector<Expr> read_args;
  std::vector<Expr> write_args;
  //! the attribute of this CallNode.
  std::map<std::string, attr_t> attrs;
  //! Type of calls.
  CallType call_type;
  //! The function to be called.
  FunctionRef func;
  //! The output value index if func's value is a tuple.
  int value_index{-1};

  static Expr Make(Type type,
                   const std::string& name,
                   const std::vector<Expr>& read_args,
                   const std::vector<Expr>& write_args,
                   CallType call_type,
                   FunctionRef func = FunctionRef(),
                   int value_index = 0,
                   const std::map<std::string, attr_t>& attrs = {});

  void Verify() const override;

  inline size_t total_args_count() const {
    return read_args.size() + write_args.size();
  }

  inline bool is_extern_call() const { return call_type == CallType::Extern; }
  inline bool is_cinn_call() const { return call_type == CallType::CINN; }
  inline bool is_intrinsic_call() const {
    return call_type == CallType::Intrinsic;
  }
  inline bool is_isl_call() const { return call_type == CallType::ISL; }

  std::vector<Expr*> expr_fields() override;
  std::vector<const Expr*> expr_fields() const override;

  static const IrNodeTy _node_type_ = IrNodeTy::Call;
};

/**
 * Variable used as iterator value or bound definition.
 */
struct _Var_ : public ExprNode<_Var_> {
  std::string name;

  bool is_reduce_axis{false};
  //! Lower bound and upper bound of a axis.
  // @{
  Expr lower_bound;
  Expr upper_bound;
  // @}

  // ! Extra tag of this variable/axis.
  std::string tag;

  _Var_() = default;
  _Var_(const std::string& name, Type type)
      : ExprNode<_Var_>(type), name(name) {}

  static Expr Make(const std::string& name, const Type& type);
  //! Make a reduce axis.
  static Expr Make(Expr lower_bound,
                   Expr upper_bound,
                   const std::string& name,
                   bool is_reduce);

  void Verify() const override;

  Expr Copy() const override;

  static const IrNodeTy _node_type_ = IrNodeTy::_Var_;
};

//! A named variable.
struct Var : public IrNodeRef {
  Var() = default;
  explicit Var(IrNode* n) : IrNodeRef(n) {}
  explicit Var(const std::string& name_hint, Type t = type_of<int>())
      : Var(_Var_::Make(name_hint, t).ptr()) {}
  Var(Expr lower_bound,
      Expr upper_bound,
      const std::string& name,
      bool is_reduce = false)
      : Var(_Var_::Make(lower_bound, upper_bound, name, is_reduce)) {}
  Var(int upper_bound, const std::string& name)
      : Var(_Var_::Make(Expr(0), Expr(upper_bound), name, false)) {}
  Var(Expr upper_bound, const std::string& name)
      : Var(_Var_::Make(Expr(0), upper_bound, name, false)) {}

  operator Expr() { return Expr(get()); }
  operator Expr() const {
    Var v = *this;
    return Expr(v);
  }

  bool operator==(const Var& o) const;
  bool operator!=(const Var& o) const;

  Var& operator=(_Var_* x);
  Var& operator=(const _Var_* x);

  const _Var_* operator->() const { return get(); }
  _Var_* operator->() { return get(); }
  const _Var_* get() const { return static_cast<const _Var_*>(ptr()); }
  _Var_* get() { return static_cast<_Var_*>(ptr()); }
};

struct Reduce : public ExprNode<Reduce> {
  enum ReduceType {
    kSum = 0,
    kSub,
    kMul,
    kDiv,
    kMax,
    kMin,
    kAll,
    kAny,
  };

  //! The initial value.
  Expr init;

  // ! The body.
  Expr body;

  utils::SmallVector<Var, 4> reduce_axis;

  //! The type of the reduce operation.
  ReduceType reduce_type;

  static Expr Make(ReduceType reduce_type,
                   Expr init,
                   Expr body,
                   const std::vector<Var>& reduce_aixs);

  Type type() const override { return body.type().ElementOf(); }

  std::vector<Expr*> expr_fields() override;
  std::vector<const Expr*> expr_fields() const override;

  void Verify() const override;

  static const IrNodeTy _node_type_ = IrNodeTy::Reduce;
};

/**
 * Evaluates `true_value` and `false_value` then selects between them based on
 * `condition`.
 */
struct Select : public ExprNode<Select> {
  Expr condition;
  Expr true_value;
  Expr false_value;

  Select(Expr condition, Expr true_value, Expr false_value)
      : ExprNode<Select>(true_value.type()),
        condition(condition),
        true_value(true_value),
        false_value(false_value) {
    CHECK_EQ(true_value.type(), false_value.type());
    CHECK(condition.type().is_bool());
  }

  static Expr Make(Expr condition, Expr true_value, Expr false_value) {
    auto node = make_shared<Select>(condition, true_value, false_value);
    return Expr(node);
  }

  Type type() const override {
    CHECK_EQ(true_value.type(), false_value.type());
    return true_value.type();
  }

  void Verify() const override;

  std::vector<Expr*> expr_fields() override {
    return {&condition, &true_value, &false_value};
  }
  std::vector<const Expr*> expr_fields() const override {
    return {&condition, &true_value, &false_value};
  }

  static const IrNodeTy _node_type_ = IrNodeTy::Select;
};

struct LoadStoreAddrMnger {
  Expr tensor;  // Should be a tensor or a scalar.
  //! Tell whether the address is a tensor.
  bool is_addr_tensor() const;
  //! Tell whether the address is a scalar.
  bool is_addr_scalar() const;
};

/**
 * Load the value from a buffer (as an array).
 */
struct Load : public ExprNode<Load>, public LoadStoreAddrMnger {
  std::vector<Expr> indices;
  //! The abstract offset.
  Expr index() const;

  static Expr Make(Expr tensor, const std::vector<Expr>& indices);

  std::vector<Expr*> expr_fields() override;
  std::vector<const Expr*> expr_fields() const override;

  void Verify() const override;

  const std::string& name() const;

  Type type() const override;

  static const IrNodeTy _node_type_ = IrNodeTy::Load;
};

/**
 * Store a `value` to the buffer at a given `index`.
 */
struct Store : public ExprNode<Store>, public LoadStoreAddrMnger {
  Expr value;
  std::vector<Expr> indices;

  static Expr Make(Expr tensor, Expr value, const std::vector<Expr>& indices);

  std::vector<Expr*> expr_fields() override;
  std::vector<const Expr*> expr_fields() const override;

  void Verify() const override;

  const std::string& name() const;

  Type type() const override;
  Expr index() const;

  static const IrNodeTy _node_type_ = IrNodeTy::Store;
};

/**
 * Allocate a buffer with the given type and size. The buffer lives for at most
 * the duration of the body statement, within which it is freed.
 */
struct Alloc : public ExprNode<Alloc> {
  //! The destination of the allocation, this might be a buffer or a variable.
  Expr destination;
  //! Dimensions of this buffer (as a multi-dimensional array).
  std::vector<Expr> extents;
  // NOTE the condition might be undefined, that means always true.
  Expr condition;
  // NOTE the body might be undefined, that means no specific logic other than
  // default.
  Expr body;

  Alloc() : ExprNode(Type()) {}

  static Expr Make(Expr dest,
                   Type type,
                   const std::vector<Expr>& extents,
                   Expr condition,
                   Expr body);

  std::vector<Expr*> expr_fields() override;
  std::vector<const Expr*> expr_fields() const override;

  void Verify() const override;

  int32_t ConstantAllocationSize() const;
  static int32_t ConstantAllocationSize(const std::vector<Expr>& extents);

  static const IrNodeTy _node_type_ = IrNodeTy::Alloc;
};

/**
 * Free the resources associated with the given buffer.
 */
struct Free : public ExprNode<Free> {
  Expr destination;

  Free() : ExprNode(Type()) {}

  static Expr Make(Expr dest);

  void Verify() const override;

  static const IrNodeTy _node_type_ = IrNodeTy::Free;
};

struct IfThenElse : public ExprNode<IfThenElse> {
  Expr condition;
  Expr true_case;
  Expr false_case;

  IfThenElse(Expr condition, Expr true_case, Expr false_case);

  static Expr Make(Expr condition, Expr true_case, Expr false_case = Expr());

  void Verify() const override {
    CHECK(condition.defined());
    CHECK(true_case.defined());
    CHECK_EQ(condition.type(), type_of<bool>());
  }

  std::vector<Expr*> expr_fields() override;
  std::vector<const Expr*> expr_fields() const override;

  static const IrNodeTy _node_type_ = IrNodeTy::IfThenElse;
};

enum class ForType : int {
  Serial = 0,           //! Serial execution.
  Parallel = 1,         //! Parallel execution.
  Vectorized = 1 << 1,  //! Vector SIMD loop annotation.
  Unrolled = 1 << 2,    //! Unroll annotation.
  GPUThread = 1 << 3,   //! GPU Thread.
  GPUBlock = 1 << 4,    //! GPU Block.
  GPULane = 1 << 5,     //! GPU Lane.
  Default = 1 << 6,
};

struct VectorizeInfo {
  VectorizeInfo() = default;
  VectorizeInfo(int level, int factor) : level(level), factor(factor) {}

  int level{-1};
  int factor{-1};

  inline void set(int level, int factor) {
    this->level = level;
    this->factor = factor;
  }
  inline bool valid() const { return level >= 0 && factor > 0; }
};

struct BindInfo {
  BindInfo() = default;
  BindInfo(const ForType& for_type, const int& offset, const DeviceAPI& device)
      : for_type(for_type), offset(offset), device(device) {}

  ForType for_type{ForType::Default};
  int offset{-1};
  DeviceAPI device{DeviceAPI::UNK};

  inline void set(const ForType& for_type,
                  const int& offset,
                  const DeviceAPI& device) {
    this->for_type = for_type;
    this->offset = offset;
    this->device = device;
  }
  // offset should be 0-2, should correspond to the thread of x, y, z
  inline bool valid() const {
    return offset >= 0 && offset < 3 &&
           (for_type == ForType::GPUThread || for_type == ForType::GPUBlock);
  }
};

struct ForBase {
  ForType for_type() const { return for_type_; }
  void set_for_type(ForType x) { for_type_ = x; }

  void set_vectorize_info(const VectorizeInfo& x) {
    if (x.valid()) set_vectorized();
    vectorize_info_ = x;
  }
  void set_bind_info(const BindInfo& x) {
    if (x.valid()) set_binded(x.for_type);
    bind_info_ = x;
  }
  const VectorizeInfo& vectorize_info() const { return vectorize_info_; }
  const BindInfo& bind_info() const { return bind_info_; }

  void reset_vectorize_info() {
    set_vectorized(false);
    vectorize_info_.factor = -1;
    vectorize_info_.level = -1;
  }
  void reset_bind_info() {
    set_binded(bind_info_.for_type, false);
    bind_info_.offset = -1;
    bind_info_.device = DeviceAPI::UNK;
  }

  void set_serial() { for_type_ = ForType::Serial; }

  void set_unrolled(bool x = true) {
    if (x)
      set_for_type_flag(ForType::Unrolled);
    else
      unset_for_type_flag(ForType::Unrolled);
  }
  void set_vectorized(bool x = true) {
    if (x)
      set_for_type_flag(ForType::Vectorized);
    else
      unset_for_type_flag(ForType::Vectorized);
  }
  void set_parallel(bool x = true) {
    if (x)
      set_for_type_flag(ForType::Parallel);
    else
      unset_for_type_flag(ForType::Parallel);
  }
  void set_binded(ForType for_type, bool x = true) {
    if (x)
      set_for_type_flag(for_type);
    else
      unset_for_type_flag(for_type);
  }

  inline bool is_serial() const { return for_type_ == ForType::Serial; }
  inline bool is_default() const { return for_type_ == ForType::Default; }
  inline bool is_unrolled() const {
    return tell_for_type_flag(ForType::Unrolled);
  }
  inline bool is_vectorized() const {
    return tell_for_type_flag(ForType::Vectorized);
  }
  inline bool is_parallel() const {
    return tell_for_type_flag(ForType::Parallel);
  }
  inline bool is_binded() const {
    return tell_for_type_flag(ForType::GPUBlock) ||
           tell_for_type_flag(ForType::GPUThread);
  }
  inline bool is_gpu_block_binded() const {
    return tell_for_type_flag(ForType::GPUBlock);
  }
  inline bool is_gpu_thread_binded() const {
    return tell_for_type_flag(ForType::GPUThread);
  }

 private:
  inline void set_for_type_flag(ForType type) {
    *reinterpret_cast<int*>(&for_type_) |= static_cast<int>(type);
  }
  inline void unset_for_type_flag(ForType type) {
    *reinterpret_cast<int*>(&for_type_) &= ~static_cast<int>(type);
  }
  inline bool tell_for_type_flag(ForType type) const {
    return static_cast<int>(for_type_) & static_cast<int>(type);
  }

  ForType for_type_{ForType::Serial};
  VectorizeInfo vectorize_info_;
  BindInfo bind_info_;
};

/// LLVM loop unroll metadata infomation
struct LLVMForLoopMeta {
  enum UnrollMode { DefaultUnroll, FullyUnroll, NoUnroll };

  UnrollMode unroll_mode{DefaultUnroll};
  bool vectorization{true};
};

struct For : public ExprNode<For>, public ForBase {
  //! The loop variable.
  Var loop_var;
  //! The minimum value of the iteration.
  Expr min;
  //! The extent of the iteration.
  Expr extent;

  Expr body;

  DeviceAPI device_api;

  LLVMForLoopMeta metadata;

  static Expr Make(Var loop_var,
                   Expr min,
                   Expr extent,
                   ForType for_type,
                   DeviceAPI device_api,
                   Expr body,
                   VectorizeInfo vector_info = VectorizeInfo(),
                   BindInfo bind_info = BindInfo());

  void Verify() const override;

  std::vector<Expr*> expr_fields() override;
  std::vector<const Expr*> expr_fields() const override;

  static const IrNodeTy _node_type_ = IrNodeTy::For;
};

//! Polyhedral forloop, which condition is more complex than the normal `For`.
struct PolyFor : public ExprNode<PolyFor>, public ForBase {
  //! The iterator variable.
  Var iterator;
  // Initial value of the iterator.
  Expr init;
  //! The condition to continue the loop.
  Expr condition;
  //! Increase the iterator.
  Expr inc;
  //! The forloop body.
  Expr body;

  DeviceAPI device_api;

  PolyFor() : ExprNode(Type()) {}

  Expr ExtractExtent() const;

  static Expr Make(Var iterator,
                   Expr init_val,
                   Expr condition,
                   Expr inc,
                   ForType for_type,
                   DeviceAPI device_api,
                   Expr body,
                   VectorizeInfo vector_info = VectorizeInfo(),
                   BindInfo bind_info = BindInfo());

  void Verify() const override;

  std::vector<Expr*> expr_fields() override;
  std::vector<const Expr*> expr_fields() const override;

  static const IrNodeTy _node_type_ = IrNodeTy::PolyFor;
};

//! A linear ramp node.
struct Ramp : public ExprNode<Ramp> {
  Expr base, stride;
  int lanes;

  static Expr Make(Expr base, Expr stride, int lanes);

  void Verify() const override;

  static const IrNodeTy _node_type_ = IrNodeTy::Ramp;
};

//! A vector with `lanes` elements and all of them are `value`.
struct Broadcast : public ExprNode<Broadcast> {
  Expr value;
  int lanes;

  static Expr Make(Expr value, int lanes);

  Type type() const override;

  void Verify() const override;

  std::vector<Expr*> expr_fields() override { return {&value}; }
  std::vector<const Expr*> expr_fields() const override { return {&value}; }

  static const IrNodeTy _node_type_ = IrNodeTy::Broadcast;
};

struct FracOp : public BinaryOpNode<FracOp> {
  FracOp() { operands().resize(2); }

  static Expr Make(Expr n, Expr d);

  bool is_constant() const { return a().is_constant() && b().is_constant(); }

  double get_constant() const {
    CHECK(is_constant());
    CHECK_NE(b().get_constant(), 0.f);
    return a().get_constant() / b().get_constant();
  }

  void Verify() const override;

  static const IrNodeTy _node_type_ = IrNodeTy::FracOp;

  using ExprNode<FracOp>::operands;
};

struct Product : public ExprNode<Product> {
  static Expr Make(const std::vector<Expr>& vs);

  using ExprNode<Product>::operand;

  Type type() const override { return operands().front().type(); }

  void Verify() const override;

  static const IrNodeTy _node_type_ = IrNodeTy::Product;
};

struct Sum : public ExprNode<Sum> {
  static Expr Make(const std::vector<Expr>& vs);

  using ExprNode<Sum>::operand;

  Type type() const override { return operands().front().type(); }

  void Verify() const override;

  static const IrNodeTy _node_type_ = IrNodeTy::Sum;
};

struct Block : public ExprNode<Block> {
  std::vector<Expr> stmts;

  Block() : ExprNode(Type()) {}

  static Expr Make(const std::vector<Expr>& stmts);

  void Verify() const override;

  std::vector<Expr*> expr_fields() override;
  std::vector<const Expr*> expr_fields() const override;

  static const IrNodeTy _node_type_ = IrNodeTy::Block;
};

// ScheduleBlock is the unit of schedule IR which represents tensor's
// computation
struct ScheduleBlock : public ExprNode<ScheduleBlock> {
  std::vector<Var> iter_vars;
  // BufferRange(s) which is read in this schedule block, it is used to
  // analyze, not a real computation expression. Must be AST DFS order.
  std::vector<Expr> read_buffers;
  // BufferRange(s) which is written in this schedule block, it is used to
  // analyze, not a real computation expression. Must be AST DFS order.
  std::vector<Expr> write_buffers;
  // Additional attributes about this schedulable block,
  // which take some auxiliary hints for future transformations.
  std::map<std::string, attr_t> attrs;
  std::string name;
  Expr body;

  static Expr Make(const std::vector<Var>& iter_vars,
                   const std::vector<Expr>& read_buffers,
                   const std::vector<Expr>& write_buffers,
                   const std::string& name,
                   Expr body);

  void Verify() const override;

  std::vector<Expr*> expr_fields() override;
  std::vector<const Expr*> expr_fields() const override;

  static const IrNodeTy _node_type_ = IrNodeTy::ScheduleBlock;
};

// ScheduleBlockRealize is used to execute ScheduleBlock with the binding
// iter_values
struct ScheduleBlockRealize : public ExprNode<ScheduleBlockRealize> {
  // values of the iter_vars
  std::vector<Expr> iter_values;
  Expr schedule_block;

  static Expr Make(const std::vector<Expr>& iter_values,
                   const Expr& schedule_block);

  void Verify() const override;

  std::vector<Expr*> expr_fields() override;
  std::vector<const Expr*> expr_fields() const override;

  static const IrNodeTy _node_type_ = IrNodeTy::ScheduleBlockRealize;
};

/**
 * Content of a module.
 */
struct _Module_ : public ExprNode<_Module_> {
  std::string name;
  Target target;
  std::vector<Expr> buffers;
  std::vector<Expr> functions;
  std::vector<Expr> submodules;

  static ir::Module Make(const std::string& name, Target target);

  void Verify() const override {}

  static const IrNodeTy _node_type_ = IrNodeTy::_Module_;
};

/**
 * \brief PrimitiveNode holds the contept of Primitive in CINN.
 * A Primitive is a basic Call to some Expr function, it is introduced to create
 * several level of coarsed-grained IR nodes for better IR optimization and
 * hardware adaption.
 */
struct PrimitiveNode : public ExprNode<PrimitiveNode> {
  std::string name;
  //! the inputs of the PrimitiveNode, the vector<vector<Expr>> can hold
  //! variadic arguments.
  std::vector<std::vector<Expr>> arguments;
  //! the attribute of this PrimitiveNode.
  std::map<std::string, attr_t> attrs;

  static Expr Make(const std::string& name,
                   const std::map<std::string, attr_t>& attrs);

  void Verify() const override;

  static const IrNodeTy _node_type_ = IrNodeTy::PrimitiveNode;
};

// possiable keys of attributes in ir nodes with are listed in the following
// namespace
namespace attr {

// max permitted steps for auto_unroll, used in unroll_loop pass
constexpr const char* auto_unroll_max_step = "auto_unroll_max_step";
// record the extra loop built during ComputeAt, used for calculate the size of
// temp buffer in post-processing
constexpr const char* compute_at_extra_var = "compute_at_extra_var";
// record the extra loop built during ReverseComputeAt, used for calculate the
// size of temp buffer in post-processing
constexpr const char* reverse_compute_at_extra_var =
    "reverse_compute_at_extra_var";
// record the cooperative process info, used in post schedule
// rule(CooperativeProcess)
constexpr const char* cooperative_process = "cooperative_process";

}  // namespace attr

}  // namespace ir

// Expose the following to cinn namespace for easier usage.
// @{
using ir::Expr;
using ir::Var;
// @}

}  // namespace cinn
