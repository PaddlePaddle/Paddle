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

#pragma once

#include <glog/logging.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "paddle/cinn/common/common.h"
#include "paddle/cinn/common/object.h"
#include "paddle/cinn/common/shared.h"
#include "paddle/cinn/common/type.h"

namespace cinn {

namespace ir {
using cinn::common::BFloat16;
using cinn::common::Float;
using cinn::common::Float16;
using cinn::common::Int;
using cinn::common::Type;
using cinn::common::type_of;

class Module;
class IRVisitor;
class _Buffer_;
class Buffer;
class _Module_;
class _LoweredFunc_;
class LoweredFunc;
class _Tensor_;
class Tensor;
class _Var_;
class Var;
class _BufferRange_;
class BufferRange;
class ScheduleBlock;
class ScheduleBlockRealize;
class Dim;

// clang-format off
#define NODETY_PRIMITIVE_TYPE_FOR_EACH(macro__) \
  macro__(IntImm)                               \
  macro__(UIntImm)                              \
  macro__(FloatImm)                             \
  macro__(StringImm)                            \

#define NODETY_BINARY_OP_FOR_EACH(macro__) \
  macro__(Add)                      \
  macro__(Sub)                      \
  macro__(Mul)                      \
  macro__(Div)                      \
  macro__(Mod)                      \
  macro__(EQ)                       \
  macro__(NE)                       \
  macro__(LT)                       \
  macro__(LE)                       \
  macro__(GT)                       \
  macro__(GE)                       \
  macro__(And)                      \
  macro__(Or)                       \
  macro__(Min)                      \
  macro__(Max)                      \

#define NODETY_UNARY_OP_FOR_EACH(macro__) \
  macro__(Minus)                          \
  macro__(Not)                            \

#define NODETY_OP_FOR_EACH(macro__)  \
  NODETY_BINARY_OP_FOR_EACH(macro__) \
  NODETY_UNARY_OP_FOR_EACH(macro__)

#define NODETY_CONTROL_OP_FOR_EACH(macro__) \
  macro__(Cast)                             \
  macro__(For)                              \
  macro__(PolyFor)                          \
  macro__(Select)                           \
  macro__(IfThenElse)                       \
  macro__(Block)                            \
  macro__(Call)                             \
  macro__(_Var_)                            \
  macro__(Load)                             \
  macro__(Store)                            \
  macro__(Alloc)                            \
  macro__(Free)                             \
  macro__(_Buffer_)                         \
  macro__(_Tensor_)                         \
  macro__(_LoweredFunc_)                    \
  macro__(_Module_)                         \
  macro__(Let)                              \
  macro__(Reduce)                           \
  macro__(Ramp)                             \
  macro__(Broadcast)                        \
  macro__(FracOp)                           \
  macro__(Product)                          \
  macro__(Sum)                              \
  macro__(PrimitiveNode)                    \
  macro__(_BufferRange_)                    \
  macro__(ScheduleBlock)                    \
  macro__(ScheduleBlockRealize)             \
  macro__(_Dim_)                            \

#define NODETY_CONTROL_OP_FOR_INTRINSIC(macro__) \
  macro__(IntrinsicOp)                      \

#define NODETY_FORALL(__m)              \
  NODETY_PRIMITIVE_TYPE_FOR_EACH(__m)   \
  NODETY_OP_FOR_EACH(__m)               \
  NODETY_CONTROL_OP_FOR_INTRINSIC(__m)  \
  NODETY_CONTROL_OP_FOR_EACH(__m)

#define NODETY_FORALL_EXCEPT_INTRINSIC(__m)              \
  NODETY_PRIMITIVE_TYPE_FOR_EACH(__m)                    \
  NODETY_OP_FOR_EACH(__m)                                \
  NODETY_CONTROL_OP_FOR_EACH(__m)
// clang-format on

//! Define IrNodeTy
// @{
#define __m(x__) x__,
enum class IrNodeTy { kUnk = -1, NODETY_FORALL(__m) };
#undef __m
// @}

//! String representations for IrNodeTy.
// @{
#define __m(x__) #x__,
const std::vector<std::string> kIrNodeTyReprs({NODETY_FORALL(__m) "None"});
#undef __m
// @}

std::ostream& operator<<(std::ostream& os, IrNodeTy type);

struct Expr;

/**
 * The base of all the nodes in the IR.
 */
class IrNode : public cinn::common::Object {
 public:
  //! The operands of this operator.
  std::vector<Expr> operands;

  IrNode() = default;
  explicit IrNode(Type t) : type_(t) {}
  virtual ~IrNode() = default;

  virtual IrNodeTy node_type() const { return IrNodeTy::kUnk; }
  virtual Type type() const { return type_; }
  void set_type(Type type);
  //! Elevate int32 to int64 if needed
  void convert_int32_to_int64();

  //! Get i-th operand
  const Expr& operand(int i);

  //! Gather all the expression fields in this node for easier visit and mutate.
  virtual std::vector<Expr*> expr_fields() { return {}; }
  virtual std::vector<const Expr*> expr_fields() const { return {}; }

  const char* type_info() const override { return __type_info__; }

  //! Verify the current IR node's correctness.
  virtual void Verify() const { CINN_NOT_IMPLEMENTED }

 protected:
  static constexpr char* __type_info__ = "IRNode";
  Type type_;
};

/**
 * A handle to store any IRNode.
 */
class IrNodeRef : public cinn::common::Shared<IrNode> {
 public:
  IrNodeRef() = default;
  IrNodeRef(const IrNodeRef& other) : Shared(other.p_) {}
  explicit IrNodeRef(IrNode* x) : Shared(x) {}

  virtual IrNodeTy node_type() const { return operator->()->node_type(); }

  template <typename T>
  const T* As() const {
    static_assert(std::is_base_of<IrNode, T>());
    CHECK(get()) << "IrNodeRef holds null";
    if (node_type() == T::_node_type_) return static_cast<const T*>(get());
    return nullptr;
  }
  template <typename T>
  T* As() {
    if (node_type() == T::_node_type_) return static_cast<T*>(get());
    return nullptr;
  }

  void operator=(const IrNodeRef& other) {
    *static_cast<Shared<IrNode>*>(this) =
        *static_cast<const Shared<IrNode>*>(&other);
  }

  IrNode* ptr() { return get(); }
  IrNode* ptr() const { return get(); }
};

template <typename T>
struct ExprNode : public IrNode {
  ExprNode() : IrNode(Type()) {}
  explicit ExprNode(Type t) : IrNode(t) { set_type(t); }
  explicit ExprNode(int num_operands) { operands().resize(num_operands); }

  T* self() { return static_cast<T*>(this); }
  const T* const_self() const { return dynamic_cast<const T*>(this); }

  const std::vector<Expr>& operands() const { return IrNode::operands; }
  std::vector<Expr>& operands() { return IrNode::operands; }

  Expr& operand(int i) {
    CHECK_LT(i, operands().size());
    return operands()[i];
  }
  const Expr& operand(int i) const {
    CHECK_LT(i, operands().size());
    return operands()[i];
  }

  virtual Expr Copy() const;

  IrNodeTy node_type() const override { return T::_node_type_; }
};

struct IntImm : public ExprNode<IntImm> {
  int64_t value;

  IntImm(Type t, int64_t v) : ExprNode<IntImm>(t), value(v) { Verify(); }

  void Verify() const override {
    CHECK(type().is_int());
    CHECK(type().is_scalar());
    CHECK(type().bits() == 8 || type().bits() == 16 || type().bits() == 32 ||
          type().bits() == 64);
  }

  static const IrNodeTy _node_type_ = IrNodeTy::IntImm;
};

struct UIntImm : public ExprNode<UIntImm> {
  uint64_t value;

  UIntImm(Type t, uint64_t v) : ExprNode<UIntImm>(t), value(v) { Verify(); }

  void Verify() const override {
    CHECK(type().is_uint());
    CHECK(type().is_scalar());
    CHECK(type().bits() == 1 /*bool*/ || type().bits() == 8 ||
          type().bits() == 16 || type().bits() == 32 || type().bits() == 64);
  }

  static const IrNodeTy _node_type_ = IrNodeTy::UIntImm;
};

struct FloatImm : public ExprNode<FloatImm> {
  double value;

  FloatImm(Type t, double v) : ExprNode<FloatImm>(t), value(v) { Verify(); }

  void Verify() const override {
    CHECK(type().is_float());
    CHECK(type().is_scalar());
  }

  static const IrNodeTy _node_type_ = IrNodeTy::FloatImm;
};

struct StringImm : public ExprNode<StringImm> {
  std::string value;

  explicit StringImm(const std::string& value) : value(value) { Verify(); }

  void Verify() const override {}

  static const IrNodeTy _node_type_ = IrNodeTy::StringImm;
};

class Var;
/**
 * An expression that represents some value or the result of some operations.
 */
struct Expr : public IrNodeRef {
 public:
  Expr() = default;
  Expr(const Expr& other) : IrNodeRef(other.ptr()) {}
  Expr(IrNode* p) : IrNodeRef(p) {}  // NOLINT
  explicit Expr(const Var& var);

  //! Helper function to construct numeric constants of various types.
  // @{
  explicit Expr(bool x) : IrNodeRef(new UIntImm(UInt(1), x)) {}

  explicit Expr(int8_t x) : IrNodeRef(new IntImm(Int(8), x)) {}
  explicit Expr(int16_t x) : IrNodeRef(new IntImm(Int(16), x)) {}
  explicit Expr(int32_t x) : IrNodeRef(new IntImm(Int(32), x)) {}
  explicit Expr(int64_t x) : IrNodeRef(new IntImm(Int(64), x)) {}

  explicit Expr(uint8_t x) : IrNodeRef(new UIntImm(UInt(8), x)) {}
  explicit Expr(uint16_t x) : IrNodeRef(new UIntImm(UInt(16), x)) {}
  explicit Expr(uint32_t x) : IrNodeRef(new UIntImm(UInt(32), x)) {}
  explicit Expr(uint64_t x) : IrNodeRef(new UIntImm(UInt(64), x)) {}

  explicit Expr(cinn::common::bfloat16 x)
      : IrNodeRef(new FloatImm(BFloat16(), x)) {}
  explicit Expr(cinn::common::float16 x)
      : IrNodeRef(new FloatImm(Float16(), x)) {}
  explicit Expr(float x) : IrNodeRef(new FloatImm(Float(32), x)) {}
  explicit Expr(double x) : IrNodeRef(new FloatImm(Float(64), x)) {}

  explicit Expr(const std::string& x) : IrNodeRef(new StringImm(x)) {}
  // @}

  Expr& operator=(const Expr& other);

  // primitive types
  // @{
  bool as_bool() const;

  int8_t as_int8() const;
  int16_t as_int16() const;
  int32_t as_int32() const;
  int64_t as_int64() const;

  uint8_t as_uint8() const;
  uint16_t as_uint16() const;
  uint32_t as_uint32() const;
  uint64_t as_uint64() const;

  cinn::common::bfloat16 as_bfloat16() const;
  cinn::common::float16 as_float16() const;
  float as_float() const;
  double as_double() const;
  // @}

  _Var_* as_var();
  const _Var_* as_var() const;
  Var as_var_ref() const;

  // @{ Other nodes caster.
  _Buffer_* as_buffer();
  const _Buffer_* as_buffer() const;
  Buffer as_buffer_ref() const;

  _LoweredFunc_* as_lowered_func();
  const _LoweredFunc_* as_lowered_func() const;
  LoweredFunc as_lowered_func_ref() const;

  _Module_* as_module();
  const _Module_* as_module() const;
  ir::Module as_module_ref() const;

  _Tensor_* as_tensor();
  const _Tensor_* as_tensor() const;
  ir::Tensor as_tensor_ref() const;
  // @}

  bool is_constant() const;
  double get_constant() const;

  //! Tell if this is a compare op.
  bool is_cmp() const;

  bool is_var() const;

  operator Var();

  Type type() const { return p_->type(); }
};

template <typename T>
struct UnaryOpNode : public ExprNode<T> {
  UnaryOpNode() { operands().resize(1); }
  UnaryOpNode(Type type, Expr v) : ExprNode<T>(type) {
    CHECK(v.defined());
    operands().resize(1);
    this->v() = v;
  }

  Type type() const override {
    CHECK(v().defined());
    return v().type();
  }

  Expr& v() { return operands().front(); }
  const Expr& v() const { return operands().front(); }

  std::vector<Expr*> expr_fields() override { return {&v()}; }
  std::vector<const Expr*> expr_fields() const override { return {&v()}; }

  using ExprNode<T>::operands;
};

template <typename T>
struct BinaryOpNode : public ExprNode<T> {
  BinaryOpNode() { operands().resize(2); }
  BinaryOpNode(Type type, Expr a, Expr b) : ExprNode<T>(type) {
    CHECK(type.valid());
    CHECK(a.defined());
    CHECK(b.defined());
    operands().resize(2);
    this->a() = a;
    this->b() = b;
    // CHECK_EQ(a.type(), b.type()) << "the type of two argument not match";
  }

  Expr& a() { return ExprNode<T>::operand(0); }
  Expr& b() { return ExprNode<T>::operand(1); }
  const Expr& a() const { return ExprNode<T>::operand(0); }
  const Expr& b() const { return ExprNode<T>::operand(1); }

  Type type() const override { return a().type(); }

  std::vector<Expr*> expr_fields() override { return {&a(), &b()}; }
  std::vector<const Expr*> expr_fields() const override { return {&a(), &b()}; }

  using ExprNode<T>::operands;
};

//! Zero in CINN type system.
Expr Zero(const Type& type);
Expr One(const Type& type);

#define DEVICE_API_FOR_ALL(__) \
  __(UNK)                      \
  __(Host)                     \
  __(GPU)                      \
  __(CUDA)                     \
  __(OpenCL)

#define __decl__(x) x,
enum class DeviceAPI { DEVICE_API_FOR_ALL(__decl__) };
#undef __decl__

static std::ostream& operator<<(std::ostream& os, DeviceAPI x) {
  switch (x) {
#define __decl__(x)  \
  case DeviceAPI::x: \
    os << #x;        \
    break;

    DEVICE_API_FOR_ALL(__decl__)
#undef __decl__

    default:
      break;
  }
  return os;
}

#define MEMORY_TYPE_FOR_ALL(__)                                                \
  __(Auto, "Auto")                                                             \
  __(Heap, "Heap")                                                             \
  __(Stack, "Stack")                                                           \
  __(GPUShared, "GPUShared")                                                   \
  __(GPULocal, "GPULocal")                                                     \
/**                                                                            \
 * An enum describing different address spaces to be used with Func::store_in. \
 */
enum class MemoryType {
#define __(token__, token_repr__) token__,
  MEMORY_TYPE_FOR_ALL(__)
#undef __
};

static std::ostream& operator<<(std::ostream& os, MemoryType t) {
  switch (t) {
#define __(token__, token_repr__) \
  case MemoryType::token__:       \
    os << token_repr__;           \
    break;

    MEMORY_TYPE_FOR_ALL(__)

    default:
      LOG(FATAL) << "Not supported memory type";
#undef __
  }
  return os;
}

template <typename T>
Expr ExprNode<T>::Copy() const {
  LOG(FATAL) << "Not Implemented";
  return Expr();
}

void TryElevateInt32ToInt64(const std::vector<Expr>& expr_vec);

}  // namespace ir
}  // namespace cinn

namespace std {

template <>
struct hash<cinn::ir::Expr> {
  size_t operator()(const cinn::ir::Expr& x) {
    return reinterpret_cast<size_t>(x.get());
  }
};

}  // namespace std
