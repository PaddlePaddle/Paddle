/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <stdint.h>
#include <cstddef>
#include "hl_tensor_ops.h"
#include "paddle/utils/Common.h"
#include "paddle/utils/Logging.h"

namespace paddle {

template <class OP, typename ExprType, class T>
class TensorConstant;
template <class OP, typename ExprType, class T>
class TensorUnaryOp;
template <class OP, typename LhsType, typename RhsType, class T>
class TensorBinaryOp;
template <typename ExprType1, typename ExprType2, typename ExprType3, class T>
class TensorTernaryOp;

template <typename LhsType, typename RhsType, class T>
class TensorAssignOp;

/**
 * \brief Tensor base class.
 *
 * This is the base class of all Tensor and Expression class.
 */
template <typename Derived, class T>
class TensorExpression {
public:
  /**
   * Element wise unary expression.
   */
  template <typename UnaryOp>
  const TensorUnaryOp<UnaryOp, const Derived, T> unaryExpression(
      const UnaryOp& op) const {
    return TensorUnaryOp<UnaryOp, const Derived, T>(op, derived());
  }

  const TensorUnaryOp<hppl::unary::add_scale<T>, const Derived, T> operator+(
      T p) const {
    return unaryExpression(hppl::unary::add_scale<T>(p));
  }

  const TensorUnaryOp<hppl::unary::sub_scale<T>, const Derived, T> operator-(
      T p) const {
    return unaryExpression(hppl::unary::sub_scale<T>(p));
  }

  const TensorUnaryOp<hppl::unary::mul_scale<T>, const Derived, T> operator*(
      T p) const {
    return unaryExpression(hppl::unary::mul_scale<T>(p));
  }

  const TensorUnaryOp<hppl::unary::div_scale<T>, const Derived, T> operator/(
      T p) const {
    return unaryExpression(hppl::unary::div_scale<T>(p));
  }

  const TensorUnaryOp<hppl::unary::neg<T>, const Derived, T> operator-() const {
    return unaryExpression(hppl::unary::neg<T>());
  }

  const TensorUnaryOp<hppl::unary::exp_op<T>, const Derived, T> exp() const {
    return unaryExpression(hppl::unary::exp_op<T>());
  }

  const TensorUnaryOp<hppl::unary::log_op<T>, const Derived, T> log() const {
    return unaryExpression(hppl::unary::log_op<T>());
  }

  const TensorUnaryOp<hppl::unary::sqrt_op<T>, const Derived, T> sqrt() const {
    return unaryExpression(hppl::unary::sqrt_op<T>());
  }

  const TensorUnaryOp<hppl::unary::square<T>, const Derived, T> square() const {
    return unaryExpression(hppl::unary::square<T>());
  }

  const TensorUnaryOp<hppl::unary::reciprocal<T>, const Derived, T> reciprocal()
      const {
    return unaryExpression(hppl::unary::reciprocal<T>());
  }

  const TensorUnaryOp<hppl::unary::abs<T>, const Derived, T> abs() const {
    return unaryExpression(hppl::unary::abs<T>());
  }

  const TensorUnaryOp<hppl::unary::sign<T>, const Derived, T> sign() const {
    return unaryExpression(hppl::unary::sign<T>());
  }

  const TensorUnaryOp<hppl::unary::pow_op<T>, const Derived, T> pow(T p) const {
    return unaryExpression(hppl::unary::pow_op<T>(p));
  }

  const TensorUnaryOp<hppl::unary::min<T>, const Derived, T> min(T p) const {
    return unaryExpression(hppl::unary::min<T>(p));
  }

  const TensorUnaryOp<hppl::unary::max<T>, const Derived, T> max(T p) const {
    return unaryExpression(hppl::unary::max<T>(p));
  }

  const TensorUnaryOp<hppl::unary::cmp_eq<T>, const Derived, T> operator==(
      T p) const {
    return unaryExpression(hppl::unary::cmp_eq<T>(p));
  }

  const TensorUnaryOp<hppl::unary::cmp_ne<T>, const Derived, T> operator!=(
      T p) const {
    return unaryExpression(hppl::unary::cmp_ne<T>(p));
  }

  const TensorUnaryOp<hppl::unary::cmp_le<T>, const Derived, T> operator<=(
      T p) const {
    return unaryExpression(hppl::unary::cmp_le<T>(p));
  }

  const TensorUnaryOp<hppl::unary::cmp_lt<T>, const Derived, T> operator<(
      T p) const {
    return unaryExpression(hppl::unary::cmp_lt<T>(p));
  }

  const TensorUnaryOp<hppl::unary::cmp_ge<T>, const Derived, T> operator>=(
      T p) const {
    return unaryExpression(hppl::unary::cmp_ge<T>(p));
  }

  const TensorUnaryOp<hppl::unary::cmp_gt<T>, const Derived, T> operator>(
      T p) const {
    return unaryExpression(hppl::unary::cmp_gt<T>(p));
  }

  const TensorUnaryOp<hppl::unary::and_op<T>, const Derived, T> operator&&(
      T p) const {
    return unaryExpression(hppl::unary::and_op<T>(p));
  }

  const TensorUnaryOp<hppl::unary::or_op<T>, const Derived, T> operator||(
      T p) const {
    return unaryExpression(hppl::unary::or_op<T>(p));
  }

  /**
   * Element wise binary expression.
   */
  template <typename BinaryOp, typename ExpressionType>
  const TensorBinaryOp<BinaryOp, const Derived, const ExpressionType, T>
  binaryExpression(const BinaryOp& op, const ExpressionType& expr) const {
    return TensorBinaryOp<BinaryOp, const Derived, const ExpressionType, T>(
        op, derived(), expr);
  }

  template <typename ExpressionType>
  const TensorBinaryOp<hppl::binary::cmp_eq<T>,
                       const Derived,
                       const ExpressionType,
                       T>
  operator==(const ExpressionType& expr) const {
    return binaryExpression(hppl::binary::cmp_eq<T>(), expr);
  }

  template <typename ExpressionType>
  const TensorBinaryOp<hppl::binary::cmp_ne<T>,
                       const Derived,
                       const ExpressionType,
                       T>
  operator!=(const ExpressionType& expr) const {
    return binaryExpression(hppl::binary::cmp_ne<T>(), expr);
  }

  template <typename ExpressionType>
  const TensorBinaryOp<hppl::binary::cmp_le<T>,
                       const Derived,
                       const ExpressionType,
                       T>
  operator<=(const ExpressionType& expr) const {
    return binaryExpression(hppl::binary::cmp_le<T>(), expr);
  }

  template <typename ExpressionType>
  const TensorBinaryOp<hppl::binary::cmp_lt<T>,
                       const Derived,
                       const ExpressionType,
                       T>
  operator<(const ExpressionType& expr) const {
    return binaryExpression(hppl::binary::cmp_lt<T>(), expr);
  }

  template <typename ExpressionType>
  const TensorBinaryOp<hppl::binary::cmp_ge<T>,
                       const Derived,
                       const ExpressionType,
                       T>
  operator>=(const ExpressionType& expr) const {
    return binaryExpression(hppl::binary::cmp_ge<T>(), expr);
  }

  template <typename ExpressionType>
  const TensorBinaryOp<hppl::binary::cmp_gt<T>,
                       const Derived,
                       const ExpressionType,
                       T>
  operator>(const ExpressionType& expr) const {
    return binaryExpression(hppl::binary::cmp_gt<T>(), expr);
  }

  template <typename ExpressionType>
  const TensorBinaryOp<hppl::binary::and_op<T>,
                       const Derived,
                       const ExpressionType,
                       T>
  operator&&(const ExpressionType& expr) const {
    return binaryExpression(hppl::binary::and_op<T>(), expr);
  }

  template <typename ExpressionType>
  const TensorBinaryOp<hppl::binary::or_op<T>,
                       const Derived,
                       const ExpressionType,
                       T>
  operator||(const ExpressionType& expr) const {
    return binaryExpression(hppl::binary::or_op<T>(), expr);
  }

  template <typename ExpressionType>
  const TensorBinaryOp<hppl::binary::add<T>,
                       const Derived,
                       const ExpressionType,
                       T>
  operator+(const ExpressionType& expr) const {
    return binaryExpression(hppl::binary::add<T>(), expr);
  }

  template <typename ExpressionType>
  const TensorBinaryOp<hppl::binary::sub<T>,
                       const Derived,
                       const ExpressionType,
                       T>
  operator-(const ExpressionType& expr) const {
    return binaryExpression(hppl::binary::sub<T>(), expr);
  }

  template <typename ExpressionType>
  const TensorBinaryOp<hppl::binary::mul<T>,
                       const Derived,
                       const ExpressionType,
                       T>
  operator*(const ExpressionType& expr) const {
    return binaryExpression(hppl::binary::mul<T>(), expr);
  }

  template <typename ExpressionType>
  const TensorBinaryOp<hppl::binary::div<T>,
                       const Derived,
                       const ExpressionType,
                       T>
  operator/(const ExpressionType& expr) const {
    return binaryExpression(hppl::binary::div<T>(), expr);
  }

  template <typename ExpressionType>
  const TensorBinaryOp<hppl::binary::min<T>,
                       const Derived,
                       const ExpressionType,
                       T>
  min(const ExpressionType& expr) const {
    return binaryExpression(hppl::binary::min<T>(), expr);
  }

  template <typename ExpressionType>
  const TensorBinaryOp<hppl::binary::max<T>,
                       const Derived,
                       const ExpressionType,
                       T>
  max(const ExpressionType& expr) const {
    return binaryExpression(hppl::binary::max<T>(), expr);
  }

  /**
   * Element wise ternary expression.
   *
   * ternary conditional operator(?: operator).
   * The conditional expression returns one of two values depending on
   * the result of derived expression.
   * If derived expression evaluates to true, then expression1 is evaluated.
   * If derived expression evaluates to false, then expression2 is evaluated.
   */
  template <typename ExprType1, typename ExprType2>
  const TensorTernaryOp<const Derived, const ExprType1, const ExprType2, T>
  condition(const ExprType1& expr1, const ExprType2& expr2) const {
    return TensorTernaryOp<const Derived, const ExprType1, const ExprType2, T>(
        derived(), expr1, expr2);
  }

  template <typename ExprType>
  const TensorTernaryOp<
      const Derived,
      const TensorConstant<hppl::unary::constant<T>, const Derived, T>,
      const ExprType,
      T>
  condition(T p, const ExprType& expr) const {
    return condition(constant(p), expr);
  }

  template <typename ExprType>
  const TensorTernaryOp<
      const Derived,
      const ExprType,
      const TensorConstant<hppl::unary::constant<T>, const Derived, T>,
      T>
  condition(const ExprType& expr, T p) const {
    return condition(expr, constant(p));
  }

  const TensorTernaryOp<
      const Derived,
      const TensorConstant<hppl::unary::constant<T>, const Derived, T>,
      const TensorConstant<hppl::unary::constant<T>, const Derived, T>,
      T>
  condition(T p1, T p2) const {
    return condition(constant(p1), constant(p2));
  }

  /**
   * return a TensorConstant. A TensorConstant object hold a constant value.
   */
  const TensorConstant<hppl::unary::constant<T>, const Derived, T> constant(
      T p) const {
    return TensorConstant<hppl::unary::constant<T>, const Derived, T>(
        hppl::unary::constant<T>(p), derived());
  }

  /**
   * return a TensorAssignOp, and use AssignEvaluate to evaluate one or more
   * TensorAssignOp objects.
   */
  template <typename ExpressionType>
  TensorAssignOp<Derived, ExpressionType, T> lazyAssign(
      const ExpressionType& expr) const {
    return TensorAssignOp<Derived, ExpressionType, T>(derived(), expr);
  }

protected:
  const Derived& derived() const { return *static_cast<const Derived*>(this); }
};

/**
 * \brief Unary Operator Expression
 */
template <class OP, typename ExprType, class T>
class TensorUnaryOp
    : public TensorExpression<TensorUnaryOp<OP, ExprType, T>, T> {
public:
  explicit TensorUnaryOp(const OP op, const ExprType& expr)
      : op_(op), expr_(expr) {}

  const OP op_;
  const ExprType expr_;
};

/**
 * \brief Binary Operator Expression
 */
template <class OP, typename LhsType, typename RhsType, class T>
class TensorBinaryOp
    : public TensorExpression<TensorBinaryOp<OP, LhsType, RhsType, T>, T> {
public:
  explicit TensorBinaryOp(const OP op, const LhsType& lhs, const RhsType& rhs)
      : op_(op), lhs_(lhs), rhs_(rhs) {}

  const OP op_;
  const LhsType lhs_;
  const RhsType rhs_;
};

/**
 * \brief Ternary Operator Expression
 */
template <typename ExprType1, typename ExprType2, typename ExprType3, class T>
class TensorTernaryOp : public TensorExpression<
                            TensorTernaryOp<ExprType1, ExprType2, ExprType3, T>,
                            T> {
public:
  explicit TensorTernaryOp(const ExprType1& expr1,
                           const ExprType2& expr2,
                           const ExprType3& expr3)
      : expr1_(expr1), expr2_(expr2), expr3_(expr3) {}

  const ExprType1 expr1_;
  const ExprType2 expr2_;
  const ExprType3 expr3_;
};

/**
 * \brief Constant Expression
 */
template <class OP, typename ExprType, class T>
class TensorConstant
    : public TensorExpression<TensorConstant<OP, ExprType, T>, T> {
public:
  explicit TensorConstant(const OP op, const ExprType& expr)
      : op_(op), expr_(expr) {}

  const OP op_;
  const ExprType expr_;
};

/**
 * \brief operator+ overload
 * \return a unary operator expression
 */
template <typename Derived, class T>
const TensorUnaryOp<hppl::unary::add_scale<T>, const Derived, T> operator+(
    T p, const TensorExpression<Derived, T>& expr) {
  return expr + p;
}

/**
 * \brief operator* overload
 * \return a unary operator expression
 */
template <typename Derived, class T>
const TensorUnaryOp<hppl::unary::mul_scale<T>, const Derived, T> operator*(
    T p, const TensorExpression<Derived, T>& expr) {
  return expr * p;
}

}  // namespace paddle

#include "TensorApply.h"
#include "TensorEvaluate.h"
