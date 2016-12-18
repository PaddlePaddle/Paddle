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

namespace paddle {

/**
 * \brief The tensor evaluator classes.
 */
template <typename Derived, class T>
class TensorApply {
public:
  explicit INLINE TensorApply(const Derived& p)
      : data_(p.data_),
        stride_(p.stride_),
        height_(p.height_),
        width_(p.width_),
        useGpu_(p.useGpu_) {}

  INLINE T apply(int i, int j) const { return data_[i * stride_ + j]; }
  INLINE T apply(int index) const { return data_[index]; }
  INLINE T& applyRef(int i, int j) { return data_[i * stride_ + j]; }
  INLINE T& applyRef(int index) { return data_[index]; }

  INLINE size_t getWidth() const { return width_; }
  INLINE size_t getHeight() const { return height_; }
  INLINE bool isContiguous() const { return stride_ == width_ || height_ == 1; }
  INLINE bool useGpu() const { return useGpu_; }

  T* data_;
  size_t stride_;
  size_t height_;
  size_t width_;
  bool useGpu_;
};

/**
 * \brief The tensor evaluator classes.
 * evaluator for rvalues
 */
template <typename Derived, class T>
class TensorApply<const Derived, T> {
public:
  explicit INLINE TensorApply(const Derived& p)
      : data_(p.data_),
        stride_(p.stride_),
        height_(p.height_),
        width_(p.width_),
        useGpu_(p.useGpu_) {}

  INLINE T apply(int i, int j) const { return data_[i * stride_ + j]; }
  INLINE T apply(int index) const { return data_[index]; }

  INLINE size_t getWidth() const { return width_; }
  INLINE size_t getHeight() const { return height_; }
  INLINE bool isContiguous() const { return stride_ == width_ || height_ == 1; }
  INLINE bool useGpu() const { return useGpu_; }

  const T* data_;
  size_t stride_;
  size_t height_;
  size_t width_;
  bool useGpu_;
};

template <typename Derived, class T>
class TensorApply<const TensorExpression<Derived, T>, T> {
public:
  explicit TensorApply(const TensorExpression<Derived, T>& expr)
      : expr_(expr.derived()) {}

  INLINE T apply(int i, int j) const { return expr_.apply(i, j); }
  INLINE T apply(int index) const { return expr_.apply(index); }

  INLINE size_t getWidth() const { return expr_.getWidth(); }
  INLINE size_t getHeight() const { return expr_.getHeight(); }
  INLINE bool isContiguous() const { return expr_.isContiguous(); }
  INLINE bool useGpu() const { return expr_.useGpu(); }

  TensorApply<const Derived, T> expr_;
};

/**
 * \brief The unary expression evaluator classes.
 */
template <class OP, typename ArgType, class T>
class TensorApply<const TensorUnaryOp<OP, ArgType, T>, T> {
public:
  explicit INLINE TensorApply(const TensorUnaryOp<OP, ArgType, T>& expr)
      : op_(expr.op_), expr_(expr.expr_) {}

  INLINE T apply(int i, int j) const { return op_(expr_.apply(i, j)); }
  INLINE T apply(int index) const { return op_(expr_.apply(index)); }

  INLINE size_t getWidth() const { return expr_.getWidth(); }
  INLINE size_t getHeight() const { return expr_.getHeight(); }
  INLINE bool isContiguous() const { return expr_.isContiguous(); }
  INLINE bool useGpu() const { return expr_.useGpu(); }

  const OP op_;
  TensorApply<ArgType, T> expr_;
};

/**
 * \brief The binary expression evaluator classes.
 */
template <class OP, typename LhsType, typename RhsType, class T>
class TensorApply<const TensorBinaryOp<OP, LhsType, RhsType, T>, T> {
public:
  explicit INLINE TensorApply(
      const TensorBinaryOp<OP, LhsType, RhsType, T>& expr)
      : op_(expr.op_), lhs_(expr.lhs_), rhs_(expr.rhs_) {
#ifndef __CUDA_ARCH__
    CHECK_EQ(lhs_.getWidth(), rhs_.getWidth());
    CHECK_EQ(lhs_.getHeight(), rhs_.getHeight());
    CHECK_EQ(lhs_.useGpu(), rhs_.useGpu());
#endif
  }

  INLINE T apply(int i, int j) const {
    return op_(lhs_.apply(i, j), rhs_.apply(i, j));
  }
  INLINE T apply(int index) const {
    return op_(lhs_.apply(index), rhs_.apply(index));
  }

  INLINE size_t getWidth() const { return lhs_.getWidth(); }
  INLINE size_t getHeight() const { return rhs_.getHeight(); }
  INLINE bool isContiguous() const {
    return lhs_.isContiguous() && rhs_.isContiguous();
  }
  INLINE bool useGpu() const { return lhs_.useGpu(); }

  const OP op_;
  TensorApply<LhsType, T> lhs_;
  TensorApply<RhsType, T> rhs_;
};

/**
 * \brief The ternary expression evaluator classes.
 */
template <typename ArgType1, typename ArgType2, typename ArgType3, class T>
class TensorApply<const TensorTernaryOp<ArgType1, ArgType2, ArgType3, T>, T> {
public:
  explicit INLINE TensorApply(
      const TensorTernaryOp<ArgType1, ArgType2, ArgType3, T>& expr)
      : expr1_(expr.expr1_), expr2_(expr.expr2_), expr3_(expr.expr3_) {
#ifndef __CUDA_ARCH__
    CHECK_EQ(expr1_.getWidth(), expr2_.getWidth());
    CHECK_EQ(expr1_.getWidth(), expr3_.getWidth());
    CHECK_EQ(expr1_.getHeight(), expr2_.getHeight());
    CHECK_EQ(expr1_.getHeight(), expr3_.getHeight());
    CHECK_EQ(expr1_.useGpu(), expr2_.useGpu());
    CHECK_EQ(expr1_.useGpu(), expr3_.useGpu());
#endif
  }

  INLINE T apply(int i, int j) const {
    return expr1_.apply(i, j) ? expr2_.apply(i, j) : expr3_.apply(i, j);
  }
  INLINE T apply(int index) const {
    return expr1_.apply(index) ? expr2_.apply(index) : expr3_.apply(index);
  }

  INLINE size_t getWidth() const { return expr1_.getWidth(); }
  INLINE size_t getHeight() const { return expr1_.getHeight(); }
  INLINE bool isContiguous() const {
    return expr1_.isContiguous() && expr2_.isContiguous() &&
           expr3_.isContiguous();
  }
  INLINE bool useGpu() const { return expr1_.useGpu(); }

  TensorApply<ArgType1, T> expr1_;
  TensorApply<ArgType2, T> expr2_;
  TensorApply<ArgType3, T> expr3_;
};

/**
 * \brief The const expression evaluator classes.
 */
template <class OP, typename ArgType, class T>
class TensorApply<const TensorConstant<OP, ArgType, T>, T> {
public:
  explicit INLINE TensorApply(const TensorConstant<OP, ArgType, T>& expr)
      : op_(expr.op_), expr_(expr.expr_) {}

  INLINE T apply(int i, int j) const { return op_(i, j); }
  INLINE T apply(int index) const { return op_(index); }

  INLINE size_t getWidth() const { return expr_.getWidth(); }
  INLINE size_t getHeight() const { return expr_.getHeight(); }
  INLINE bool isContiguous() const { return true; }
  INLINE bool useGpu() const { return expr_.useGpu(); }

  const OP op_;
  TensorApply<ArgType, T> expr_;
};

}  // namespace paddle
