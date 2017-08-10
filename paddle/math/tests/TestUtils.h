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

/**
 * This file provides a AutoCompare calss to simplify the comparison
 * of CPU and GPU member functions.
 *
 * This takes two steps
 * 1. Construct an AutoCompare object.
 *    When constructing an AutoCompare object, you can set the err argument
 * to specify the maximum error for CPU and GPU functions.
 *
 * 2. Use the template functions cmpWithArg or cmpWithoutArg.
 * A. [cmpWithArg] Requires the caller construct the cpu arguments.
 *
 *  AutoCompare test;
 *  Init Argument arg1,arg2...
 *  test.cmpWithArg(function, arg1, arg2....)
 *
 * B. [cmpWithoutArg] The caller do not need construct arguments.
 *    If matrix used in these functions arguments is the same size.
 *    Such as the element wise function and the aggregate function
 *    defined in the BaseMatrix.cpp.
 *
 *  AutoCompare test;
 *  test.cmpWithoutArg<I...>(function, height, width)
 */

#include <gtest/gtest.h>
#include "TensorCheck.h"
#include "paddle/math/Matrix.h"
#include "paddle/math/SparseMatrix.h"

namespace autotest {

using paddle::BaseMatrix;
using paddle::CpuMatrix;
using paddle::GpuMatrix;
using paddle::CpuIVector;
using paddle::GpuIVector;
using paddle::CpuSparseMatrix;
using paddle::GpuSparseMatrix;

template <typename T1, typename T2>
class ReplaceType {
public:
  typedef T1 type;
};

template <>
class ReplaceType<BaseMatrix, CpuMatrix> {
public:
  typedef CpuMatrix type;
};

template <>
class ReplaceType<BaseMatrix, GpuMatrix> {
public:
  typedef GpuMatrix type;
};

template <>
class ReplaceType<Matrix, CpuMatrix> {
public:
  typedef CpuMatrix type;
};

template <>
class ReplaceType<Matrix, GpuMatrix> {
public:
  typedef GpuMatrix type;
};

// construct a argument
template <typename T>
T construct(int height, int width);

template <>
float construct(int height, int width) {
  return 0.5;
}

template <>
double construct(int height, int width) {
  return 0.5;
}

template <>
size_t construct(int height, int width) {
  size_t offset = std::rand() % (height < width ? height : width);
  return offset;
}

template <>
CpuMatrix construct(int height, int width) {
  CpuMatrix a(height, width);
  return a;
}

template <>
GpuMatrix construct(int height, int width) {
  GpuMatrix a(height, width);
  return a;
}

// init a argument
template <typename T>
void init(T& v) {
  return;
}

template <>
void init(CpuMatrix& v) {
  v.randomizeUniform();
}

template <>
void init(GpuMatrix& v) {
  v.randomizeUniform();
}

// init a tuple which contains a set of arguments.
template <std::size_t I = 0, typename... Args>
inline typename std::enable_if<I == sizeof...(Args), void>::type initTuple(
    std::tuple<Args...>& t) {}

template <std::size_t I = 0, typename... Args>
    inline typename std::enable_if <
    I<sizeof...(Args), void>::type initTuple(std::tuple<Args...>& t) {
  init(std::get<I>(t));
  initTuple<I + 1>(t);
}

// copy a argument, copy src to dest
template <typename T1, typename T2>
void copy(T1& dest, T2& src) {
  dest = src;
}

template <>
void copy(GpuMatrix& dest, CpuMatrix& src) {
  dest.copyFrom(src);
}

// copy a tuple, copy src to dest
template <std::size_t I = 0, typename... Args1, typename... Args2>
inline typename std::enable_if<I == sizeof...(Args1), void>::type copyTuple(
    std::tuple<Args1...>& dest, std::tuple<Args2...>& src) {}

template <std::size_t I = 0, typename... Args1, typename... Args2>
    inline typename std::enable_if <
    I<sizeof...(Args1), void>::type copyTuple(std::tuple<Args1...>& dest,
                                              std::tuple<Args2...>& src) {
  copy(std::get<I>(dest), std::get<I>(src));
  copyTuple<I + 1>(dest, src);
}

// call member function
template <typename C,
          typename FC,
          typename R,
          typename... FArgs,
          typename... Args>
R call(C& obj, R (FC::*f)(FArgs...), Args&&... args) {
  return (obj.*f)(args...);
}

template <typename T>
class ReturnType {
public:
  typedef T type;
};

template <>
class ReturnType<CpuMatrix> {
public:
  typedef GpuMatrix type;
};

template <>
class ReturnType<CpuIVector> {
public:
  typedef GpuIVector type;
};

template <>
class ReturnType<CpuSparseMatrix> {
public:
  typedef GpuSparseMatrix type;
};

template <typename T>
typename ReturnType<T>::type autoArgs(T& v) {
  return v;
}

template <>
GpuMatrix autoArgs(CpuMatrix& v) {
  GpuMatrix a(v.getHeight(), v.getWidth());
  a.copyFrom(v);
  return a;
}

template <>
GpuIVector autoArgs(CpuIVector& v) {
  GpuIVector a(v.getSize());
  a.copyFrom(v);
  return a;
}

template <>
GpuSparseMatrix autoArgs(CpuSparseMatrix& v) {
  GpuSparseMatrix a(v.getHeight(),
                    v.getWidth(),
                    v.getElementCnt(),
                    v.getValueType(),
                    v.getFormat());
  a.copyFrom(v, HPPL_STREAM_DEFAULT);
  hl_stream_synchronize(HPPL_STREAM_DEFAULT);
  return a;
}

class AutoCompare {
public:
  /**
   * err is the allowed calculation error.
   * The smaller the value of err,
   * the stricter the comparison is between CPU and GPU calculations.
   */
  AutoCompare(size_t height, size_t width, real err = 1e-3)
      : cpu(height, width), gpu(height, width), compare(err) {
    init(cpu);
    copy(gpu, cpu);
  }

  template <typename C, typename R, typename... FArgs, typename... Args>
  void cmpWithArg(R (C::*f)(FArgs...), Args&&... args) {
    static_assert(sizeof...(FArgs) == sizeof...(Args),
                  "size of parameter packs are not equal");
    call(cpu, f, args...);
    call(gpu, f, autoArgs(args)...);

    TensorCheck(compare, cpu, gpu);
  }

  template <std::size_t... I, typename C, typename R, typename... Args>
  void cmpWithoutArg(R (C::*f)(Args...), size_t height, size_t width) {
    static_assert(sizeof...(I) == sizeof...(Args),
                  "size of parameter packs are not equal");
    (void)height;
    (void)width;
    auto tuple1 = std::make_tuple(
        construct<typename ReplaceType<
            typename std::decay<
                typename std::tuple_element<I,
                                            std::tuple<Args...>>::type>::type,
            CpuMatrix>::type>(height, width)...);

    auto tuple2 = std::make_tuple(
        construct<typename ReplaceType<
            typename std::decay<
                typename std::tuple_element<I,
                                            std::tuple<Args...>>::type>::type,
            GpuMatrix>::type>(height, width)...);

    initTuple(tuple1);
    copyTuple(tuple2, tuple1);

    call(cpu, f, std::get<I>(tuple1)...);
    call(gpu, f, std::get<I>(tuple2)...);

    TensorCheck(compare, cpu, gpu);
  }

protected:
  CpuMatrix cpu;
  GpuMatrix gpu;
  AssertEqual compare;
};

}  // namespace autotest
