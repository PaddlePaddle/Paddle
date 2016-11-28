/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */


/**
 * TestUtils.h is used to automatically compare CPU and GPU code is consistent.
 *
 * Auto compare BaseMatrix member function:
 * Use case:
 * a. void BaseMatrix::tanh(BaseMatrixT& b);
 *   Compare method: BaseMatrixCompare<0>(&BaseMatrix::tanh);
 *
 * b. 
 *
*/

#include <gtest/gtest.h>
#include "paddle/math/Matrix.h"
#include "TensorCheck.h"

using namespace paddle;  // NOLINT

namespace autotest {

template<typename T1, typename T2>
class ReplaceType {
public:
  typedef T1 type;
};

template<>
class ReplaceType<BaseMatrix, CpuMatrix> {
public:
  typedef CpuMatrix type;
};

template<>
class ReplaceType<BaseMatrix, GpuMatrix> {
public:
  typedef GpuMatrix type;
};

// construct a argument
template<typename T> T construct(int height, int width);
template<> float construct(int height, int width) { return 0.0; }
template<> CpuMatrix construct(int height, int width) {
  CpuMatrix a(height, width);
  return a;
}
template<> GpuMatrix construct(int height, int width) {
  GpuMatrix a(height, width);
  return a;
}

// init a argument
template<typename T> void init(T& v);
template<> void init(float& v) { v = 0.5; }
template<> void init(CpuMatrix& v) { v.randomizeUniform(); }
template<> void init(GpuMatrix& v) { v.randomizeUniform(); }

// init a tuple which contains a set of arguments.
template<std::size_t I = 0, typename... Args>
inline typename std::enable_if<I == sizeof...(Args), void>::type
initTuple(std::tuple<Args...>& t){}

template<std::size_t I = 0, typename... Args>
inline typename std::enable_if<I < sizeof...(Args), void>::type
initTuple(std::tuple<Args...>& t) {
  init(std::get<I>(t));
  initTuple<I + 1>(t);
}

// copy a argument, copy src to dest
template<typename T1, typename T2> void copy(T1& dest, T2& src);
template<> void copy(float& dest, float& src) { dest = src; }
template<> void copy(GpuMatrix& dest, CpuMatrix& src) {
  dest.copyFrom(src);
}

// copy a tuple, copy src to dest
template<std::size_t I = 0, typename... Args1, typename... Args2>
inline typename std::enable_if<I == sizeof...(Args1), void>::type
copyTuple(std::tuple<Args1...>& dest, std::tuple<Args2...>& src) {}

template<std::size_t I = 0, typename... Args1, typename... Args2>
inline typename std::enable_if<I < sizeof...(Args1), void>::type
copyTuple(std::tuple<Args1...>& dest, std::tuple<Args2...>& src) {
  copy(std::get<I>(dest), std::get<I>(src));
  copyTuple<I + 1>(dest, src);
}

// call member function
template <typename C,
          typename FC, typename R, typename ...FArgs, typename ...Args>
R call(C& obj, R (FC::*f)(FArgs...), Args&&... args) {
  return (obj.*f)(args...);
}

template <std::size_t... I, typename C, typename R, typename ...Args>
void BaseMatrixCompare(R (C::*f)(Args...)) {
  for (auto height : {1, 11, 73, 128, 200, 330}) {
    for (auto width : {1, 3, 32, 100, 512, 1000, 3210}) {
      CpuMatrix obj1(height, width);
      GpuMatrix obj2(height, width);
      init(obj1);
      copy(obj2, obj1);

      auto tuple1 = std::make_tuple(
        construct<typename ReplaceType<typename std::decay<
          typename std::tuple_element<I, std::tuple<Args...>>::type>::type,
          CpuMatrix>::type>(height, width)...);

      auto tuple2 = std::make_tuple(
        construct<typename ReplaceType<typename std::decay<
          typename std::tuple_element<I, std::tuple<Args...>>::type>::type,
          GpuMatrix>::type>(height, width)...);

      initTuple(tuple1);
      copyTuple(tuple2, tuple1);

      call(obj1, f, std::get<I>(tuple1)...);
      call(obj2, f, std::get<I>(tuple2)...);

      TensorCheckErr(obj1, obj2);
    }
  }
}

}  // namespace autotest

template <std::size_t... I, typename C, typename R, typename ...Args>
void BaseMatrixCompare(R (C::*f)(Args...)) {
  static_assert(sizeof...(I) == sizeof...(Args),
    "size of parameter packs are not equal");

  autotest::BaseMatrixCompare<I...>(f);
}

