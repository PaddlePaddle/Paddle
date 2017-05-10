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

/*
 execViaCpu is used to do operations on GpuMatirx and/or GpuIVector through
 cpu functions. It can automatically make a temporary CPU copy for the
 gpu matrix/vector, and copy back after executing the CPU function.

 Examples:
 1. For a function, functor or lambda:
   r = execViaCpu(&f, mat, vec)

 2. For member function of CpuMatirx, execViaCpu2 should be used:
   execViaCpu2(&CpuMatrix::selectElements, *this, table, ids)
*/

#pragma once

namespace paddle {

template <typename Arg>
class CopyToCpu {
public:
  explicit CopyToCpu(Arg& arg) : arg_(arg) {}
  Arg& copiedArg() const { return arg_; }

private:
  Arg& arg_;
};

template <>
class CopyToCpu<Matrix> {
public:
  explicit CopyToCpu(Matrix& arg) : arg_(arg) {
    if (arg.useGpu()) {
      CHECK(!arg.isTransposed()) << "Not supported";
      copied_ = Matrix::create(arg.getHeight(),
                               arg.getWidth(),
                               /* trans= */ false,
                               /* useGpu= */ false);
      copied_->copyFrom(arg);
    }
  }
  ~CopyToCpu() {
    if (copied_) {
      arg_.copyFrom(*copied_);
    }
  }
  Matrix& copiedArg() const { return copied_ ? *copied_ : arg_; }

private:
  Matrix& arg_;
  MatrixPtr copied_;
};

template <>
class CopyToCpu<const Matrix> {
public:
  explicit CopyToCpu(const Matrix& arg) : arg_(arg) {
    if (arg.useGpu()) {
      CHECK(!arg.isTransposed()) << "Not supported";
      copied_ = Matrix::create(arg.getHeight(),
                               arg.getWidth(),
                               /* trans= */ false,
                               /* useGpu= */ false);
      copied_->copyFrom(arg);
    }
  }
  const Matrix& copiedArg() const { return copied_ ? *copied_ : arg_; }

private:
  const Matrix& arg_;
  MatrixPtr copied_;
};

template <>
class CopyToCpu<IVector> {
public:
  explicit CopyToCpu(IVector& arg) : arg_(arg) {
    if (arg.useGpu()) {
      copied_ = IVector::create(arg.getSize(), /* useGpu= */ false);
      copied_->copyFrom(arg);
    }
  }
  ~CopyToCpu() {
    if (copied_) {
      arg_.copyFrom(*copied_);
    }
  }
  IVector& copiedArg() const { return copied_ ? *copied_ : arg_; }

private:
  IVector& arg_;
  IVectorPtr copied_;
};

template <>
class CopyToCpu<const IVector> {
public:
  explicit CopyToCpu(const IVector& arg) : arg_(arg) {
    if (arg.useGpu()) {
      copied_ = IVector::create(arg.getSize(), /* useGpu= */ false);
      copied_->copyFrom(arg);
    }
  }
  const IVector& copiedArg() const { return copied_ ? *copied_ : arg_; }

private:
  const IVector& arg_;
  IVectorPtr copied_;
};

namespace detail {

template <bool isFunction, bool isFunctionPointer, bool isClass, typename F>
class GpuFuncWrapperImp;

template <typename F, typename R, typename... Args>
class GpuFuncWrapperBase {
public:
  typedef R ResultType;
  R operator()(F&& f, Args... args) {
    return f(CopyToCpu<typename std::remove_reference<Args>::type>(args)
                 .copiedArg()...);
  }
};

// function
template <typename R, typename... Args>
class GpuFuncWrapperImp<true, false, false, R(Args...)>
    : public GpuFuncWrapperBase<R(Args...), R, Args...> {};

// function pointer
template <typename R, typename... Args>
class GpuFuncWrapperImp<false, true, false, R (*)(Args...)>
    : public GpuFuncWrapperBase<R (*)(Args...), R, Args...> {};

template <typename F, typename Op>
class GpuFuncWrapperImp2;

template <typename F, typename C, typename R, typename... Args>
class GpuFuncWrapperImp2<F, R (C::*)(Args...) const>
    : public GpuFuncWrapperBase<F, R, Args...> {};

template <typename F, typename C, typename R, typename... Args>
class GpuFuncWrapperImp2<F, R (C::*)(Args...)>
    : public GpuFuncWrapperBase<F, R, Args...> {};

// functor or lambda
template <typename F>
class GpuFuncWrapperImp<false, false, true, F>
    : public GpuFuncWrapperImp2<F, decltype(&F::operator())> {};

template <typename F>
class GpuFuncWrapper2
    : public GpuFuncWrapperImp<
          std::is_function<F>::value,
          std::is_pointer<F>::value &&
              std::is_function<typename std::remove_pointer<F>::type>::value,
          std::is_class<F>::value,
          F> {};

template <typename F>
class GpuFuncWrapper
    : public GpuFuncWrapper2<typename std::remove_reference<F>::type> {};

}  // namespace detail

template <typename F, typename... Args>
typename detail::GpuFuncWrapper<F>::ResultType execViaCpu(F&& f,
                                                          Args&&... args) {
  return detail::GpuFuncWrapper<F>()(std::move(f), args...);
}

// The second version is for F as member function of CpuMatrix
template <typename R, typename... FArgs, typename... Args>
R execViaCpu2(R (CpuMatrix::*f)(FArgs...), Args&&... args) {
  auto lambda = [](R (CpuMatrix::*f)(FArgs...), Matrix& ths, FArgs... args) {
    return (((CpuMatrix&)ths).*f)(args...);
  };
  return execViaCpu(lambda, f, args...);
}

}  // namespace paddle
