/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include <functional>
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "paddle/fluid/extension/include/operator.h"
#include "paddle/fluid/extension/include/utils.h"

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor.h"

namespace paddle {
namespace framework {

namespace detail {

// default prefix

constexpr char kCustomOpInputPrefix[] = "X";
constexpr char kCustomOpOutputPrefix[] = "Out";

template <typename... Items>
struct typelist final {
 private:
  typelist() = delete;
};

// dereference & apply tools

template <typename traits, std::size_t... INDEX>
typename traits::ArgsTuple DereferenceImpl(
    const framework::ExecutionContext& ctx, std::index_sequence<INDEX...>) {
  VLOG(0) << "Dereference Impl";
  return std::make_tuple((*(ctx.Input<Tensor>(detail::kCustomOpInputPrefix +
                                              std::to_string(INDEX))))...);
}

template <typename traits>
typename traits::ArgsTuple Dereference(const framework::ExecutionContext& ctx) {
  using Indices = std::make_index_sequence<traits::arity>;
  VLOG(0) << "Dereference";
  return DereferenceImpl<traits>(ctx, Indices{});
}

template <typename func_t, typename Tuple, std::size_t... INDEX>
decltype(auto) ApplyImpl(func_t&& f, Tuple&& t, std::index_sequence<INDEX...>) {
  VLOG(0) << "ApplyImpl";
  return std::forward<func_t>(f)(std::get<INDEX>(std::forward<Tuple>(t))...);
}

template <typename func_t, typename Tuple>
decltype(auto) Apply(func_t&& f, Tuple&& t) {
  VLOG(0) << "Apply";
  return ApplyImpl(
      std::forward<func_t>(f), std::forward<Tuple>(t),
      std::make_index_sequence<
          std::tuple_size<std::remove_reference_t<Tuple>>::value>{});
}

template <typename func_t>
size_t ArgNumParser(func_t&& func) {
  using traits = paddle::detail::function_traits<func_t>;
  return traits::arity;
}

template <typename T>
T* DynLoad(void* handle, std::string name) {
  T* func = reinterpret_cast<T*>(dlsym(handle, name.c_str()));
#if !defined(_WIN32)
  auto errorno = dlerror();
#else
  auto errorno = GetLastError();
#endif  // !_WIN32
  PADDLE_ENFORCE_NOT_NULL(
      func,
      platform::errors::NotFound(
          "Failed to load dynamic operator library, error code(%s).", errorno));
  return func;
}

}  // namespace detail

class CustomOperator : public OperatorWithKernel {
 public:
  using OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    VLOG(0) << "Infer shape of custom operator.";
  }
};

// custom op kernel define

template <typename func_t>
static void CallKernelFunc(const framework::ExecutionContext& ctx,
                           func_t&& func) {
  using traits = paddle::detail::function_traits<func_t>;
  // using result_type = typename traits::result_type;
  VLOG(0) << "run in CallKernelFunc";
  // auto outs = detail::Apply(std::forward<func_t>(func),
  // detail::Dereference<traits>(ctx));
  auto* x = ctx.Input<Tensor>(detail::kCustomOpInputPrefix + std::to_string(0));
  PADDLE_ENFORCE_NOT_NULL(x, "input x is nullptr.");
  PADDLE_ENFORCE(x->IsInitialized(), "input x is not initialized.");
  // VLOG(0) << *x;
  auto outs = func(*x);
  VLOG(0) << "after apply in CallKernelFunc";
  // VLOG(0) << outs[0];
  auto* t = ctx.Output<Tensor>(detail::kCustomOpOutputPrefix);
  t->ShareDataWith(outs.at(0));
  // for (size_t i = 0; i < outs.size(); ++i) {
  //   auto* t = ctx.Output<Tensor>(detail::kCustomOpOutputPrefix +
  //   std::to_string(i));
  //   t->ShareDataWith(outs.at(i));
  // }
}

template <typename DataType, typename func_t>
class CustomOpKernel : public framework::OpKernel<DataType> {
 public:
  explicit CustomOpKernel(func_t func) : func_(func) {}

  void Compute(const framework::ExecutionContext& ctx) const override {
    VLOG(0) << "run in compute";
    CallKernelFunc(ctx, func_);
  }

 private:
  func_t func_;
};

// load op api
void LoadCustomOperator(const std::string& dso_name);

}  // namespace framework
}  // namespace paddle
