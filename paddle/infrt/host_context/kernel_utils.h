// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <llvm/ADT/ArrayRef.h>

#include <utility>

#include "paddle/infrt/host_context/kernel_frame.h"
#include "paddle/infrt/host_context/value.h"

namespace infrt {
namespace host_context {

template <typename T>
class Argument {
 public:
  explicit Argument(ValueRef value) : value_(value) {}

  ValueRef& value() { return value_; }
  const ValueRef& value() const { return value_; }

  T& get() const { return value_.get<T>(); }

 private:
  ValueRef value_;
};

/**
 * RemainingArguments collects all remaining arguments in an ArrayRef.
 */
class RemainingArguments {
 public:
  explicit RemainingArguments(llvm::ArrayRef<Value*> remaining_arguments)
      : remaining_arguments_(remaining_arguments) {}

  llvm::ArrayRef<Value*> values() const { return remaining_arguments_; }
  size_t size() const { return remaining_arguments_.size(); }
  const Value* operator[](size_t i) const { return remaining_arguments_[i]; }

 private:
  llvm::ArrayRef<Value*> remaining_arguments_;
};

/**
 * RemainingResults collects all remaining results in a MutableArrayRef.
 */
class RemainingResults {
 public:
  explicit RemainingResults(llvm::MutableArrayRef<ValueRef> remaining_results)
      : remaining_results_(remaining_results) {}
  llvm::MutableArrayRef<ValueRef> values() { return remaining_results_; }
  size_t size() const { return remaining_results_.size(); }

  template <typename T>
  const ValueRef& AllocateAt(int index) {
    // eagerly create a ValueRef
    if (remaining_results_[index].get()) return remaining_results_[index];
    remaining_results_[index] = ValueRef(new Value);
    return remaining_results_[index];
  }
  ValueRef& operator[](size_t i) const { return remaining_results_[i]; }

 private:
  llvm::MutableArrayRef<ValueRef> remaining_results_;
};

template <typename T>
class Result {
 public:
  explicit Result(ValueRef* result) : result_(result) {}

  template <typename... Args>
  void Emplace(Args&&... args) {
    ValueRef v;
    Set(T(std::forward<Args>(args)...));
  }

  void Set(Argument<T> argument) {
    CHECK(!result_->IsValid());
    *result_ = argument.value();
  }

 private:
  ValueRef* result_{};
};

template <typename T>
class Attribute {
 public:
  explicit Attribute(const Value* value) : value_(value) {}

  const T& get() const { return value_->get<T>(); }

 private:
  const Value* value_;
};

template <typename ViewT>
class ArgumentView {
  using UnderlyingT = typename ViewT::UnderlyingT;

 public:
  explicit ArgumentView(Value* value)
      : value_(value), arg_(&value->template get<UnderlyingT>()) {}

  Value* value() const { return value_; }
  ViewT& get() const { return arg_; }
  ViewT* operator->() const { return &get(); }
  ViewT& operator*() const { return get(); }

 private:
  Value* value_{};
  mutable ViewT arg_;
};

template <typename F, F f>
struct KernelImpl;

template <typename T>
struct TypeTag {};

#define INFRT_KERNEL(...)                                   \
  ::infrt::host_context::KernelImpl<decltype(&__VA_ARGS__), \
                                    &__VA_ARGS__>::Invoke

template <typename Return, typename... Args, Return (*impl_fn)(Args...)>
struct KernelImpl<Return (*)(Args...), impl_fn> {
  static void Invoke(KernelFrame* frame) {
    KernelCallHelper<Args..., TypeTag<int>>::template Invoke<0, 0, 0>(frame);
  }

  // Helper that introspects the arguments to derive the signature and cast
  // parts of the KernelFrame to their type before passing them to impl_fn.
  template <typename... RemainingArgs>
  struct KernelCallHelper;

  // Casts the return value of the kernel, if non-void.
  // bool _ is an unnecessary parameter to make compiler allow templace specific
  // in non-namespace scope.
  template <typename T, bool _>
  struct KernelReturnHelper {
    static void Invoke(KernelFrame* frame, const Args&... args) {
      HandleReturn(frame, impl_fn(args...));
    }
  };

  template <bool _>
  struct KernelReturnHelper<void, _> {
    static void Invoke(KernelFrame* frame, const Args&... args) {
      impl_fn(args...);
    }
  };

  // Specialization to cast a single input argument(Head).
  template <typename Head, typename... Tail>
  struct KernelCallHelper<Argument<Head>, Tail...> {
    template <int in_idx, int out_idx, int const_idx, typename... PreviousArgs>
    static void Invoke(KernelFrame* frame, const PreviousArgs&... pargs) {
      static_assert(in_idx != -1,
                    "Do not place Arguments after RemainingArguments");
      static_assert(out_idx == 0, "Arguments should appear before results");
      static_assert(const_idx == 0,
                    "Arguments and results should appear before attributes.");

      Argument<Head> arg(frame->GetArgAt(in_idx));
      KernelCallHelper<
          Tail...>::template Invoke<in_idx + 1, out_idx, const_idx>(frame,
                                                                    pargs...,
                                                                    arg);
    }
  };

  template <typename Head, typename... Tail>
  struct KernelCallHelper<ArgumentView<Head>, Tail...> {
    template <int in_idx, int out_idx, int const_idx, typename... PreviousArgs>
    static void Invoke(KernelFrame* frame, const PreviousArgs&... pargs) {
      static_assert(in_idx != -1,
                    "Do not place Arguments after RemainingArguments");
      static_assert(out_idx == 0, "Arguments should appear before results");
      static_assert(const_idx == 0,
                    "Arguments and results should appear before attributes.");

      ArgumentView<Head> arg(frame->GetArgAt(in_idx));
      KernelCallHelper<
          Tail...>::template Invoke<in_idx + 1, out_idx, const_idx>(frame,
                                                                    pargs...,
                                                                    arg);
    }
  };

  // Specialization to cast a single result argument (Head).
  template <typename Head, typename... Tail>
  struct KernelCallHelper<Result<Head>, Tail...> {
    template <int in_idx, int out_idx, int const_idx, typename... PreviousArgs>
    static void Invoke(KernelFrame* frame, const PreviousArgs&... pargs) {
      static_assert(out_idx != -1,
                    "Do not place Results after RemainingResults");
      // static_assert(const_idx == 0,
      //              "Arguments and results should appear before attributes");

      // Result<Head> arg(&frame->GetResults()[out_idx]);
      Result<Head> arg(new ValueRef());
      KernelCallHelper<
          Tail...>::template Invoke<in_idx, out_idx + 1, const_idx>(frame,
                                                                    pargs...,
                                                                    arg);
    }
  };

  // Specialization to cast a single attribute.
  template <typename Head, typename... Tail>
  struct KernelCallHelper<Attribute<Head>, Tail...> {
    template <int in_idx, int out_idx, int const_idx, typename... PreviousArgs>
    static void Invoke(KernelFrame* frame, const PreviousArgs&... pargs) {
      // static_assert(const_idx != -1,
      //              "Do not place Attributes after RemainingAttributes");
      Attribute<Head> arg(frame->GetAttributeAt(const_idx));
      KernelCallHelper<
          Tail...>::template Invoke<in_idx, out_idx, const_idx + 1>(frame,
                                                                    pargs...,
                                                                    arg);
    }
  };

  // Treat other pointer as an Argument.
  template <typename Head, typename... Tail>
  struct KernelCallHelper<Head*, Tail...> {
    template <int in_idx, int out_idx, int const_idx, typename... PreviousArgs>
    static void Invoke(KernelFrame* frame, const PreviousArgs&... pargs) {
      static_assert(in_idx != -1,
                    "Do not place Arguments after RemainingArguments");
      static_assert(out_idx == 0, "Arguments should appear before results");
      // static_assert(const_idx == 0,
      //              "Arguments and results should appear before attributes.");
      auto* arg = &frame->template GetElementAt<Head>(in_idx);
      KernelCallHelper<
          Tail...>::template Invoke<in_idx + 1, out_idx, const_idx>(frame,
                                                                    pargs...,
                                                                    arg);
    }
  };

  // Treat any other type as an Argument.
  template <typename Head, typename... Tail>
  struct KernelCallHelper<Head, Tail...> {
    using ArgT = std::decay_t<Head>;

    template <int in_idx, int out_idx, int const_idx, typename... PreviousArgs>
    static void Invoke(KernelFrame* frame, const PreviousArgs&... pargs) {
      static_assert(in_idx != -1,
                    "Do not place Arguments after RemainingArguments");
      static_assert(out_idx == 0, "Arguments should appear before results");
      static_assert(const_idx == 0,
                    "Arguments and results should appear before attributes.");

      auto* value = frame->GetElementAt(in_idx);
      auto&& arg = value->get<ArgT>();

      KernelCallHelper<
          Tail...>::template Invoke<in_idx + 1, out_idx, const_idx>(frame,
                                                                    pargs...,
                                                                    arg);
    }
  };

  // RemainingArguments provides an ArrayRef<AsyncValue*> containing all
  // remaining arguments. Useful for variadic
  // kernels.
  template <typename... Tail>
  struct KernelCallHelper<RemainingArguments, Tail...> {
    template <int in_idx, int out_idx, int const_idx, typename... PreviousArgs>
    static void Invoke(KernelFrame* frame, const PreviousArgs&... pargs) {
      static_assert(in_idx != -1,
                    "Do not use more than one RemainingArguments");
      static_assert(out_idx == 0, "Arguments should appear before results.");
      static_assert(const_idx == 0,
                    "Arguments and results should appear before attributes");
      RemainingArguments remaining_arguments(
          frame->GetArguments().drop_front(in_idx));

      KernelCallHelper<Tail...>::template Invoke<-1, out_idx, const_idx>(
          frame, pargs..., remaining_arguments);
    }
  };

  // RemainingResults provides an MutableArrayRef<AsyncValue*> containing all
  // remaining results.
  template <typename... Tail>
  struct KernelCallHelper<RemainingResults, Tail...> {
    template <int in_idx, int out_idx, int const_idx, typename... PreviousArgs>
    static void Invoke(KernelFrame* frame, const PreviousArgs&... pargs) {
      static_assert(out_idx != -1, "Do not use more than one RemainingResults");
      static_assert(const_idx == 0,
                    "Arguments and results should appear before attributes");
      llvm::MutableArrayRef<Value*> returned_results =
          frame->GetResults().drop_front(out_idx);

      llvm::SmallVector<ValueRef, 4> result_values;
      for (size_t i = 0; i < returned_results.size(); i++)
        result_values.emplace_back(returned_results[i]);

      RemainingResults remaining_results(result_values);
      KernelCallHelper<Tail...>::template Invoke<in_idx, -1, const_idx>(
          frame, pargs..., remaining_results);
    }
  };

  // No arguments left.
  template <typename T>
  struct KernelCallHelper<TypeTag<T>> {
    template <int in_idx, int out_idx, int const_idx, typename... PreviousArgs>
    static void Invoke(KernelFrame* frame, const PreviousArgs&... pargs) {
      KernelReturnHelper<Return, false>::Invoke(frame, pargs...);
    }
  };

  // Handle pair result
  template <typename T0, typename T1>
  static void HandleReturn(KernelFrame* frame, std::pair<T0, T1>&& t) {
    CHECK_EQ(frame->GetNumResults(), 2);
    StoreResultAt(frame, 0, std::move(t.first));
    StoreResultAt(frame, 1, std::move(t.second));
  }

  // Store the function result back to the output Value in KernelFrame.
  template <typename T>
  static void HandleReturn(KernelFrame* frame, T&& t) {
    assert(frame->GetNumResults() == 1 && "Extra results passed to kernel.");
    StoreResultAt(frame, 0, std::forward<T>(t));
  }

  // Store result as an Value output in KernelFrame.
  template <typename T>
  static void StoreResultAt(KernelFrame* frame, int index, T&& t) {
    frame->EmplaceResult<std::decay_t<T>>(index, std::forward<T>(t));
  }
};

}  // namespace host_context
}  // namespace infrt
