// Copyright (c) 2025 Paddle Authors. All Rights Reserved.
// This file copy from boost/any.hpp and boost version: 1.41.0
// Modified the following points:
// 1. modify namespace from boost::any to paddle::any
// 2. remove the depending boost header files
// 3. remove/modify some macro
// 4. use std::unique_ptr instead of raw pointer

// See http://www.boost.org/libs/any for Documentation.

#pragma once

// what:  variant type boost::any
// who:   contributed by Kevlin Henney,
//        with features contributed and bugs found by
//        Ed Brey, Mark Rodgers, Peter Dimov, and James Curran
// when:  July 2001
// where: tested with BCC 5.5, MSVC 6.0, and g++ 2.95

#include <algorithm>
#include <memory>
#include <type_traits>
#include <typeinfo>

// See boost/python/type_id.hpp
#if (defined(__GNUC__) && __GNUC__ >= 3) || defined(_AIX) || \
    (defined(__sgi) && defined(__host_mips)) ||              \
    (defined(__hpux) && defined(__HP_aCC)) ||                \
    (defined(linux) && defined(__INTEL_COMPILER) && defined(__ICC))
#define BOOST_AUX_ANY_TYPE_ID_NAME
#include <cstring>
#endif

namespace paddle {
class any {
 public:  // structors
  any() : content(nullptr) {}

  template <typename ValueType>
  explicit any(const ValueType &value)
      : content(std::make_unique<holder<ValueType>>(value)) {}

  any(const any &other)
      : content(other.content ? other.content->clone() : nullptr) {}

  any(any &&other) noexcept = default;

  ~any() = default;

 public:  // modifiers
  any &swap(any &rhs) {
    std::swap(content, rhs.content);
    return *this;
  }

  template <typename ValueType>
  any &operator=(const ValueType &rhs) {
    any(rhs).swap(*this);  // NOLINT(runtime/explicit)
    return *this;
  }

  any &operator=(any rhs) {
    rhs.swap(*this);
    return *this;
  }

 public:  // queries
  bool empty() const { return !content; }

  const std::type_info &type() const {
    return content ? content->type() : typeid(void);
  }

 public:  // types (public so any_cast can be non-friend)
  class placeholder {
   public:  // structors
    virtual ~placeholder() {}

   public:  // queries
    virtual const std::type_info &type() const = 0;

    virtual std::unique_ptr<placeholder> clone() const = 0;
  };

  template <typename ValueType>
  class holder : public placeholder {
   public:  // structors
    explicit holder(const ValueType &value) : held(value) {}

   public:  // queries
    const std::type_info &type() const override { return typeid(ValueType); }

    std::unique_ptr<placeholder> clone() const override {
      return std::make_unique<holder>(held);
    }

   public:  // representation
    ValueType held;

   private:  // intentionally left unimplemented
    holder &operator=(const holder &);
  };

 public:  // representation (public so any_cast can be non-friend)
  std::unique_ptr<placeholder> content;
};

class bad_any_cast : public std::bad_cast {
 public:
  const char *what() const throw() override {
    return "paddle::bad_any_cast: "
           "failed conversion using paddle::any_cast";
  }
};

template <typename ValueType>
ValueType *any_cast(any *operand) {
  return operand &&
#ifdef BOOST_AUX_ANY_TYPE_ID_NAME
                 std::strcmp(operand->type().name(),
                             typeid(ValueType).name()) == 0
#else
                 operand->type() == typeid(ValueType)
#endif
             ? &(static_cast<any::holder<ValueType> *>(operand->content.get())
                     ->held)
             : 0;
}

template <typename ValueType>
inline const ValueType *any_cast(const any *operand) {
  return any_cast<ValueType>(const_cast<any *>(operand));
}

template <typename ValueType>
ValueType any_cast(const any &operand) {
  typedef typename std::remove_reference<ValueType>::type nonref;

  static_assert(!std::is_reference<nonref>::value,
                "!std::is_reference<nonref>::value");

  nonref *result = any_cast<nonref>(&operand);
  if (!result) throw bad_any_cast();
  return *result;
}

template <typename ValueType>
inline ValueType any_cast(const any &operand) {
  typedef typename std::remove_reference<ValueType>::type nonref;

  static_assert(!std::is_reference<nonref>::value,
                "!std::is_reference<nonref>::value");

  return any_cast<const nonref &>(const_cast<any &>(operand));
}

// Note: The "unsafe" versions of any_cast are not part of the
// public interface and may be removed at any time. They are
// required where we know what type is stored in the any and can't
// use typeid() comparison, e.g., when our types may travel across
// different shared libraries.
template <typename ValueType>
inline ValueType *unsafe_any_cast(any *operand) {
  return &(static_cast<any::holder<ValueType> *>(operand->content.get())->held);
}

template <typename ValueType>
inline const ValueType *unsafe_any_cast(const any *operand) {
  return unsafe_any_cast<ValueType>(const_cast<any *>(operand));
}
}  // namespace paddle

// Copyright Kevlin Henney, 2000, 2001, 2002. All rights reserved.
//
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
