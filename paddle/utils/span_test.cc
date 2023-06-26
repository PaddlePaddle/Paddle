// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
// This file copy from https://github.com/tcbrindle/span
// Modified the following points
// 1. remove macros for backward compatibility with pre-C++17 standards
// 2. instantiated namespace name with paddle

/*
This is an implementation of C++20's std::span
http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/n4820.pdf
*/
//          Copyright Tristan Brindle 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file ../../LICENSE_1_0.txt or copy at
//          https://www.boost.org/LICENSE_1_0.txt)

#include <cassert>
#include <deque>
#include <initializer_list>
#include <vector>

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/utils/span.h"

using paddle::span;

// span();
TEST(default_ctor, span) {
  static_assert(std::is_nothrow_default_constructible<span<int>>::value, "");
  static_assert(std::is_nothrow_default_constructible<span<int, 0>>::value, "");
  static_assert(!std::is_default_constructible<span<int, 42>>::value, "");

  // dynamic size
  {
    constexpr span<int> s{};
    static_assert(s.size() == 0, "");
    static_assert(s.data() == nullptr, "");
#ifndef _MSC_VER
    static_assert(s.begin() == s.end(), "");
#else
    CHECK(s.begin() == s.end());
#endif
  }

  // fixed size
  {
    constexpr span<int, 0> s{};
    static_assert(s.size() == 0, "");
    static_assert(s.data() == nullptr, "");
#ifndef _MSC_VER
    static_assert(s.begin() == s.end(), "");
#else
    CHECK(s.begin() == s.end());
#endif
  }
}

// span(pointer ptr, size_type count);
TEST(pointer_length_ctor, span) {
  static_assert(std::is_constructible<span<int>, int*, int>::value, "");
  static_assert(std::is_constructible<span<const int>, int*, int>::value, "");
  static_assert(std::is_constructible<span<const int>, const int*, int>::value,
                "");
  static_assert(std::is_constructible<span<int, 42>, int*, int>::value, "");
  static_assert(std::is_constructible<span<const int, 42>, int*, int>::value,
                "");
  static_assert(
      std::is_constructible<span<const int, 42>, const int*, int>::value, "");

  // dynamic size
  {
    int arr[] = {1, 2, 3};
    span<int> s(arr, 3);

    CHECK_EQ(s.size(), 3UL);
    CHECK_EQ(s.data(), arr);
    CHECK_EQ(s.begin(), std::begin(arr));
    CHECK_EQ(s.end(), std::end(arr));
  }

  // fixed size
  {
    int arr[] = {1, 2, 3};
    span<int, 3> s(arr, 3);

    CHECK_EQ(s.size(), 3UL);
    CHECK_EQ(s.data(), arr);
    CHECK_EQ(s.begin(), std::begin(arr));
    CHECK_EQ(s.end(), std::end(arr));
  }
}

// span(pointer ptr, pointer ptr);
TEST(pointer_pointer_ctor, span) {
  static_assert(std::is_constructible<span<int>, int*, int*>::value, "");
  static_assert(!std::is_constructible<span<int>, float*, float*>::value, "");
  static_assert(std::is_constructible<span<int, 42>, int*, int*>::value, "");
  static_assert(!std::is_constructible<span<int, 42>, float*, float*>::value,
                "");

  // dynamic size
  {
    int arr[] = {1, 2, 3};
    span<int> s{arr, arr + 3};

    CHECK_EQ(s.size(), 3UL);
    CHECK_EQ(s.data(), arr);
    CHECK_EQ(s.begin(), std::begin(arr));
    CHECK_EQ(s.end(), std::end(arr));
  }

  // fixed size
  {
    int arr[] = {1, 2, 3};
    span<int, 3> s{arr, arr + 3};

    CHECK_EQ(s.size(), 3UL);
    CHECK_EQ(s.data(), arr);
    CHECK_EQ(s.begin(), std::begin(arr));
    CHECK_EQ(s.end(), std::end(arr));
  }
}

TEST(c_array_ctor, span) {
  using int_array_t = int[3];
  using float_array_t = float[3];

  static_assert(std::is_nothrow_constructible<span<int>, int_array_t&>::value,
                "");
  static_assert(!std::is_constructible<span<int>, int_array_t const&>::value,
                "");
  static_assert(!std::is_constructible<span<int>, float_array_t>::value, "");

  static_assert(
      std::is_nothrow_constructible<span<const int>, int_array_t&>::value, "");
  static_assert(
      std::is_nothrow_constructible<span<const int>, int_array_t const&>::value,
      "");
  static_assert(!std::is_constructible<span<const int>, float_array_t>::value,
                "");

  static_assert(
      std::is_nothrow_constructible<span<int, 3>, int_array_t&>::value, "");
  static_assert(!std::is_constructible<span<int, 3>, int_array_t const&>::value,
                "");
  static_assert(!std::is_constructible<span<int, 3>, float_array_t&>::value,
                "");

  static_assert(
      std::is_nothrow_constructible<span<const int, 3>, int_array_t&>::value,
      "");
  static_assert(std::is_nothrow_constructible<span<const int, 3>,
                                              int_array_t const&>::value,
                "");
  static_assert(
      !std::is_constructible<span<const int, 3>, float_array_t>::value, "");

  static_assert(!std::is_constructible<span<int, 42>, int_array_t&>::value, "");
  static_assert(
      !std::is_constructible<span<int, 42>, int_array_t const&>::value, "");
  static_assert(!std::is_constructible<span<int, 42>, float_array_t&>::value,
                "");

  static_assert(
      !std::is_constructible<span<const int, 42>, int_array_t&>::value, "");
  static_assert(
      !std::is_constructible<span<const int, 42>, int_array_t const&>::value,
      "");
  static_assert(
      !std::is_constructible<span<const int, 42>, float_array_t&>::value, "");

  // non-const, dynamic size
  {
    int arr[] = {1, 2, 3};
    span<int> s{arr};
    CHECK_EQ(s.size(), 3UL);
    CHECK_EQ(s.data(), arr);
    CHECK_EQ(s.begin(), std::begin(arr));
    CHECK_EQ(s.end(), std::end(arr));
  }

  // const, dynamic size
  {
    int arr[] = {1, 2, 3};
    span<int const> s{arr};
    CHECK_EQ(s.size(), 3UL);
    CHECK_EQ(s.data(), arr);
    CHECK_EQ(s.begin(), std::begin(arr));
    CHECK_EQ(s.end(), std::end(arr));
  }

  // non-const, static size
  {
    int arr[] = {1, 2, 3};
    span<int, 3> s{arr};
    CHECK_EQ(s.size(), 3UL);
    CHECK_EQ(s.data(), arr);
    CHECK_EQ(s.begin(), std::begin(arr));
    CHECK_EQ(s.end(), std::end(arr));
  }

  // const, dynamic size
  {
    int arr[] = {1, 2, 3};
    span<int const, 3> s{arr};
    CHECK_EQ(s.size(), 3UL);
    CHECK_EQ(s.data(), arr);
    CHECK_EQ(s.begin(), std::begin(arr));
    CHECK_EQ(s.end(), std::end(arr));
  }
}

TEST(std_array_ctor, span) {
  using int_array_t = std::array<int, 3>;
  using float_array_t = std::array<float, 3>;
  using zero_array_t = std::array<int, 0>;

  static_assert(std::is_nothrow_constructible<span<int>, int_array_t&>::value,
                "");
  static_assert(!std::is_constructible<span<int>, int_array_t const&>::value,
                "");
  static_assert(!std::is_constructible<span<int>, float_array_t>::value, "");

  static_assert(
      std::is_nothrow_constructible<span<const int>, int_array_t&>::value, "");
  static_assert(
      std::is_nothrow_constructible<span<const int>, int_array_t const&>::value,
      "");
  static_assert(
      !std::is_constructible<span<const int>, float_array_t const&>::value, "");

  static_assert(
      std::is_nothrow_constructible<span<int, 3>, int_array_t&>::value, "");
  static_assert(!std::is_constructible<span<int, 3>, int_array_t const&>::value,
                "");
  static_assert(!std::is_constructible<span<int, 3>, float_array_t>::value, "");

  static_assert(
      std::is_nothrow_constructible<span<const int, 3>, int_array_t&>::value,
      "");
  static_assert(std::is_nothrow_constructible<span<const int, 3>,
                                              int_array_t const&>::value,
                "");
  static_assert(
      !std::is_constructible<span<const int, 3>, float_array_t const&>::value,
      "");

  static_assert(!std::is_constructible<span<int, 42>, int_array_t&>::value, "");
  static_assert(
      !std::is_constructible<span<int, 42>, int_array_t const&>::value, "");
  static_assert(
      !std::is_constructible<span<int, 42>, float_array_t const&>::value, "");

  static_assert(
      !std::is_constructible<span<const int, 42>, int_array_t&>::value, "");
  static_assert(
      !std::is_constructible<span<const int, 42>, int_array_t const&>::value,
      "");
  static_assert(
      !std::is_constructible<span<const int, 42>, float_array_t&>::value, "");

  static_assert(std::is_constructible<span<int>, zero_array_t&>::value, "");
  static_assert(!std::is_constructible<span<int>, const zero_array_t&>::value,
                "");
  static_assert(std::is_constructible<span<const int>, zero_array_t&>::value,
                "");
  static_assert(
      std::is_constructible<span<const int>, const zero_array_t&>::value, "");

  static_assert(std::is_constructible<span<int, 0>, zero_array_t&>::value, "");
  static_assert(
      !std::is_constructible<span<int, 0>, const zero_array_t&>::value, "");
  static_assert(std::is_constructible<span<const int, 0>, zero_array_t&>::value,
                "");
  static_assert(
      std::is_constructible<span<const int, 0>, const zero_array_t&>::value,
      "");

  // non-const, dynamic size
  {
    int_array_t arr = {1, 2, 3};
    span<int> s{arr};
    CHECK_EQ(s.size(), 3UL);
    CHECK_EQ(s.data(), arr.data());
    CHECK_EQ(s.begin(), arr.data());
    CHECK_EQ(s.end(), arr.data() + 3);
  }

  // const, dynamic size
  {
    int_array_t arr = {1, 2, 3};
    span<int const> s{arr};
    CHECK_EQ(s.size(), 3UL);
    CHECK_EQ(s.data(), arr.data());
    CHECK_EQ(s.begin(), arr.data());
    CHECK_EQ(s.end(), arr.data() + 3);
  }

  // non-const, static size
  {
    int_array_t arr = {1, 2, 3};
    span<int, 3> s{arr};
    CHECK_EQ(s.size(), 3UL);
    CHECK_EQ(s.data(), arr.data());
    CHECK_EQ(s.begin(), arr.data());
    CHECK_EQ(s.end(), arr.data() + 3);
  }

  // const, dynamic size
  {
    int_array_t arr = {1, 2, 3};
    span<int const, 3> s{arr};
    CHECK_EQ(s.size(), 3UL);
    CHECK_EQ(s.data(), arr.data());
    CHECK_EQ(s.begin(), arr.data());
    CHECK_EQ(s.end(), arr.data() + 3);
  }
}

TEST(ctor_from_containers, span) {
  using vec_t = std::vector<int>;
  using deque_t = std::deque<int>;

  static_assert(std::is_constructible<span<int>, vec_t&>::value, "");
  static_assert(!std::is_constructible<span<int>, const vec_t&>::value, "");
  static_assert(!std::is_constructible<span<int>, const deque_t&>::value, "");

  static_assert(std::is_constructible<span<const int>, vec_t&>::value, "");
  static_assert(std::is_constructible<span<const int>, const vec_t&>::value,
                "");
  static_assert(!std::is_constructible<span<const int>, const deque_t&>::value,
                "");

  static_assert(!std::is_constructible<span<int, 3>, vec_t&>::value, "");
  static_assert(!std::is_constructible<span<int, 3>, const vec_t&>::value, "");
  static_assert(!std::is_constructible<span<int, 3>, const deque_t&>::value,
                "");

  static_assert(!std::is_constructible<span<const int, 3>, vec_t&>::value, "");
  static_assert(!std::is_constructible<span<const int, 3>, const vec_t&>::value,
                "");
  static_assert(
      !std::is_constructible<span<const int, 3>, const deque_t&>::value, "");

  // vector<bool> is not contiguous and cannot be converted to span<bool>
  // Regression test for https://github.com/tcbrindle/span/issues/24
  static_assert(!std::is_constructible<span<bool>, std::vector<bool>&>::value,
                "");
  static_assert(
      !std::is_constructible<span<const bool>, const std::vector<bool>&>::value,
      "");

  // non-const, dynamic size
  {
    vec_t arr = {1, 2, 3};
    span<int> s{arr};
    CHECK_EQ(s.size(), 3UL);
    CHECK_EQ(s.data(), arr.data());
    CHECK_EQ(s.begin(), arr.data());
    CHECK_EQ(s.end(), arr.data() + 3);
  }

  // const, dynamic size
  {
    vec_t arr = {1, 2, 3};
    span<int const> s{arr};
    CHECK_EQ(s.size(), 3UL);
    CHECK_EQ(s.data(), arr.data());
    CHECK_EQ(s.begin(), arr.data());
    CHECK_EQ(s.end(), arr.data() + 3);
  }

  // non-const, static size
  {
    std::array<int, 3> arr = {1, 2, 3};
    span<int, 3> s{arr};
    CHECK_EQ(s.size(), 3UL);
    CHECK_EQ(s.data(), arr.data());
    CHECK_EQ(s.begin(), arr.data());
    CHECK_EQ(s.end(), arr.data() + 3);
  }

  // const, dynamic size
  {
    std::array<int, 3> arr = {1, 2, 3};
    span<int const, 3> s{arr};
    CHECK_EQ(s.size(), 3UL);
    CHECK_EQ(s.data(), arr.data());
    CHECK_EQ(s.begin(), arr.data());
    CHECK_EQ(s.end(), arr.data() + 3);
  }
}

TEST(ctor_from_spans, span) {
  using zero_span = span<int, 0>;
  using zero_const_span = span<const int, 0>;
  using big_span = span<int, 1000000>;
  using big_const_span = span<const int, 1000000>;
  using dynamic_span = span<int>;
  using dynamic_const_span = span<const int>;

  static_assert(std::is_trivially_copyable<zero_span>::value, "");
  static_assert(std::is_trivially_move_constructible<zero_span>::value, "");
  static_assert(!std::is_constructible<zero_span, zero_const_span>::value, "");
  static_assert(!std::is_constructible<zero_span, big_span>::value, "");
  static_assert(!std::is_constructible<zero_span, big_const_span>::value, "");
  static_assert(std::is_nothrow_constructible<zero_span, dynamic_span>::value,
                "");
  static_assert(!std::is_constructible<zero_span, dynamic_const_span>::value,
                "");

  static_assert(
      std::is_nothrow_constructible<zero_const_span, zero_span>::value, "");
  static_assert(std::is_trivially_copyable<zero_const_span>::value, "");
  static_assert(std::is_trivially_move_constructible<zero_const_span>::value,
                "");
  static_assert(!std::is_constructible<zero_const_span, big_span>::value, "");
  static_assert(!std::is_constructible<zero_const_span, big_const_span>::value,
                "");
  static_assert(
      std::is_nothrow_constructible<zero_const_span, dynamic_span>::value, "");
  static_assert(
      std::is_nothrow_constructible<zero_const_span, dynamic_const_span>::value,
      "");

  static_assert(!std::is_constructible<big_span, zero_span>::value, "");
  static_assert(!std::is_constructible<big_span, zero_const_span>::value, "");
  static_assert(std::is_trivially_copyable<big_span>::value, "");
  static_assert(std::is_trivially_move_constructible<big_span>::value, "");
  static_assert(!std::is_constructible<big_span, big_const_span>::value, "");
  static_assert(std::is_nothrow_constructible<big_span, dynamic_span>::value,
                "");
  static_assert(!std::is_constructible<big_span, dynamic_const_span>::value,
                "");

  static_assert(!std::is_constructible<big_const_span, zero_span>::value, "");
  static_assert(!std::is_constructible<big_const_span, zero_const_span>::value,
                "");
  static_assert(std::is_trivially_copyable<big_const_span>::value, "");
  static_assert(std::is_trivially_move_constructible<big_const_span>::value,
                "");
  static_assert(std::is_nothrow_constructible<big_const_span, big_span>::value,
                "");
  static_assert(
      std::is_nothrow_constructible<big_const_span, dynamic_span>::value, "");
  static_assert(
      std::is_nothrow_constructible<big_const_span, dynamic_const_span>::value,
      "");

  static_assert(std::is_nothrow_constructible<dynamic_span, zero_span>::value,
                "");
  static_assert(!std::is_constructible<dynamic_span, zero_const_span>::value,
                "");
  static_assert(std::is_nothrow_constructible<dynamic_span, big_span>::value,
                "");
  static_assert(!std::is_constructible<dynamic_span, big_const_span>::value,
                "");
  static_assert(std::is_trivially_copyable<dynamic_span>::value, "");
  static_assert(std::is_trivially_move_constructible<dynamic_span>::value, "");
  static_assert(!std::is_constructible<dynamic_span, dynamic_const_span>::value,
                "");

  static_assert(
      std::is_nothrow_constructible<dynamic_const_span, zero_span>::value, "");
  static_assert(
      std::is_nothrow_constructible<dynamic_const_span, zero_const_span>::value,
      "");
  static_assert(
      std::is_nothrow_constructible<dynamic_const_span, big_span>::value, "");
  static_assert(
      std::is_nothrow_constructible<dynamic_const_span, big_const_span>::value,
      "");
  static_assert(
      std::is_nothrow_constructible<dynamic_const_span, dynamic_span>::value,
      "");
  static_assert(std::is_trivially_copyable<dynamic_const_span>::value, "");
  static_assert(std::is_trivially_move_constructible<dynamic_const_span>::value,
                "");

  constexpr zero_const_span s0{};
  constexpr dynamic_const_span d{s0};

  static_assert(d.size() == 0, "");
  static_assert(d.data() == nullptr, "");
#ifndef _MSC_VER
  static_assert(d.begin() == d.end(), "");
#else
  CHECK(d.begin() == d.end());
#endif
}

TEST(subview, span) {
  // first<N>
  {
    int arr[] = {1, 2, 3, 4, 5};
    span<int, 5> s{arr};
    auto f = s.first<3>();

    static_assert(std::is_same<decltype(f), span<int, 3>>::value, "");
    CHECK_EQ(f.size(), 3UL);
    CHECK_EQ(f.data(), arr);
    CHECK_EQ(f.begin(), arr);
    CHECK_EQ(f.end(), arr + 3);
  }

  // last<N>
  {
    int arr[] = {1, 2, 3, 4, 5};
    span<int, 5> s{arr};
    auto l = s.last<3>();

    static_assert(std::is_same<decltype(l), span<int, 3>>::value, "");
    CHECK_EQ(l.size(), 3UL);
    CHECK_EQ(l.data(), arr + 2);
    CHECK_EQ(l.begin(), arr + 2);
    CHECK_EQ(l.end(), std::end(arr));
  }

  // subspan<N>
  {
    int arr[] = {1, 2, 3, 4, 5};
    span<int, 5> s{arr};
    auto ss = s.subspan<1, 2>();

    static_assert(std::is_same<decltype(ss), span<int, 2>>::value, "");
    CHECK_EQ(ss.size(), 2UL);
    CHECK_EQ(ss.data(), arr + 1);
    CHECK_EQ(ss.begin(), arr + 1);
    CHECK_EQ(ss.end(), arr + 1 + 2);
  }

  // first(n)
  {
    int arr[] = {1, 2, 3, 4, 5};
    span<int, 5> s{arr};
    auto f = s.first(3);

    static_assert(std::is_same<decltype(f), span<int>>::value, "");
    CHECK_EQ(f.size(), 3UL);
    CHECK_EQ(f.data(), arr);
    CHECK_EQ(f.begin(), arr);
    CHECK_EQ(f.end(), arr + 3);
  }

  // last(n)
  {
    int arr[] = {1, 2, 3, 4, 5};
    span<int, 5> s{arr};
    auto l = s.last(3);

    static_assert(std::is_same<decltype(l), span<int>>::value, "");
    CHECK_EQ(l.size(), 3UL);
    CHECK_EQ(l.data(), arr + 2);
    CHECK_EQ(l.begin(), arr + 2);
    CHECK_EQ(l.end(), std::end(arr));
  }

  // subspan(n)
  {
    int arr[] = {1, 2, 3, 4, 5};
    span<int, 5> s{arr};
    auto ss = s.subspan(1, 2);

    static_assert(std::is_same<decltype(ss), span<int>>::value, "");
    CHECK_EQ(ss.size(), 2UL);
    CHECK_EQ(ss.data(), arr + 1);
    CHECK_EQ(ss.begin(), arr + 1);
    CHECK_EQ(ss.end(), arr + 1 + 2);
  }

  // TODO(tcbrindle): Test all the dynamic subspan possibilities
}

TEST(observers, span) {
  // We already use this everywhere, but whatever
  constexpr span<int, 0> empty{};
  static_assert(empty.size() == 0, "");
  static_assert(empty.empty(), "");

  constexpr int arr[] = {1, 2, 3};
  static_assert(span<const int>{arr}.size() == 3, "");
  static_assert(!span<const int>{arr}.empty(), "");
}

TEST(element_access, span) {
  constexpr int arr[] = {1, 2, 3};
  span<const int> s{arr};

  CHECK_EQ(s[0], arr[0]);
  CHECK_EQ(s[1], arr[1]);
  CHECK_EQ(s[2], arr[2]);
}

TEST(iterator, span) {
  {
    std::vector<int> vec;
    span<int> s{vec};
    std::sort(s.begin(), s.end());
    CHECK(std::is_sorted(vec.cbegin(), vec.cend()));
  }

  {
    const std::vector<int> vec{1, 2, 3};
    span<const int> s{vec};
    CHECK(std::equal(s.rbegin(), s.rend(), vec.crbegin()));
  }
}
