/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/scope.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <vector>
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/fluid/platform/device_context.h"

using paddle::framework::Scope;
using paddle::framework::Variable;

TEST(Scope, VarsShadowing) {
  Scope s;
  Scope& ss1 = s.NewScope();
  Scope& ss2 = s.NewScope();

  Variable* v0 = s.Var("a");
  Variable* v1 = ss1.Var("a");

  EXPECT_NE(v0, v1);

  EXPECT_EQ(v0, s.FindVar("a"));
  EXPECT_EQ(v1, ss1.FindVar("a"));
  EXPECT_EQ(v0, ss2.FindVar("a"));
}

TEST(Scope, FindVar) {
  Scope s;
  Scope& ss = s.NewScope();

  EXPECT_EQ(nullptr, s.FindVar("a"));
  EXPECT_EQ(nullptr, ss.FindVar("a"));

  ss.Var("a");

  EXPECT_EQ(nullptr, s.FindVar("a"));
  EXPECT_NE(nullptr, ss.FindVar("a"));
}

TEST(Scope, FindScope) {
  Scope s;
  Scope& ss = s.NewScope();
  Variable* v = s.Var("a");

  EXPECT_EQ(&s, s.FindScope(v));
  EXPECT_EQ(&s, ss.FindScope(v));
}

TEST(Scope, GetAllNames) {
  Scope s;
  Variable* v = s.Var("a");
  EXPECT_EQ(&s, s.FindScope(v));

  std::vector<std::string> ans = s.LocalVarNames();
  std::string str;
  for (auto& var : ans) {
    str += var;
  }

  EXPECT_STREQ("a", str.c_str());
}

// to mimic the op running, generate the difference sized string
// and have a test.
class PerThreadData {
 public:
  PerThreadData(const int& max_length, const int size)
      : max_length_(max_length), size_(size) {
    items_.reserve(size_);
    // fill vector with dynamic sized string.
    std::mt19937 gen;
    std::uniform_int_distribution<int> dist;
    for (int i = 0; i < size_; ++i) {
      int word_length = rand_r_r() % max_length + 1;
      std::string ret;
      std::generate_n(std::back_inserter(ret), word_length,
                      [&]() { return 'a' + dist(gen); });
      items_.emplace_back(std::move(ret));
    }
  }

  std::string operator[](const int& idx) {
    PADDLE_ENFORCE(idx >= 0 && idx < size_);
    return items_[idx];
  }

  int size() { return size_; }

 private:
  int max_length_;
  int size_;
  std::vector<std::string> items_;
};

template <typename T>
double GetTimeDiff(const T& start) {
  auto startu = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_span =
      std::chrono::duration_cast<std::chrono::duration<double>>(startu - start);
  double used_time_ms = static_cast<double>(time_span.count()) * 1000.0;
  return used_time_ms;
}

// single thread, max word length = 5
TEST(Scope, ConcurrentScopeTest) {
  Scope scope;
  int max_name_length = 5, item_count = 100000;
  PerThreadData data(max_name_length, item_count);

  auto s1 = std::chrono::system_clock::now();
  for (int i = 0; i < data.size(); ++i) {
    if (rand_r_r() % 3 == 0) {
      scope.Var(data[i]);
    } else {
      scope.FindVar(data[i]);
    }
  }
  std::cout << "max_name_length : " << max_name_length << "\n"
            << "item count : " << item_count << "\n"
            << "Time : " << GetTimeDiff(s1) << "ms" << std::endl;
}

TEST(Scope, ConcurrentScopeTest2) {
  Scope scope;
  int max_name_length = 20, item_count = 100000;
  PerThreadData data(max_name_length, item_count);

  auto s1 = std::chrono::system_clock::now();
  for (int i = 0; i < data.size(); ++i) {
    if (rand_r_r() % 3 == 0) {
      scope.Var(data[i]);
    } else {
      scope.FindVar(data[i]);
    }
  }
  std::cout << "max_name_length : " << max_name_length << "\n"
            << "item count : " << item_count << "\n"
            << "Time : " << GetTimeDiff(s1) << "ms" << std::endl;
}

// multi thread max word length = 20
TEST(Scope, ConcurrentScopeTest3) {
  int max_name_length = 20, item_count = 100000;
  constexpr int kThreadNum = 10;

  Scope scope;
  PerThreadData data(max_name_length, item_count);
  std::vector<std::thread> threads;

  auto s1 = std::chrono::system_clock::now();
  for (int tid = 0; tid < kThreadNum; ++tid) {
    threads.emplace_back(std::thread([&]() {
      for (int i = 0; i < data.size(); ++i) {
        if (rand_r_r() % 3 == 0) {
          scope.Var(data[i]);
        } else {
          scope.FindVar(data[i]);
        }
      }
    }));
  }
  for (auto& th : threads) {
    th.join();
  }
  std::cout << "max_name_length : " << max_name_length << "\n"
            << "item count : " << item_count << "\n"
            << "Time : " << GetTimeDiff(s1) << "ms" << std::endl;
}
