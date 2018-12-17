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
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <vector>
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/profiler.h"

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

using paddle::platform::CPUDeviceContext;
using paddle::platform::RecordEvent;
using paddle::platform::EventSortingKey;

constexpr int kItemCount = 100000;
constexpr int kCharSet = 26;
class ConcurrentScopeTest {
 public:
  explicit ConcurrentScopeTest(int n) : nthreads_(n), dist_(0, kCharSet) {
    gen_.seed(1331);
    vars_.reserve(kItemCount);
    paddle::platform::EnableProfiler(paddle::platform::ProfilerState::kCPU);
  }
  ~ConcurrentScopeTest() {
    paddle::platform::DisableProfiler(EventSortingKey::kTotal,
                                      "/tmp/scope_profiler");
  }

  void PerThreadProducer(Scope* scope, const int& thread_id) {
    thread_local int offset = thread_id * (kItemCount / nthreads_);
    std::cout << offset << std::endl;
    std::cout << thread_id << std::endl;
    for (int i = 0; i < (kItemCount / nthreads_); ++i) {
      vars_[offset + i] = RandomString();
      if (i % 2 == 0) {
        RecordEvent("Producer", &dev_ctx_);
        scope->Var(vars_[offset + i]);
      }
    }
  }

  void PerThreadConsumer(Scope* scope, const int& thread_id) {
    int offset = thread_id * (kItemCount / nthreads_);
    for (int i = 0; i < (kItemCount / nthreads_); ++i) {
      Variable* var = nullptr;
      {
        RecordEvent("Consumer", &dev_ctx_);
        var = scope->FindVar(vars_[offset + i]);
      }
      if (i % 2 == 0) {
        PADDLE_ENFORCE(var != nullptr);
      } else {
        PADDLE_ENFORCE(var == nullptr);
      }
    }
  }

 private:
  std::string RandomString(const size_t& n = 5) {
    std::string ret;
    std::generate_n(std::back_inserter(ret), n,
                    [&]() { return 'a' + dist_(gen_); });
    return ret;
  }

  int nthreads_;
  std::mt19937 gen_;
  std::uniform_int_distribution<size_t> dist_;
  std::vector<std::string> vars_;
  paddle::platform::CPUDeviceContext dev_ctx_;
};

template <typename T>
double GetTimeDiff(const T& start) {
  auto startu = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_span =
      std::chrono::duration_cast<std::chrono::duration<double>>(startu - start);
  double used_time_ms = static_cast<double>(time_span.count()) * 1000.0;
  return used_time_ms;
}

TEST(Scope, Concurrent) {
  Scope scope;
  const int kThreadNum = 10;
  ConcurrentScopeTest mock(kThreadNum);
  std::vector<std::thread> producer;
  std::vector<std::thread> consumer;

  auto s1 = std::chrono::system_clock::now();
  for (int i = 0; i < kThreadNum; ++i) {
    producer.emplace_back(std::thread(
        [&scope, &mock, &i]() { mock.PerThreadProducer(&scope, i); }));
  }
  for (auto& t : producer) {
    t.join();
  }
  std::cout << "Producer Time : " << GetTimeDiff(s1) << "ms" << std::endl;
  auto s2 = std::chrono::system_clock::now();
  for (int i = 0; i < kThreadNum; ++i) {
    consumer.emplace_back(std::thread(
        [&scope, &mock, &i]() { mock.PerThreadConsumer(&scope, i); }));
  }
  for (auto& t : consumer) {
    t.join();
  }
  std::cout << "Consumer Time : " << GetTimeDiff(s2) << "ms" << std::endl;
}
