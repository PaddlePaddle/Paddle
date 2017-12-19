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

#include "threadpool.h"
#include <gtest/gtest.h>
#include <chrono>
#include <map>
#include <thread>

namespace framework = paddle::framework;

TEST(ThreadPool, Start) {
  size_t num_threads = 4;
  framework::ThreadPool* pool = framework::ThreadPool::Instance(num_threads);
  std::map<int, bool> dict;
  int sum = 0;
  for (int i = 0; i < 10; ++i) {
    pool->Run([&sum]() { sum++; });
  }
  pool->Wait();
  EXPECT_EQ(sum, 10);
}