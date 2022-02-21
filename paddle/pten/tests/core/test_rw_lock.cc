/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/pten/core/utils/rw_lock.h"

#include <gtest/gtest.h>  // NOLINT
#include <thread>         // NOLINT

namespace pten {
namespace tests {

void f1(pten::RWLock *lock) {
  lock->RDLock();
  lock->UNLock();
}

TEST(RWLOCK, read_read) {
  pten::RWLock lock;
  lock.RDLock();
  std::thread t1(f1, &lock);
  std::thread t2(f1, &lock);
  t1.join();
  t2.join();
  lock.UNLock();
}

void f2(pten::RWLock *lock, std::vector<int> *result) {
  lock->RDLock();
  ASSERT_EQ(result->size(), 0UL);
  lock->UNLock();
}

void f3(pten::RWLock *lock, std::vector<int> *result) {
  lock->WRLock();
  result->push_back(1);
  lock->UNLock();
}

TEST(RWLOCK, read_write) {
  pten::RWLock lock;
  std::vector<int> result;

  lock.RDLock();
  std::thread t1(f2, &lock, &result);
  t1.join();
  std::thread t2(f3, &lock, &result);
  std::this_thread::sleep_for(std::chrono::seconds(1));
  ASSERT_EQ(result.size(), 0UL);
  lock.UNLock();
  t2.join();
  ASSERT_EQ(result.size(), 1UL);
}

void f4(pten::RWLock *lock, std::vector<int> *result) {
  lock->RDLock();
  ASSERT_EQ(result->size(), 1UL);
  lock->UNLock();
}

TEST(RWLOCK, write_read) {
  pten::RWLock lock;
  std::vector<int> result;

  lock.WRLock();
  std::thread t1(f4, &lock, &result);
  std::this_thread::sleep_for(std::chrono::seconds(1));
  result.push_back(1);
  lock.UNLock();
  t1.join();
}
}  // namespace tests
}  // namespace pten
