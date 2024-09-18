//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/details/exception_holder.h"

#include "gtest/gtest.h"
#include "paddle/phi/core/memory/allocation/allocator.h"

namespace paddle {
namespace framework {
namespace details {

TEST(ExceptionHolderTester, TestEnforceNotMetCatch) {
  ExceptionHolder exception_holder;

  try {
    throw platform::EnforceNotMet("enforce not met test", "test_file", 0);
  } catch (...) {
    exception_holder.Catch(std::current_exception());
  }
  ASSERT_TRUE(exception_holder.IsCaught());
  ASSERT_EQ(exception_holder.Type(), "EnforceNotMet");

  bool catch_enforce_not_met = false;
  try {
    exception_holder.ReThrow();
  } catch (platform::EnforceNotMet& ex) {
    catch_enforce_not_met = true;
  } catch (...) {
    catch_enforce_not_met = false;
  }

  ASSERT_TRUE(catch_enforce_not_met);
}

TEST(ExceptionHolderTester, TestBadAllocCatch) {
  ExceptionHolder exception_holder;

  try {
    throw memory::allocation::BadAlloc("bad alloc test", "test_file", 0);
  } catch (...) {
    exception_holder.Catch(std::current_exception());
  }
  ASSERT_TRUE(exception_holder.IsCaught());
  ASSERT_EQ(exception_holder.Type(), "BadAlloc");

  bool catch_bad_alloc = false;
  try {
    exception_holder.ReThrow();
  } catch (memory::allocation::BadAlloc& ex) {
    catch_bad_alloc = true;
  } catch (...) {
    catch_bad_alloc = false;
  }

  ASSERT_TRUE(catch_bad_alloc);
}

TEST(ExceptionHolderTester, TestBaseExceptionCatch) {
  ExceptionHolder exception_holder;

  try {
    throw std::exception();
  } catch (...) {
    exception_holder.Catch(std::current_exception());
  }
  ASSERT_TRUE(exception_holder.IsCaught());
  ASSERT_EQ(exception_holder.Type(), "BaseException");

  bool catch_base_exception = false;
  try {
    exception_holder.ReThrow();
  } catch (std::exception& ex) {
    catch_base_exception = true;
  } catch (...) {
    catch_base_exception = false;
  }

  ASSERT_TRUE(catch_base_exception);
}

TEST(ExceptionHolderTester, TestExceptionReplace) {
  ExceptionHolder exception_holder;

  try {
    throw platform::EnforceNotMet("enforce not met test", "test_file", 0);
  } catch (...) {
    exception_holder.Catch(std::current_exception());
  }
  ASSERT_TRUE(exception_holder.IsCaught());
  ASSERT_EQ(exception_holder.Type(), "EnforceNotMet");

  try {
    throw std::exception();
  } catch (...) {
    exception_holder.Catch(std::current_exception());
  }
  ASSERT_TRUE(exception_holder.IsCaught());
  ASSERT_EQ(exception_holder.Type(), "EnforceNotMet");

  try {
    throw memory::allocation::BadAlloc("bad alloc test", "test_file", 0);
  } catch (...) {
    exception_holder.Catch(std::current_exception());
  }
  ASSERT_TRUE(exception_holder.IsCaught());
  ASSERT_EQ(exception_holder.Type(), "EnforceNotMet");

  try {
    throw platform::EOFException("eof test", "test_file", 0);
  } catch (...) {
    exception_holder.Catch(std::current_exception());
  }
  ASSERT_EQ(exception_holder.Type(), "EnforceNotMet");

  exception_holder.Clear();

  try {
    throw memory::allocation::BadAlloc("bad alloc test", "test_file", 0);
  } catch (...) {
    exception_holder.Catch(std::current_exception());
  }
  ASSERT_TRUE(exception_holder.IsCaught());
  ASSERT_EQ(exception_holder.Type(), "BadAlloc");

  try {
    throw platform::EnforceNotMet("enforce not met test", "test_file", 0);
  } catch (...) {
    exception_holder.Catch(std::current_exception());
  }
  ASSERT_TRUE(exception_holder.IsCaught());
  ASSERT_EQ(exception_holder.Type(), "BadAlloc");
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
