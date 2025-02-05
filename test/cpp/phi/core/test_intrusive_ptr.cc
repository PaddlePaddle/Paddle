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

#include <future>
#include <vector>

#include "gtest/gtest.h"
#include "paddle/phi/core/utils/intrusive_ptr.h"
#include "paddle/phi/core/utils/intrusive_ref_counter.h"

namespace phi {
namespace tests {

struct SharedObject : public intrusive_ref_counter<SharedObject> {
  int i{0};
};

TEST(intrusive_ref_counter, async) {
  SharedObject obj;
  const size_t num{100};
  std::vector<std::future<void>> results;
  auto add_ref_and_release = [](const SharedObject* p) {
    intrusive_ptr_add_ref<SharedObject>(p);
    intrusive_ptr_release<SharedObject>(p);
  };
  for (size_t i = 0; i < num; ++i) {
    results.emplace_back(std::async(add_ref_and_release, &obj));
  }
  for (auto& result : results) {
    result.get();
  }
  PADDLE_ENFORCE_EQ(obj.use_count(),
                    1U,
                    common::errors::InvalidArgument(
                        "Required obj.use_count() should be equal to 1, "
                        "But received obj.use_count() = %d.",
                        obj.use_count()));
}

TEST(intrusive_ptr, default_ctor) {
  intrusive_ptr<SharedObject> p;
  PADDLE_ENFORCE_EQ(p == nullptr,
                    true,
                    common::errors::Fatal("Input pointer is not a nullptr"));
}
TEST(intrusive_ptr, private_ctor) {
  auto p = make_intrusive<SharedObject>();
  const auto* ptr0 = p.get();
  auto p1 = std::move(p);
  intrusive_ptr<intrusive_ref_counter<SharedObject>> p2(std::move(p1));
  const auto* ptr1 = p2.get();
  PADDLE_ENFORCE_EQ(ptr0,
                    ptr1,
                    common::errors::InvalidArgument(
                        "Required ptr0 should be equal to ptr1. "));
}

TEST(intrusive_ptr, reset_with_obj) {
  SharedObject obj;
  obj.i = 1;
  intrusive_ptr<SharedObject> p;
  p.reset(&obj, true);
  PADDLE_ENFORCE_EQ(p->i,
                    obj.i,
                    common::errors::InvalidArgument(
                        "Required p->i should be equal to obj.i. "));
}

TEST(intrusive_ptr, reset_with_ptr) {
  auto* ptr = new SharedObject;
  ptr->i = 1;
  intrusive_ptr<SharedObject> p;
  p.reset(ptr, false);
  PADDLE_ENFORCE_EQ((*p).i,
                    ptr->i,
                    common::errors::InvalidArgument(
                        "Required (*p).i should be equal to ptr->i. "));
  p.reset();
  PADDLE_ENFORCE_EQ(
      p == nullptr,
      true,
      common::errors::Fatal(
          "p is not a nullptr, something wrong with intrusive_ptr<T>.reset"));
}
TEST(intrusive_ptr, op_comp) {
  auto p = make_intrusive<SharedObject>();
  auto copy = copy_intrusive<SharedObject>(p);
  auto null = intrusive_ptr<SharedObject>();
  auto p1 = make_intrusive<SharedObject>();
  PADDLE_ENFORCE_EQ(p == copy,
                    true,
                    common::errors::Fatal(
                        "intrusive_ptr p is not euqal to its copy, something "
                        "wrong with copy constructor "));
  PADDLE_ENFORCE_EQ(
      p != p1,
      true,
      common::errors::Fatal("intrusive_ptr p is equal to another pointer, "
                            "something wrong with constructor"));
  PADDLE_ENFORCE_EQ(
      p == copy.get(),
      true,
      common::errors::Fatal(
          "blank intrusive_ptr p's content is not equal to its copy, something "
          "wrong with constructor or get function"));
  PADDLE_ENFORCE_EQ(
      p != p1.get(),
      true,
      common::errors::Fatal(
          "intrusive_ptr p's content is equal to another blank pointer, "
          "something wrong with constructor or get function"));
  PADDLE_ENFORCE_EQ(
      p.get() == copy,
      true,
      common::errors::Fatal(
          "blank intrusive_ptr p's content is not equal to its copy, something "
          "wrong with constructor or get function"));
  PADDLE_ENFORCE_EQ(
      p.get() != p1,
      true,
      common::errors::Fatal(
          "intrusive_ptr p's content is equal to another blank pointer, "
          "something wrong with constructor or get function"));
  PADDLE_ENFORCE_EQ(
      null == nullptr,
      true,
      common::errors::Fatal("variable or constant whose name is null is not a "
                            "nullptr, something wrong with operator=="));
  PADDLE_ENFORCE_EQ(
      nullptr == null,
      true,
      common::errors::Fatal("variable or constant whose name is null is not a "
                            "nullptr, something wrong with operator=="));
  PADDLE_ENFORCE_EQ(p != nullptr,
                    true,
                    common::errors::Fatal(
                        "intrusive_ptr p is not not_equal to null, something "
                        "wrong with constructor or operator!= "));
  PADDLE_ENFORCE_EQ(nullptr != p,
                    true,
                    common::errors::Fatal(
                        "intrusive_ptr p is not not_equal to null, something "
                        "wrong with constructor or operator!= "));
}
}  // namespace tests
}  // namespace phi
