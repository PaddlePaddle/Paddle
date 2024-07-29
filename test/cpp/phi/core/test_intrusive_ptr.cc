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
  PADDLE_ENFORCE_EQ(
      obj.use_count(),
      1u,
      phi::errors::Fatal("Input object's using count is %u, not 1",
                         obj.use_count()));
}

TEST(intrusive_ptr, default_ctor) {
  intrusive_ptr<SharedObject> p;
  PADDLE_ENFORCE_EQ(
      p == nullptr, true, phi::errors::Fatal("Input pointer is not a nullptr"));
}
TEST(intrusive_ptr, private_ctor) {
  auto p = make_intrusive<SharedObject>();
  const auto* ptr0 = p.get();
  auto p1 = std::move(p);
  intrusive_ptr<intrusive_ref_counter<SharedObject>> p2(std::move(p1));
  const auto* ptr1 = p2.get();
  PADDLE_ENFORCE_EQ(
      ptr0,
      ptr1,
      phi::errors::Fatal(
          "p1 is not equal to p2, something wrong with move function"));
}

TEST(intrusive_ptr, reset_with_obj) {
  SharedObject obj;
  obj.i = 1;
  intrusive_ptr<SharedObject> p;
  p.reset(&obj, true);
  PADDLE_ENFORCE_EQ(
      p->i,
      obj.i,
      phi::errors::Fatal("p->i is not equal to obj.j, something wrong with "
                         "intrusive_ptr<T>.reset function or operator->"));
}

TEST(intrusive_ptr, reset_with_ptr) {
  auto* ptr = new SharedObject;
  ptr->i = 1;
  intrusive_ptr<SharedObject> p;
  p.reset(ptr, false);
  PADDLE_ENFORCE_EQ(
      (*p).i,
      ptr->i,
      phi::errors::Fatal("(*p).i is not equal to ptr->i, something wrong with "
                         "intrusive_ptr<T>.reset function or operator*"));
  p.reset();
  PADDLE_ENFORCE_EQ(
      p == nullptr,
      true,
      phi::errors::Fatal(
          "p is not a nullptr, something wrong with intrusive_ptr<T>.reset"));
}
TEST(intrusive_ptr, op_comp) {
  auto p = make_intrusive<SharedObject>();
  auto copy = copy_intrusive<SharedObject>(p);
  auto null = intrusive_ptr<SharedObject>();
  auto p1 = make_intrusive<SharedObject>();
  PADDLE_ENFORCE_EQ(
      p == copy,
      true,
      phi::errors::Fatal("intrusive_ptr p is not euqal to its copy, something "
                         "wrong with copy constructor "));
  PADDLE_ENFORCE_EQ(
      p != p1,
      true,
      phi::errors::Fatal("intrusive_ptr p is equal to another pointer, "
                         "something wrong with constructor"));
  PADDLE_ENFORCE_EQ(
      p == copy.get(),
      true,
      phi::errors::Fatal(
          "blank intrusive_ptr p's content is not equal to its copy, something "
          "wrong with constructor or get funtion"));
  PADDLE_ENFORCE_EQ(
      p != p1.get(),
      true,
      phi::errors::Fatal(
          "intrusive_ptr p's content is equal to another blank pointer, "
          "something wrong with constructor or get function"));
  PADDLE_ENFORCE_EQ(
      p.get() == copy,
      true,
      phi::errors::Fatal(
          "blank intrusive_ptr p's content is not equal to its copy, something "
          "wrong with constructor or get funtion"));
  PADDLE_ENFORCE_EQ(
      p.get() != p1,
      true,
      phi::errors::Fatal(
          "intrusive_ptr p's content is equal to another blank pointer, "
          "something wrong with constructor or get function"));
  PADDLE_ENFORCE_EQ(
      null == nullptr,
      true,
      phi::errors::Fatal("variable or constant whose name is null is not a "
                         "nullptr, something wrong with operator=="));
  PADDLE_ENFORCE_EQ(
      nullptr == null,
      true,
      phi::errors::Fatal("variable or constant whose name is null is not a "
                         "nullptr, something wrong with operator=="));
  PADDLE_ENFORCE_EQ(
      p != nullptr,
      true,
      phi::errors::Fatal("intrusive_ptr p is not not_equal to null, something "
                         "wrong with constructor or operator!= "));
  PADDLE_ENFORCE_EQ(
      nullptr != p,
      true,
      phi::errors::Fatal("intrusive_ptr p is not not_equal to null, something "
                         "wrong with constructor or operator!= "));
}
}  // namespace tests
}  // namespace phi
