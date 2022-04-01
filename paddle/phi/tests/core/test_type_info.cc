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

#include "gtest/gtest.h"

#include "paddle/phi/core/utils/type_registry.h"

namespace phi {
namespace tests {

template <typename T>
class Base {
 public:
  TypeInfo<Base<T>> type_info() const { return type_info_; }

 private:
  template <typename T1, typename T2>
  friend class phi::TypeInfoTraits;
  TypeInfo<Base<T>> type_info_{TypeInfo<Base<T>>::kUnknownType};
};

template <typename T>
class DerivedA : public Base<T>, public TypeInfoTraits<Base<T>, DerivedA<T>> {
 public:
  static const char* name() { return "DerivedA"; }
};

template <typename T>
class DerivedB : public Base<T>, public TypeInfoTraits<Base<T>, DerivedB<T>> {
 public:
  static const char* name() { return "DerivedB"; }
};

template <typename T>
void check_type_info() {
  std::unique_ptr<Base<T>> base(new Base<T>);
  std::unique_ptr<Base<T>> derived_a(new DerivedA<T>);
  std::unique_ptr<Base<T>> derived_b(new DerivedB<T>);

  EXPECT_EQ(DerivedA<T>::classof(derived_a.get()), true);
  EXPECT_EQ(DerivedB<T>::classof(derived_b.get()), true);
  EXPECT_EQ(DerivedB<T>::classof(derived_a.get()), false);
  EXPECT_EQ(DerivedA<T>::classof(derived_b.get()), false);

  EXPECT_EQ(base->type_info().id(), 0);
  EXPECT_EQ(derived_a->type_info().id(), 1);
  EXPECT_EQ(derived_b->type_info().id(), 2);

  EXPECT_EQ(base->type_info().name(), "Unknown");
  EXPECT_EQ(derived_a->type_info().name(), "DerivedA");
  EXPECT_EQ(derived_b->type_info().name(), "DerivedB");
}

TEST(type_info, base) {
  check_type_info<int>();
  check_type_info<float>();
}

}  // namespace tests
}  // namespace phi
