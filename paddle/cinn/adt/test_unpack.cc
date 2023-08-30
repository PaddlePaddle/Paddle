// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <iostream>
#include <tuple>
#include <utility>

class Foo {
 public:
  Foo(int x, int y) : x_(x), y_(y) {}

  template <std::size_t I>
  const auto& get() const {
    if constexpr (I == 0) {
      return x_;
    } else if constexpr (I == 1) {
      return y_;
    }
  }

 private:
  int x_;
  int y_;
};

class Bar final : public Foo {
  using Foo::Foo;
};

namespace std {
template <>
struct tuple_size<Foo> : std::integral_constant<std::size_t, 2> {};

template <std::size_t I>
struct tuple_element<I, Foo> {
  using type = std::decay_t<decltype(std::declval<Foo>().get<I>())>;
};

}  // namespace std

int main() {
  const auto& [x, y] = Bar{30, 60};
  std::cout << "x = " << x << " y = " << y << std::endl;
  return 0;
}
