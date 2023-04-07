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

#pragma once

#include <random>
#include <type_traits>

namespace phi {
namespace tests {

template <typename T,
          typename =
              typename std::enable_if<std::is_arithmetic<T>::value>::type>
class RandomGenerator {
  using distribution_type =
      typename std::conditional<std::is_integral<T>::value,
                                std::uniform_int_distribution<T>,
                                std::uniform_real_distribution<T>>::type;

  std::default_random_engine engine;
  distribution_type distribution;

 public:
  auto operator()() -> decltype(distribution(engine)) {
    return distribution(engine);
  }
};

template <typename Container, typename T = typename Container::value_type>
auto make_generator(Container const&) -> decltype(RandomGenerator<T>()) {
  return RandomGenerator<T>();
}

}  // namespace tests
}  // namespace phi
