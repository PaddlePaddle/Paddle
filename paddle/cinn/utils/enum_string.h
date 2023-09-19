// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#pragma once

namespace cinn {
namespace utils {

#include <array>
#include <string>
#include <string_view>
#include <utility>

template <typename E, E V>
constexpr auto PrettyName() {
  std::string_view name{__PRETTY_FUNCTION__, sizeof(__PRETTY_FUNCTION__) - 2};
  name.remove_prefix(name.find_last_of(" ") + 1);
  if (name.front() == '(') name.remove_prefix(name.size());
  return name;
}

template <typename E, E V>
constexpr bool IsValidEnum() {
  return !PrettyName<E, V>().empty();
}

template <int... Seq>
constexpr auto MakeIntegerSequence(std::integer_sequence<int, Seq...>) {
  return std::integer_sequence<int, (Seq)...>();
}

constexpr auto NormalIntegerSequence =
    MakeIntegerSequence(std::make_integer_sequence<int, 32>());

template <typename E, int... Seq>
constexpr size_t GetEnumSize(std::integer_sequence<int, Seq...>) {
  constexpr std::array<bool, sizeof...(Seq)> valid{
      IsValidEnum<E, static_cast<E>(Seq)>()...};
  constexpr std::size_t count =
      [](decltype((valid)) v) constexpr noexcept->std::size_t {
    auto cnt = std::size_t{0};
    for (auto b : v) {
      if (b) {
        ++cnt;
      }
    }
    return cnt;
  }
  (valid);
  return count;
}

template <typename E, int... Seq>
constexpr auto GetAllValidValues(std::integer_sequence<int, Seq...>) {
  constexpr std::size_t count = sizeof...(Seq);
  constexpr std::array<bool, count> valid{
      IsValidEnum<E, static_cast<E>(Seq)>()...};
  constexpr std::array<int, count> seq{Seq...};
  std::array<int, GetEnumSize<E>(NormalIntegerSequence)> values{};

  for (std::size_t i = 0, v = 0; i < count; ++i) {
    if (valid[i]) {
      values[v++] = seq[i];
    }
  }
  return values;
}

template <typename E, int... Seq>
constexpr auto GetAllValidNames(std::integer_sequence<int, Seq...>) {
  constexpr std::array<std::string_view, sizeof...(Seq)> names{
      PrettyName<E, static_cast<E>(Seq)>()...};
  std::array<std::string_view, GetEnumSize<E>(NormalIntegerSequence)>
      valid_names{};

  for (std::size_t i = 0, v = 0; i < names.size(); ++i) {
    if (!names[i].empty()) {
      valid_names[v++] = names[i];
    }
  }
  return valid_names;
}

template <typename E>
constexpr std::string_view Enum2String(E V) {
  constexpr auto names = GetAllValidNames<E>(NormalIntegerSequence);
  constexpr auto values = GetAllValidValues<E>(NormalIntegerSequence);
  constexpr auto size = GetEnumSize<E>(NormalIntegerSequence);

  for (size_t i = 0; i < size; ++i) {
    if (static_cast<int>(V) == values[i]) {
      return names[i];
    }
  }
  return std::to_string(static_cast<int>(V));
}

}  // namespace utils
}  // namespace cinn
