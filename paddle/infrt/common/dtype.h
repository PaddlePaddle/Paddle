// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>

namespace infrt {
class DType {
 public:
  enum class Kind : uint8_t {
    Unk = 0,

// Automatically generate the enum definition
#define INFRT_DTYPE(enum__, value__) enum__ = value__,
#include "paddle/infrt/common/dtype.def"
#undef INFRT_DTYPE

    BOOL = I1,
  };

  DType() = default;
  explicit constexpr DType(Kind kind) : kind_(kind) { assert(IsValid()); }

  DType(const DType&) = default;
  DType& operator=(const DType&) = default;
  bool operator==(DType other) const { return kind_ == other.kind_; }
  bool operator!=(DType other) const { return !(*this == other); }

  constexpr Kind kind() const { return kind_; }

  bool IsValid() const { return kind_ != Kind::Unk; }
  bool IsInvalid() const { return !IsValid(); }

  const char* name() const;

  size_t GetHostSize() const;

 private:
  Kind kind_{Kind::Unk};
};

template <typename T>
constexpr DType GetDType();

template <DType::Kind kind>
struct DTypeInternal;

#define INFRT_IMPL_GET_DTYPE(cpp_type__, enum__)  \
  template <>                                     \
  inline constexpr DType GetDType<cpp_type__>() { \
    return DType{DType::Kind::enum__};            \
  }                                               \
  template <>                                     \
  struct DTypeInternal<DType::Kind::enum__> {     \
    using type = cpp_type__;                      \
  };

INFRT_IMPL_GET_DTYPE(bool, I1);
INFRT_IMPL_GET_DTYPE(int8_t, I8);
INFRT_IMPL_GET_DTYPE(int16_t, I16);
INFRT_IMPL_GET_DTYPE(int32_t, I32);
INFRT_IMPL_GET_DTYPE(int64_t, I64);
INFRT_IMPL_GET_DTYPE(uint8_t, UI8);
INFRT_IMPL_GET_DTYPE(uint16_t, UI16);
INFRT_IMPL_GET_DTYPE(uint32_t, UI32);
INFRT_IMPL_GET_DTYPE(uint64_t, UI64);
INFRT_IMPL_GET_DTYPE(float, F32);
INFRT_IMPL_GET_DTYPE(double, F64);
INFRT_IMPL_GET_DTYPE(std::string, STRING);

}  // namespace infrt
