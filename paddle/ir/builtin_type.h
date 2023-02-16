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

#pragma once

#include "paddle/ir/type.h"

namespace ir {
///
/// \brief Interfaces for user-created built-in types. For example:
/// Type fp32 = Float32Type::get(ctx);
///
class Float32Type : public ir::Type {
 public:
  REGISTER_TYPE_UTILS(Float32Type, ir::TypeStorage);

  static Float32Type get(ir::IrContext *context);
};

struct IntegerTypeStorage : public TypeStorage {
  IntegerTypeStorage(unsigned width, unsigned signedness)
      : width_(width), signedness_(signedness) {}
  using ParamKey = std::pair<unsigned, unsigned>;

  static std::size_t HashValue(const ParamKey &key) {
    return hash_combine(std::hash<unsigned>()(std::get<0>(key)),
                        std::hash<unsigned>()(std::get<1>(key)));
  }

  bool operator==(const ParamKey &key) const {
    return ParamKey(width_, signedness_) == key;
  }

  static IntegerTypeStorage *Construct(ParamKey key) {
    return new IntegerTypeStorage(key.first, key.second);
  }

  ParamKey GetAsKey() const { return ParamKey(width_, signedness_); }

  unsigned width_ : 30;
  unsigned signedness_ : 2;

 private:
  static std::size_t hash_combine(std::size_t lhs, std::size_t rhs) {
    return lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
  }
};

class IntegerType : public ir::Type {
 public:
  REGISTER_TYPE_UTILS(IntegerType, ir::IntegerTypeStorage);

  static IntegerType get(ir::IrContext *context,
                         unsigned width,
                         unsigned signedness = 0);
};

}  // namespace ir
