// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <unordered_map>
#include "paddle/ap/include/axpr/adt.h"
#include "paddle/ap/include/axpr/binary_func.h"
#include "paddle/ap/include/axpr/type.h"
#include "paddle/ap/include/axpr/unary_func.h"

namespace ap::axpr {

inline constexpr const char* kBuiltinIf() { return "__builtin_if__"; }
inline constexpr const char* kBuiltinIdentity() {
  return "__builtin_identity__";
}
inline constexpr const char* kBuiltinList() { return "__builtin_list__"; }
inline constexpr const char* kBuiltinStarred() { return "__builtin_starred__"; }
inline constexpr const char* kBuiltinCall() { return "__builtin_call__"; }
inline constexpr const char* kBuiltinToString() {
  return "__builtin_ToString__";
}
inline constexpr const char* kBuiltinHash() { return "__builtin_hash__"; }
inline constexpr const char* kBuiltinGetAttr() { return "__builtin_getattr__"; }
inline constexpr const char* kBuiltinSetAttr() { return "__builtin_setattr__"; }
inline constexpr const char* kBuiltinGetItem() { return "__builtin_getitem__"; }
inline constexpr const char* kBuiltinSetItem() { return "__builtin_setitem__"; }
inline constexpr const char* kBuiltinLength() { return "__builtin_len__"; }
inline constexpr const char* kBuiltinReturn() { return "__builtin_return__"; }

#define DEFINE_PEXPR_BUILTIN_CONSTANT_NAME(name, op) \
  inline constexpr const char* kBuiltin##name() {    \
    return "__builtin_" #name "__";                  \
  }
PEXPR_FOR_EACH_BINARY_OP(DEFINE_PEXPR_BUILTIN_CONSTANT_NAME)
PEXPR_FOR_EACH_UNARY_OP(DEFINE_PEXPR_BUILTIN_CONSTANT_NAME)
#undef DEFINE_PEXPR_BUILTIN_CONSTANT_NAME

namespace builtin_symbol {

struct If : public std::monostate {
  using std::monostate::monostate;
  static constexpr const char* Name() { return kBuiltinIf(); }
  std::size_t GetHashValue() const { return 0; }
};

struct Id : public std::monostate {
  using std::monostate::monostate;
  static constexpr const char* Name() { return kBuiltinIdentity(); }
  std::size_t GetHashValue() const { return 0; }
};

struct List : public std::monostate {
  using std::monostate::monostate;
  static constexpr const char* Name() { return kBuiltinList(); }
  std::size_t GetHashValue() const { return 0; }
};

struct Starred : public std::monostate {
  using std::monostate::monostate;
  static constexpr const char* Name() { return kBuiltinStarred(); }
  static constexpr int num_operands = 1;
  std::size_t GetHashValue() const { return 0; }
};

struct Call : public std::monostate {
  using std::monostate::monostate;
  static constexpr const char* Name() { return kBuiltinCall(); }
  static constexpr int num_operands = 1;
  std::size_t GetHashValue() const { return 0; }
};

struct ToString : public std::monostate {
  using std::monostate::monostate;
  static constexpr const char* Name() { return kBuiltinToString(); }
  static constexpr int num_operands = 1;
  std::size_t GetHashValue() const { return 0; }
};

struct Hash : public std::monostate {
  using std::monostate::monostate;
  static constexpr const char* Name() { return kBuiltinHash(); }
  static constexpr int num_operands = 1;
  std::size_t GetHashValue() const { return 0; }
};

struct GetAttr : public std::monostate {
  using std::monostate::monostate;
  static constexpr const char* Name() { return kBuiltinGetAttr(); }
  static constexpr int num_operands = 2;
  std::size_t GetHashValue() const { return 0; }
};

struct SetAttr : public std::monostate {
  using std::monostate::monostate;
  static constexpr const char* Name() { return kBuiltinSetAttr(); }
  static constexpr int num_operands = 2;
  std::size_t GetHashValue() const { return 0; }
};

struct GetItem : public std::monostate {
  using std::monostate::monostate;
  static constexpr const char* Name() { return kBuiltinGetItem(); }
  static constexpr int num_operands = 2;
  std::size_t GetHashValue() const { return 0; }
};

struct SetItem : public std::monostate {
  using std::monostate::monostate;
  static constexpr const char* Name() { return kBuiltinSetItem(); }
  static constexpr int num_operands = 2;
  std::size_t GetHashValue() const { return 0; }
};

struct Length : public std::monostate {
  using std::monostate::monostate;
  static constexpr const char* Name() { return kBuiltinLength(); }
  static constexpr int num_operands = 1;
  std::size_t GetHashValue() const { return 0; }
};

#define DEFINE_UNARY_SYMBOL(name, op)                                \
  struct name : public std::monostate {                              \
    using std::monostate::monostate;                                 \
    static constexpr const char* Name() { return kBuiltin##name(); } \
    static constexpr int num_operands = 1;                           \
    std::size_t GetHashValue() const { return 0; }                   \
  };

PEXPR_FOR_EACH_UNARY_OP(DEFINE_UNARY_SYMBOL);

#undef DEFINE_UNARY_SYMBOL

#define DEFINE_BINARY_SYMBOL(name, op)                               \
  struct name : public std::monostate {                              \
    using std::monostate::monostate;                                 \
    static constexpr const char* Name() { return kBuiltin##name(); } \
    static constexpr int num_operands = 2;                           \
    std::size_t GetHashValue() const { return 0; }                   \
  };

PEXPR_FOR_EACH_BINARY_OP(DEFINE_BINARY_SYMBOL);

#undef DEFINE_BINARY_SYMBOL

#define AXPR_FOR_EACH_SYMBOL_OP(_) \
  PEXPR_FOR_EACH_BINARY_OP(_)      \
  PEXPR_FOR_EACH_UNARY_OP(_)       \
  _(Call, ())                      \
  _(ToString, str)                 \
  _(Hash, hash)                    \
  _(Starred, *)                    \
  _(GetAttr, .)                    \
  _(SetAttr, .)                    \
  _(GetItem, [])                   \
  _(SetItem, [])                   \
  _(Length, len)

using OpImpl = std::variant<
#define MAKE_OP_IMPL_ALTERNATIVE(name, op) name,
    PEXPR_FOR_EACH_BINARY_OP(MAKE_OP_IMPL_ALTERNATIVE)
        PEXPR_FOR_EACH_UNARY_OP(MAKE_OP_IMPL_ALTERNATIVE)
#undef MAKE_OP_IMPL_ALTERNATIVE
            Call,
    ToString,
    Hash,
    Starred,
    GetAttr,
    SetAttr,
    GetItem,
    SetItem,
    Length>;

struct Op : public OpImpl {
  using OpImpl::OpImpl;
  ADT_DEFINE_VARIANT_METHODS(OpImpl);

  const char* Name() const {
    return Match([](const auto& impl) { return impl.Name(); });
  }

  std::size_t GetHashValue() const {
    std::size_t hash_value =
        Match([&](const auto& impl) { return impl.GetHashValue(); });
    return adt::hash_combine(hash_value, this->index());
  }
};

using SymbolImpl = std::variant<If, Id, List, Op>;

struct Symbol : public SymbolImpl {
  using SymbolImpl::SymbolImpl;
  ADT_DEFINE_VARIANT_METHODS(SymbolImpl);

  const char* Name() const {
    return Match([](const auto& impl) { return impl.Name(); });
  }

  std::size_t GetHashValue() const {
    std::size_t hash_value =
        Match([&](const auto& impl) { return impl.GetHashValue(); });
    return adt::hash_combine(hash_value, this->index());
  }
};

inline adt::Maybe<Symbol> GetSymbolFromString(const std::string& name) {
  static const std::unordered_map<std::string, Symbol> map{
      {If::Name(), If{}},
      {Id::Name(), Id{}},
      {List::Name(), List{}},
      {Call::Name(), Op{Call{}}},
      {ToString::Name(), Op{ToString{}}},
      {Hash::Name(), Op{Hash{}}},
      {Starred::Name(), Op{Starred{}}},
      {GetAttr::Name(), Op{GetAttr{}}},
      {SetAttr::Name(), Op{SetAttr{}}},
      {GetItem::Name(), Op{GetItem{}}},
      {SetItem::Name(), Op{SetItem{}}},
      {Length::Name(), Op{Length{}}},
#define MAKE_SYMBOL_ENTRY(cls, op) {cls::Name(), Op{cls{}}},
      PEXPR_FOR_EACH_BINARY_OP(MAKE_SYMBOL_ENTRY)
          PEXPR_FOR_EACH_UNARY_OP(MAKE_SYMBOL_ENTRY)
#undef MAKE_SYMBOL_ENTRY
  };
  const auto& iter = map.find(name);
  if (iter == map.end()) {
    return adt::Nothing{};
  }
  return iter->second;
}

}  // namespace builtin_symbol

template <typename BuiltinSymbol>
struct ConvertBuiltinSymbolToArithmetic {
  static const bool convertible = false;
  using arithmetic_op_type = void;
};

#define SPECIALIZE_ConvertBuiltinSymbolToArithmetic(cls, op)     \
  template <>                                                    \
  struct ConvertBuiltinSymbolToArithmetic<builtin_symbol::cls> { \
    static const bool convertible = true;                        \
    using arithmetic_op_type = Arithmetic##cls;                  \
  };

PEXPR_FOR_EACH_BINARY_OP(SPECIALIZE_ConvertBuiltinSymbolToArithmetic);
PEXPR_FOR_EACH_UNARY_OP(SPECIALIZE_ConvertBuiltinSymbolToArithmetic);
#undef SPECIALIZE_ConvertBuiltinSymbolToArithmetic

template <typename BuiltinSymbol>
constexpr const char* GetBuiltinSymbolDebugString() {
  if constexpr (ConvertBuiltinSymbolToArithmetic<BuiltinSymbol>::convertible) {
    return ConvertBuiltinSymbolToArithmetic<
        BuiltinSymbol>::arithmetic_op_type::Name();
  } else {
    return BuiltinSymbol::Name();
  }
}

template <>
struct TypeImpl<builtin_symbol::Symbol> : public std::monostate {
  using value_type = builtin_symbol::Symbol;

  const char* Name() const { return "builtin_symbol"; }
};

}  // namespace ap::axpr
