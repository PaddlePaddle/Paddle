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

#include "paddle/fluid/framework/data_type.h"
#include <string>
#include <unordered_map>

using float16 = paddle::platform::float16;
using bfloat16 = paddle::platform::bfloat16;

namespace paddle {
namespace framework {

struct DataTypeMap {
  std::unordered_map<std::type_index, proto::VarType::Type> cpp_to_proto_;
  std::unordered_map<int, std::type_index> proto_to_cpp_;
  std::unordered_map<int, std::string> proto_to_str_;
  std::unordered_map<int, size_t> proto_to_size_;
};

static DataTypeMap* InitDataTypeMap();
// C++11 removes the need for manual locking. Concurrent execution shall wait if
// a static local variable is already being initialized.
// https://stackoverflow.com/questions/11711920/how-to-implement-multithread-safe-singleton-in-c11-without-using-mutex
static DataTypeMap& gDataTypeMap() {
  static DataTypeMap* g_data_type_map_ = InitDataTypeMap();
  return *g_data_type_map_;
}

template <typename T>
static inline void RegisterType(DataTypeMap* map,
                                proto::VarType::Type proto_type,
                                const std::string& name) {
  map->proto_to_cpp_.emplace(static_cast<int>(proto_type), typeid(T));
  map->cpp_to_proto_.emplace(typeid(T), proto_type);
  map->proto_to_str_.emplace(static_cast<int>(proto_type), name);
  map->proto_to_size_.emplace(static_cast<int>(proto_type), sizeof(T));
}

static DataTypeMap* InitDataTypeMap() {
  auto retv = new DataTypeMap();

#define RegType(cc_type, proto_type) \
  RegisterType<cc_type>(retv, proto_type, #cc_type)

  _ForEachDataType_(RegType);

#undef RegType
  return retv;
}

proto::VarType::Type ToDataType(std::type_index type) {
  auto it = gDataTypeMap().cpp_to_proto_.find(type);
  if (it != gDataTypeMap().cpp_to_proto_.end()) {
    return it->second;
  }
  PADDLE_THROW(platform::errors::Unimplemented(
      "Not support %s as tensor data type.", platform::demangle(type.name())));
}

std::type_index ToTypeIndex(proto::VarType::Type type) {
  auto it = gDataTypeMap().proto_to_cpp_.find(static_cast<int>(type));
  if (it != gDataTypeMap().proto_to_cpp_.end()) {
    return it->second;
  }
  PADDLE_THROW(platform::errors::Unimplemented(
      "Not support proto::VarType::Type(%d) as tensor type.",
      static_cast<int>(type)));
}

std::string DataTypeToString(const proto::VarType::Type type) {
  auto it = gDataTypeMap().proto_to_str_.find(static_cast<int>(type));
  if (it != gDataTypeMap().proto_to_str_.end()) {
    return it->second;
  }
  PADDLE_THROW(platform::errors::Unimplemented(
      "Not support proto::VarType::Type(%d) as tensor type.",
      static_cast<int>(type)));
}

size_t SizeOfType(proto::VarType::Type type) {
  auto it = gDataTypeMap().proto_to_size_.find(static_cast<int>(type));
  if (it != gDataTypeMap().proto_to_size_.end()) {
    return it->second;
  }
  PADDLE_THROW(platform::errors::Unimplemented("Not support %s as tensor type.",
                                               DataTypeToString(type)));
}

int DataTypeNumAlign(const proto::VarType::Type t) {
  int cast_type_num = -1;
  if (t == proto::VarType::UINT8 || t == proto::VarType::INT8 ||
      t == proto::VarType::BF16 || t == proto::VarType::COMPLEX64 ||
      t == proto::VarType::COMPLEX128) {
    cast_type_num = static_cast<int>(t) - 13;
  } else if (t == proto::VarType::BOOL || t == proto::VarType::INT16 ||
             t == proto::VarType::INT32 || t == proto::VarType::INT64 ||
             t == proto::VarType::FP16 || t == proto::VarType::FP32 ||
             t == proto::VarType::FP64) {
    cast_type_num = static_cast<int>(t);
  } else {
    PADDLE_THROW(platform::errors::Unavailable(
        "Only supports align tensor data type, but received "
        "proto::VarType::Type(%d) is not a tensor data type.",
        static_cast<int>(t)));
  }
  return cast_type_num;
}

proto::VarType::Type PromoteTypes(const proto::VarType::Type type_a,
                                  const proto::VarType::Type type_b) {
  constexpr auto b1 = proto::VarType::BOOL;   // 0
  constexpr auto i2 = proto::VarType::INT16;  // 1
  constexpr auto i4 = proto::VarType::INT32;  // 2
  constexpr auto i8 = proto::VarType::INT64;  // 3
  constexpr auto f2 = proto::VarType::FP16;   // 4
  constexpr auto f4 = proto::VarType::FP32;   // 5
  constexpr auto f8 = proto::VarType::FP64;   // 6

  constexpr auto u1 = proto::VarType::UINT8;       // 20
  constexpr auto i1 = proto::VarType::INT8;        // 21
  constexpr auto bf = proto::VarType::BF16;        // 22
  constexpr auto c4 = proto::VarType::COMPLEX64;   // 23
  constexpr auto c8 = proto::VarType::COMPLEX128;  // 24

  PADDLE_ENFORCE_EQ(IsDataType(type_a), true,
                    platform::errors::InvalidArgument(
                        "Only supports promote tensor data types, but received "
                        "proto::VarType::Type(%d) is not a tensor data type.",
                        static_cast<int>(type_a)));
  PADDLE_ENFORCE_EQ(IsDataType(type_b), true,
                    platform::errors::InvalidArgument(
                        "Only supports promote tensor data types, but received "
                        "proto::VarType::Type(%d) is not a tensor data type.",
                        static_cast<int>(type_b)));

  if (type_a == type_b) {
    return type_a;
  }

  int type_an = DataTypeNumAlign(type_a);
  int type_bn = DataTypeNumAlign(type_b);

  static constexpr proto::VarType::Type promote_types_table[12][12] = {
      /*        b1  i2  i4  i8  f2  f4  f8  u1  i1  bf  c4  c8*/
      /* b1 */ {b1, i2, i4, i8, f2, f4, f8, u1, i1, bf, c4, c8},
      /* i2 */ {i2, i2, i4, i8, f2, f4, f8, i2, i2, bf, c4, c8},
      /* i4 */ {i4, i4, i4, i8, f2, f4, f8, i4, i4, bf, c4, c8},
      /* i8 */ {i8, i8, i8, i8, f2, f4, f8, i8, i8, bf, c4, c8},
      /* f2 */ {f2, f2, f2, f2, f2, f4, f8, f2, f2, f4, c4, c8},
      /* f4 */ {f4, f4, f4, f4, f4, f4, f8, f4, f4, f4, c4, c8},
      /* f8 */ {f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, c8, c8},
      /* u1 */ {u1, i2, i4, i8, f2, f4, f8, u1, i2, bf, c4, c8},
      /* i1 */ {i1, i2, i4, i8, f2, f4, f8, i2, i1, bf, c4, c8},
      /* bf */ {bf, bf, bf, bf, f4, f4, f8, bf, bf, bf, c4, c8},
      /* c4 */ {c4, c4, c4, c4, c4, c4, c8, c4, c4, c4, c4, c8},
      /* c8 */ {c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8},
  };

  return promote_types_table[type_an][type_bn];
}

}  // namespace framework
}  // namespace paddle
