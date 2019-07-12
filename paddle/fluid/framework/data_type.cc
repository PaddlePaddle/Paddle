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
#include <stdint.h>
#include <string>
#include <unordered_map>

using float16 = paddle::platform::float16;

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
  PADDLE_THROW("Not support %s as tensor type", type.name());
}

std::type_index ToTypeIndex(proto::VarType::Type type) {
  auto it = gDataTypeMap().proto_to_cpp_.find(static_cast<int>(type));
  if (it != gDataTypeMap().proto_to_cpp_.end()) {
    return it->second;
  }
  PADDLE_THROW("Not support proto::VarType::Type(%d) as tensor type",
               static_cast<int>(type));
}

std::string DataTypeToString(const proto::VarType::Type type) {
  auto it = gDataTypeMap().proto_to_str_.find(static_cast<int>(type));
  if (it != gDataTypeMap().proto_to_str_.end()) {
    return it->second;
  }
  PADDLE_THROW("Not support proto::VarType::Type(%d) as tensor type",
               static_cast<int>(type));
}

size_t SizeOfType(proto::VarType::Type type) {
  auto it = gDataTypeMap().proto_to_size_.find(static_cast<int>(type));
  if (it != gDataTypeMap().proto_to_size_.end()) {
    return it->second;
  }
  PADDLE_THROW("Not support %s as tensor type", DataTypeToString(type));
}

}  // namespace framework
}  // namespace paddle
