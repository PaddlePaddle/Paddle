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

namespace paddle {
namespace framework {

struct DataTypeMap {
  std::unordered_map<std::type_index, proto::VarType::Type> cpp_to_proto_;
  std::unordered_map<proto::VarType::Type, std::type_index> proto_to_cpp_;
  std::unordered_map<proto::VarType::Type, std::string> proto_to_str_;
};

static DataTypeMap g_data_type_map_;

template <typename T>
static inline void RegisterType(proto::VarType::Type proto_type,
                                const std::string &name) {
  g_data_type_map_.proto_to_cpp_.emplace(proto_type, typeid(T));
  g_data_type_map_.cpp_to_proto_.emplace(typeid(T), proto_type);
  g_data_type_map_.proto_to_str_.emplace(proto_type, name);
}

static int RegisterAllTypes() {
#define RegType(cc_type, proto_type) RegisterType<cc_type>(proto_type, #cc_type)

  RegType(platform::float16, proto::VarType::FP16);
  RegType(float, proto::VarType::FP32);
  RegType(double, proto::VarType::FP64);
  RegType(int, proto::VarType::INT32);
  RegType(int64_t, proto::VarType::INT64);
  RegType(bool, proto::VarType::BOOL);

#undef RegType
  return 0;
}

static std::once_flag register_once_flag_;

proto::VarType::Type ToDataType(std::type_index type) {
  std::call_once(register_once_flag_, RegisterAllTypes);
  auto it = g_data_type_map_.cpp_to_proto_.find(type);
  if (it != g_data_type_map_.cpp_to_proto_.end()) {
    return it->second;
  }
  PADDLE_THROW("Not support %s as tensor type", type.name());
}

std::type_index ToTypeIndex(proto::VarType::Type type) {
  std::call_once(register_once_flag_, RegisterAllTypes);
  auto it = g_data_type_map_.proto_to_cpp_.find(type);
  if (it != g_data_type_map_.proto_to_cpp_.end()) {
    return it->second;
  }
  PADDLE_THROW("Not support proto::VarType::Type(%d) as tensor type",
               static_cast<int>(type));
}

std::string DataTypeToString(const proto::VarType::Type type) {
  std::call_once(register_once_flag_, RegisterAllTypes);
  auto it = g_data_type_map_.proto_to_str_.find(type);
  if (it != g_data_type_map_.proto_to_str_.end()) {
    return it->second;
  }
  PADDLE_THROW("Not support proto::VarType::Type(%d) as tensor type",
               static_cast<int>(type));
}

}  // namespace framework
}  // namespace paddle
