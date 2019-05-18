// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/lite/core/type_system.h"

namespace paddle {
namespace lite {

size_t ParamTypeRegistry::KernelIdTy::hash() const {
  std::hash<std::string> h;
  size_t hash = h(kernel_type);
  hash = hash_combine(hash, place.hash());
  hash = hash_combine(hash, std::hash<int>()(static_cast<int>(io)));
  hash = hash_combine(hash, std::hash<std::string>()(arg_name));
  return hash;
}

std::ostream &operator<<(std::ostream &os, const Type &other) {
  os << other.name();
  return os;
}

// An map is used to maintain a global repo for types. We don't use
// MACROs with static variables for that the TypeSystem should only used in
// compile time, that is not performance sensitive, and a map-based way is
// easier to implement and maintain.
//
// The map is declared in each Type::GetXXX method other than in the Type class
// so that it will force to construct before any usage.

const Type *Type::GetTensorTy(TargetType target, PrecisionType precision,
                              DataLayoutType layout, int device) {
  static std::map<size_t, const Type *> type_repo;
  // NOTE quite naive implementation here, but not performance sensitive.
  DataType::ID type_id = DataType::ID::Tensor;

#define HASH_ONE(x) v = hash_combine(v, hasher(static_cast<int>(x)))

  std::hash<int> hasher;
  size_t v = hasher(static_cast<int>(type_id));
  HASH_ONE(target);
  HASH_ONE(precision);
  HASH_ONE(layout);
  HASH_ONE(device);
#undef HASH_ONE

  std::stringstream name;
  name << "Tensor<";
  name << TargetToStr(target) << ",";
  name << PrecisionToStr(precision) << ",";
  name << DataLayoutToStr(layout) << ",";
  name << device;
  name << ">";

  if (!type_repo[v])
    // The Types should alive across the process life, no need to delete.
    type_repo[v] =
        new Type(type_id, name.str(), target, precision, layout, device);
  return type_repo[v];
}

const Type *Type::GetTensorListTy(TargetType target, PrecisionType precision,
                                  DataLayoutType layout, int device) {
  static std::map<size_t, const Type *> type_repo;
  DataType::ID type_id = DataType::ID::TensorList;

#define HASH_ONE(x) v = hash_combine(v, hasher(static_cast<int>(x)))

  std::hash<int> hasher;
  size_t v = hasher(static_cast<int>(type_id));
  HASH_ONE(target);
  HASH_ONE(precision);
  HASH_ONE(layout);
  HASH_ONE(device);
#undef HASH_ONE

  std::stringstream name;
  name << "TensorList<";
  name << TargetToStr(target) << ",";
  name << PrecisionToStr(precision) << ",";
  name << DataLayoutToStr(layout) << ",";
  name << device;
  name << ">";

  if (!type_repo[v])
    // The Types should alive across the process life, no need to delete.
    type_repo[v] =
        new Type(type_id, name.str(), target, precision, layout, device);
  return type_repo[v];
}

const Type *Type::GetUnsupportedTy() {
  static std::map<size_t, const Type *> type_repo;
  std::hash<int> hasher;
  size_t v = hasher(static_cast<int>(DataType::ID::Unsupported));
  if (!type_repo[v])
    type_repo[v] =
        new Type(DataType::ID::Unsupported, "Unsupported", TARGET(kUnk),
                 PRECISION(kUnk), DATALAYOUT(kUnk), -1);
  return type_repo[v];
}

const Type *Type::GetVoidTy() {
  static std::map<size_t, const Type *> type_repo;
  std::hash<int> hasher;
  size_t v = hasher(static_cast<int>(DataType::ID::Void));
  if (!type_repo[v])
    type_repo[v] = new Type(DataType::ID::Void, "Void", TARGET(kAny),
                            PRECISION(kAny), DATALAYOUT(kAny), -1);
  return type_repo[v];
}

const Type *Type::Get(DataType::ID type_id, TargetType target,
                      PrecisionType precision, DataLayoutType layout,
                      int device) {
  switch (type_id) {
    case DataType::ID::Void:
      return GetVoidTy();
    case DataType::ID::Unsupported:
      return GetUnsupportedTy();
    case DataType::ID::Tensor:
      return GetTensorTy(target, precision, layout, device);
    case DataType::ID::TensorList:
      return GetTensorListTy(target, precision, layout, device);
    default:
      LOG(FATAL) << "Unknown Type found";
      return nullptr;
  }
}

}  // namespace lite
}  // namespace paddle
