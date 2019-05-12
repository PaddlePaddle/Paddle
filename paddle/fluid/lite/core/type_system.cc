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
#include "type_system.h"

namespace paddle {
namespace lite {

// ------------------------- GetType specification ----------------------------
// ------------------------- end GetType specification ------------------------

size_t ParamTypeRegistry::KernelIdTy::hash() const {
  std::hash<std::string> h;
  size_t hash = h(kernel_type);
  hash = hash_combine(hash, place.hash());
  hash = hash_combine(hash, std::hash<int>()(static_cast<int>(io)));
  hash = hash_combine(hash, std::hash<std::string>()(arg_name));
  return hash;
}

std::ostream &operator<<(std::ostream &os, const Type &other) {
  if (other.IsUnsupported()) {
    os << "<Unsupported>";
    return os;
  }
  if (other.IsVoid()) {
    os << "<Void>";
    return os;
  }
  if (other.IsTensor()) {
    os << "<Tensor:";
  } else {
    os << "<";
  }
  os << TargetToStr(other.target()) << "/" << PrecisionToStr(other.precision())
     << "/" << DataLayoutToStr(other.layout()) << ">";
  return os;
}

const Type *Type::GetTensorTy(TargetType target, PrecisionType precision,
                              DataLayoutType layout, int device) {
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

  auto it = type_repo_.find(v);
  if (it == type_repo_.end()) {
    // The Types should alive across the process life, no need to delete.
    type_repo_[v] =
        new Type(type_id, name.str(), target, precision, layout, device);
  }
  return type_repo_[v];
}

const Type *Type::GetTensorListTy(TargetType target, PrecisionType precision,
                                  DataLayoutType layout, int device) {
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

  if (!type_repo_[v])
    // The Types should alive across the process life, no need to delete.
    type_repo_[v] =
        new Type(type_id, name.str(), target, precision, layout, device);
  return type_repo_[v];
}

const Type *Type::GetUnsupportedTy() {
  std::hash<int> hasher;
  size_t v = hasher(static_cast<int>(DataType::ID::Unsupported));
  if (!type_repo_[v])
    type_repo_[v] =
        new Type(DataType::ID::Unsupported, "Unsupported", TARGET(kUnk),
                 PRECISION(kUnk), DATALAYOUT(kUnk), -1);
}

const Type *Type::GetVoidTy() {
  std::hash<int> hasher;
  size_t v = hasher(static_cast<int>(DataType::ID::Void));
  if (!type_repo_[v])
    type_repo_[v] = new Type(DataType::ID::Void, "Void", TARGET(kAny),
                             PRECISION(kAny), DATALAYOUT(kAny), -1);
}

}  // namespace lite
}  // namespace paddle
