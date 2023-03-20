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

#include "paddle/ir/builtin_attribute_storage.h"
#include "paddle/ir/builtin_attribute.h"

namespace ir {

DictionaryAttributeStorage::DictionaryAttributeStorage(const ParamKey &key) {
  size_ = key.size();
  data_ = reinterpret_cast<NamedAttribute *>(
      malloc(sizeof(NamedAttribute) * size_));
  uint32_t idx = 0;
  for (auto iter = key.begin(); iter != key.end(); ++iter) {
    data_[idx].SetName(iter->first);
    data_[idx].SetValue(iter->second);
    idx++;
  }
}

DictionaryAttributeStorage::ParamKey DictionaryAttributeStorage::GetAsKey()
    const {
  ParamKey rtn;
  for (size_t idx = 0; idx < size_; idx++) {
    rtn.insert(std::pair<StrAttribute, Attribute>(data_[idx].name(),
                                                  data_[idx].value()));
  }
  return rtn;
}

std::size_t DictionaryAttributeStorage::HashValue(const ParamKey &key) {
  std::size_t hash_value = 0;
  hash_value = hash_combine(hash_value, std::hash<uint32_t>()(key.size()));
  for (auto iter = key.begin(); iter != key.end(); ++iter) {
    hash_value = hash_combine(
        hash_value,
        std::hash<NamedAttribute>()(NamedAttribute(iter->first, iter->second)));
  }
  return hash_value;
}

bool DictionaryAttributeStorage::operator==(const ParamKey &key) const {
  uint32_t size = key.size();
  if (size_ != size) return false;
  uint32_t idx = 0;
  for (auto iter = key.begin(); iter != key.end(); ++iter) {
    if (data_[idx] != NamedAttribute(iter->first, iter->second)) {
      return false;
    }
    idx++;
  }
  return true;
}

}  // namespace ir
