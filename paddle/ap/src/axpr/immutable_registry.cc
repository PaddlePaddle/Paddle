// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/ap/include/axpr/immutable_registry.h"
#include <atomic>
#include <unordered_map>

namespace ap::axpr {

namespace {

const std::string& ImmutableValueRegistryKeyPrefix() {
  static std::string prefix("iv");
  return prefix;
}

std::size_t AutoIncrementalSeqNo() {
  static std::atomic<std::size_t> atomic;
  return atomic++;
}

std::unordered_map<std::string, axpr::Value>* MutKey2Immutable() {
  static std::unordered_map<std::string, axpr::Value> key2immutable;
  return &key2immutable;
}

const std::unordered_map<std::string, axpr::Value>& Key2Immutable() {
  return *MutKey2Immutable();
}

}  // namespace

adt::Result<std::string> AutoImmutableValueRegistryKey() {
  return ImmutableValueRegistryKeyPrefix() +
         std::to_string(AutoIncrementalSeqNo());
}

adt::Result<bool> StartsWithImmutableValueRegistryKeyPrefix(
    const std::string& key) {
  if (key.size() < ImmutableValueRegistryKeyPrefix().size()) return false;
  return key.substr(0, ImmutableValueRegistryKeyPrefix().size()) ==
         ImmutableValueRegistryKeyPrefix();
}

adt::Result<bool> IsImmutableValueRegistered(const std::string& key) {
  return Key2Immutable().find(key) != Key2Immutable().end();
}

adt::Result<axpr::Value> GetRegisteredImmutableValue(const std::string& key) {
  const auto& iter = Key2Immutable().find(key);
  ADT_CHECK(iter != Key2Immutable().end());
  return iter->second;
}

adt::Result<adt::Ok> RegisterImmutableValue(const std::string& key,
                                            const axpr::Value& value) {
  ADT_CHECK(MutKey2Immutable()->emplace(key, value).second)
      << adt::errors::InvalidArgumentError{
             std::string() + "immutable value registered. key: " + key};
  return adt::Ok{};
}

adt::Result<axpr::Value> ApiAutoImmutableValueRegistryKey(
    const axpr::Value&, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 0);
  ADT_LET_CONST_REF(key, AutoImmutableValueRegistryKey());
  return key;
}

adt::Result<axpr::Value> ApiIsImmutableValueRegistered(
    const axpr::Value&, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 1);
  ADT_LET_CONST_REF(key, args.at(0).template CastTo<std::string>());
  ADT_LET_CONST_REF(is_registered, IsImmutableValueRegistered(key));
  return is_registered;
}

adt::Result<axpr::Value> ApiGetRegisteredImmutableValue(
    const axpr::Value&, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 1);
  ADT_LET_CONST_REF(key, args.at(0).template CastTo<std::string>());
  ADT_LET_CONST_REF(registered_value, GetRegisteredImmutableValue(key));
  return registered_value;
}

adt::Result<axpr::Value> ApiRegisterImmutableValue(
    const axpr::Value&, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 2);
  ADT_LET_CONST_REF(key, args.at(0).template CastTo<std::string>());
  ADT_RETURN_IF_ERR(RegisterImmutableValue(key, args.at(1)));
  return adt::Nothing{};
}

}  // namespace ap::axpr
