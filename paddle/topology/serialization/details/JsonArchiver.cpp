/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "JsonArchiver.h"
#include <json.hpp>
#include <type_traits>

namespace paddle {
namespace topology {
namespace serialization {
namespace details {

using json = nlohmann::json;

template <size_t I, typename... Types>
static bool castImpl(
    json&,
    const std::string&,
    const any&,
    typename std::enable_if<I >= std::tuple_size<std::tuple<Types...>>::value,
                            bool>::type) {
  return false;
}

template <typename T>
static void castElement(json& attr, const std::string& key, const any& value) {
  attr[key] = any_cast<T>(value);
}

template <size_t I, typename... Types>
static bool castImpl(
    json& attr,
    const std::string& key,
    const any& value,
    typename std::enable_if <
        I<std::tuple_size<std::tuple<Types...>>::value, bool>::type) {
  using T = typename std::tuple_element<I, std::tuple<Types...>>::type;
  if (typeid(T) == value.type()) {
    castElement<T>(attr, key, value);
    return true;
  } else {
    return castImpl<I + 1, Types...>(attr, key, value, false);
  }
}
template <typename... ARGS>
static bool cast(json& attrs, const std::string& key, const any& value) {
  return castImpl<0, ARGS...>(attrs, key, value, false);
}

#define SUPPORT_TYPES \
  double, int, std::string, std::vector<int>, std::vector<size_t>

Error JsonArchiver::serialize(const paddle::topology::AttributeMap& attrs,
                              std::ostream& sout) {
  json jsonAttr;

  for (auto it = attrs.begin(); it != attrs.end(); ++it) {
    auto key = it->first;
    auto value = it->second;
    auto ok = cast<SUPPORT_TYPES>(jsonAttr, key, value);
    if (!ok) {
      return Error("Cannot cast %s to json, because type %s not support",
                   key.c_str(),
                   value.type().name());
    }
  }
  sout << jsonAttr;
  return Error();
}

template <size_t LEN, size_t I, typename... Types>
static bool deserializeCastImpl(AttributeMap*,
                                const std::string&,
                                const json&,
                                const std::type_info&,
                                typename std::enable_if<I >= LEN, bool>::type) {
  return false;
}

template <typename T>
static void deserializeCastImpl(AttributeMap* attrs,
                                const std::string& key,
                                const json& value) {
  T tmp = value;
  (*attrs)[key] = tmp;
}

template <size_t LEN, size_t I, typename... Types>
static bool deserializeCastImpl(AttributeMap* attrs,
                                const std::string& key,
                                const json& value,
                                const std::type_info& type,
                                typename std::enable_if < I<LEN, bool>::type) {
  using T = typename std::tuple_element<I, std::tuple<Types...>>::type;
  if (typeid(T) == type) {
    deserializeCastImpl<T>(attrs, key, value);
    return true;
  } else {
    return deserializeCastImpl<LEN, I + 1, Types...>(
        attrs, key, value, type, false);
  }
}

template <typename... Types>
static bool deserializeCast(AttributeMap* attrs,
                            const std::string& key,
                            const json& value,
                            const std::type_info& type) {
  return deserializeCastImpl<std::tuple_size<std::tuple<Types...>>::value,
                             0,
                             Types...>(attrs, key, value, type, false);
}

static Error deserializeImpl(json& j,
                             const meta::AttributeMetaMap& meta,
                             AttributeMap* attrs) {
  auto& metaAttrs = meta.getAttributes();
  for (auto it = j.begin(); it != j.end(); ++it) {
    auto key = it.key();
    auto& jsonValue = it.value();
    auto metaIt = metaAttrs.find(key);
    if (metaIt == metaAttrs.end()) {
      return Error("Cannot found meta information of attribute %s",
                   key.c_str());
    }
    auto& type = metaIt->second->type;
    auto ok = deserializeCast<SUPPORT_TYPES>(attrs, key, jsonValue, type);
    if (!ok) {
      return Error("Not supported types %s", type.name());
    }
  }

  return Error();
}

Error JsonArchiver::deserialize(std::istream& in,
                                const meta::AttributeMetaMap& meta,
                                AttributeMap* attrs) {
  json attr;
  in >> attr;

  if (!attr.is_object()) {
    return Error("Attribute must be a object.");
  }
  return deserializeImpl(attr, meta, attrs);
}

}  // namespace details
}  // namespace serialization
}  // namespace topology
}  // namespace paddle
