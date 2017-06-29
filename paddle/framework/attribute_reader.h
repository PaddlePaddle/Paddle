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

#pragma once
#include <google/protobuf/map.h>
#include <paddle/framework/attribute.pb.h>
#include <paddle/framework/enforce.h>
#include <algorithm>
#include <type_traits>
#include <vector>

namespace paddle {
namespace framework {
using AttributeMap = google::protobuf::Map<std::string, Attribute>;
class AttributeReader {
 public:
  explicit AttributeReader(const AttributeMap& attrs) : attrs_(attrs) {}

  /**
   * @brief Contains a attribute with name and type T.
   *
   * The example code like
   * @code{cpp}
   * AttributeReader reader;
   *
   * assert(reader.Contain<int>("SomeIntValue")==true);
   * assert(reader.Contain<float>("SomeIntValue")==false);
   * assert(reader.Contain<std::vector<int>>("SomeIntList")==true);
   * @endcode{cpp}
   *
   * @tparam T Attribute Type, could be {int, float, string and std::vector of
   * them}.
   * @param name attribute name
   * @return true if contain an attribute with name and type T, false if Type
   * mismatch or not contains that name.
   */
  template <typename T>
  bool Contains(const std::string& name) const;

  /**
   * @brief Get Attribute value. Not support std::vector. If want to return a
   * std::vector, use `GetArray`
   * @tparam T could be int, float, string.
   * @param name attribute name.
   * @return Value
   * @throw If attribute is not found or type mismatch, an EnforceNotMet will
   * throw
   */
  template <typename T>
  T Get(const std::string& name) const;

  /**
   * @brief Get Attribute array values.
   * @tparam T could be int, float, string.
   * @param name attribute name.
   * @param vec the return vector. Must be empty.
   * @throw If attribute is not found, or type mismatch, or vec is not empty, an
   * EnforceNotMet will throw
   */
  template <typename T>
  void GetArray(const std::string& name, std::vector<T>* vec) const;

 private:
  const AttributeMap& attrs_;
};

/// Implementation of Contain
namespace details {
inline const ::paddle::framework::Attribute* GetField(const AttributeMap& attrs,
                                                      const std::string& name) {
  auto it = attrs.find(name);
  if (it == attrs.end()) {
    return nullptr;
  } else {
    return &it->second;
  }
}

template <typename T>
inline bool IsType(const ::paddle::framework::Attribute* attr);

template <typename T, bool IsArray>
struct ContainsImpl {};

template <typename T>
struct ContainsImpl<T, false> {
  bool operator()(const AttributeMap& attrs, const std::string& name) {
    auto attr = GetField(attrs, name);
    if (attr) {
      return details::IsType<T>(attr);
    } else {
      return false;
    }
  }
};

template <typename T>
struct ContainsImpl<T, true> {
  bool operator()(const AttributeMap& attrs, const std::string& name) {
    auto attr = GetField(attrs, name);
    if (attr) {
      return attr->has_list();
    } else {
      return false;
    }
  }
};
}  // namespace details

template <typename T>
bool AttributeReader::Contains(const std::string& name) const {
  constexpr bool is_vec = std::is_same<T, std::vector<int>>::value ||
                          std::is_same<T, std::vector<float>>::value ||
                          std::is_same<T, std::vector<std::string>>::value;
  return details::ContainsImpl<T, is_vec>()(attrs_, name);
}

#define ATTR_READER_ISTYPE_IMPL(T, CASE)                              \
  namespace details {                                                 \
  template <>                                                         \
  inline bool IsType<T>(const ::paddle::framework::Attribute* attr) { \
    return attr->value_case() == CASE;                                \
  }                                                                   \
  }

ATTR_READER_ISTYPE_IMPL(int, ::paddle::framework::Attribute::kI);
ATTR_READER_ISTYPE_IMPL(float, ::paddle::framework::Attribute::kF);
ATTR_READER_ISTYPE_IMPL(std::string, ::paddle::framework::Attribute::kS);

#undef ATTR_READER_ISTYPE_IMPL

/// Implementation of Get
namespace details {
template <typename T>
inline T GetValue(const ::paddle::framework::Attribute* attr);
}

template <typename T>
T AttributeReader::Get(const std::string& name) const {
  auto attr = details::GetField(attrs_, name);
  PADDLE_ENFORCE(attr != nullptr, "Attribute %s not found", name);
  PADDLE_ENFORCE(details::IsType<T>(attr),
                 "Attribute type mismatch. Expected %s", typeid(T).name());
  return details::GetValue<T>(attr);
}

#define ATTR_READER_GETVALUE_IMPL(T, FIELD)                          \
  namespace details {                                                \
  template <>                                                        \
  inline T GetValue<T>(const ::paddle::framework::Attribute* attr) { \
    return attr->FIELD();                                            \
  }                                                                  \
  }

ATTR_READER_GETVALUE_IMPL(int, i);
ATTR_READER_GETVALUE_IMPL(float, f);
ATTR_READER_GETVALUE_IMPL(std::string, s);

#undef ATTR_READER_GETVALUE_IMPL

/// Implementation of GetArray
#define ATTR_GETARRAY_IMPL(T, FIELD)                                     \
  template <>                                                            \
  void AttributeReader::GetArray<T>(const std::string& name,             \
                                    std::vector<T>* vec) const {         \
    PADDLE_ENFORCE(vec->empty(), "Input vector should be empty");        \
    auto attr = details::GetField(attrs_, name);                         \
    PADDLE_ENFORCE(attr != nullptr, "Attribute %s not found", name);     \
    PADDLE_ENFORCE(attr->has_list(), "Attribute %s is not array", name); \
    auto& field = attr->list().FIELD();                                  \
    vec->reserve(field.size());                                          \
    std::copy(field.begin(), field.end(), std::back_inserter(*vec));     \
  }

ATTR_GETARRAY_IMPL(int, ints);
ATTR_GETARRAY_IMPL(float, floats);
ATTR_GETARRAY_IMPL(std::string, strings);

#undef ATTR_GETARRAY_IMPL

}  // namespace framework
}  // namespace paddle
