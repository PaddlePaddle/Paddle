#pragma once
#include <google/protobuf/map.h>
#include <paddle/framework/enforce.h>
#include "attribute.pb.h"

namespace paddle {
namespace framework {
using AttributeMap = google::protobuf::Map<std::string, Attribute>;
class AttributeReader {
 public:
  explicit AttributeReader(const AttributeMap& attrs) : attrs_(attrs) {}

  template <typename T>
  bool ContainPlain(const std::string& name) const;

  bool IsArray(const std::string& name) const;

  template <typename T>
  T Get(const std::string& name) const;

  template <typename T>
  void GetArray(const std::string& name, std::vector<T>* vec) const;

 private:
  const AttributeMap& attrs_;
};

/// Implementation
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
}  // namespace details

bool AttributeReader::IsArray(const std::string& name) const {
  auto attr = details::GetField(attrs_, name);
  if (attr) {
    return attr->has_list();
  } else {
    return false;
  }
}

namespace details {
template <typename T>
inline bool IsType(const ::paddle::framework::Attribute* attr);
}  // namespace details

template <typename T>
bool AttributeReader::ContainPlain(const std::string& name) const {
  auto attr = details::GetField(attrs_, name);
  if (attr) {
    return details::IsType<T>(attr);
  } else {
    return false;
  }
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

}  // namespace framework
}  // namespace paddle
