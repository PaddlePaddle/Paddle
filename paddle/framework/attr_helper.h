#pragma once
#include <google/protobuf/map.h>
#include <paddle/framework/error.h>
#include <iterator>
#include <string>
#include <type_traits>
#include "attr.pb.h"
namespace paddle {
namespace framework {
using AttributeMap = google::protobuf::Map<std::string, Attribute>;

class AttributeReader final {
 public:
  explicit AttributeReader(const AttributeMap& attrs) : attrs_(attrs) {}

  template <typename T>
  Error __must_check Get(const std::string& attributeName, T* attr) const;

  template <typename T>
  Error __must_check GetArray(const std::string& attributeName,
                              std::vector<T>* array) const;

 private:
  const AttributeMap& attrs_;
};

namespace details {
template <typename Iterator, typename T>
struct SetArrayImpl {
  Error __must_check operator()(AttributeMap* attrs,
                                const std::string& attributeName,
                                Iterator begin, Iterator end, bool overwrite);
};
}  // namespace details

class AttributeWriter {
 public:
  explicit AttributeWriter(AttributeMap* attrs) : attrs_(attrs) {}

  template <typename T>
  Error __must_check Set(const std::string& attributeName, const T& attr,
                         bool overwrite = false);

  template <typename Iterator>
  Error __must_check SetArray(const std::string& attributeName, Iterator begin,
                              Iterator end, bool overwrite = false) {
    return details::SetArrayImpl<
        Iterator, typename std::iterator_traits<Iterator>::value_type>()(
        attrs_, attributeName, begin, end, overwrite);
  }

  template <typename T, typename Container = std::initializer_list<T>>
  Error __must_check SetArray(const std::string& attributeName,
                              Container container, bool overwrite = false) {
    return SetArray(attributeName, container.begin(), container.end(),
                    overwrite);
  }

 private:
  AttributeMap* attrs_;
};

#define ATTR_READER_IMPL_PLAIN_TYPE(T, CASE, FIELD_NAME)                       \
  template <>                                                                  \
  Error __must_check AttributeReader::Get<T>(const std::string& attributeName, \
                                             T* attr) const {                  \
    auto it = attrs_.find(attributeName);                                      \
    if (it == attrs_.end()) {                                                  \
      return Error("Attribute %s not found", attributeName.c_str());           \
    }                                                                          \
    if (it->second.value_case() != CASE) {                                     \
      return Error("Attribute should be in field " #FIELD_NAME);               \
    }                                                                          \
    *attr = it->second.FIELD_NAME();                                           \
    return Error();                                                            \
  }

ATTR_READER_IMPL_PLAIN_TYPE(int, Attribute::kI, i);
ATTR_READER_IMPL_PLAIN_TYPE(float, Attribute::kF, f);
ATTR_READER_IMPL_PLAIN_TYPE(std::string, Attribute::kS, s);

#undef ATTR_READER_IMPL_PLAIN_TYPE

#define ATTR_READER_IMPL_ARRAY_TYPE(T, FIELD_NAME)                     \
  template <>                                                          \
  Error __must_check AttributeReader::GetArray<T>(                     \
      const std::string& attributeName, std::vector<T>* array) const { \
    if (!array->empty()) {                                             \
      return Error("The output array must be empty.");                 \
    }                                                                  \
                                                                       \
    auto it = attrs_.find(attributeName);                              \
    if (it == attrs_.end()) {                                          \
      return Error("Attribute %s not found", attributeName.c_str());   \
    }                                                                  \
                                                                       \
    auto& lst = it->second.list();                                     \
    auto& field = lst.FIELD_NAME();                                    \
    array->reserve(field.size());                                      \
    std::copy(field.begin(), field.end(), std::back_inserter(*array)); \
    return Error();                                                    \
  }

ATTR_READER_IMPL_ARRAY_TYPE(float, floats);
ATTR_READER_IMPL_ARRAY_TYPE(int, ints);
ATTR_READER_IMPL_ARRAY_TYPE(std::string, strings);

#undef ATTR_READER_IMPL_ARRAY_TYPE

#define ATTR_WRITER_IMPL_PLAIN_TYPE(T, FIELD_NAME)                             \
  template <>                                                                  \
  Error __must_check AttributeWriter::Set<T>(const std::string& attributeName, \
                                             const T& attr, bool overwrite) {  \
    auto it = attrs_->find(attributeName);                                     \
    if (it != attrs_->end() && !overwrite) {                                   \
      return Error("Attribute %s has been set", attributeName.c_str());        \
    }                                                                          \
    (*attrs_)[attributeName].set_##FIELD_NAME(attr);                           \
    return Error();                                                            \
  }

ATTR_WRITER_IMPL_PLAIN_TYPE(int, i);
ATTR_WRITER_IMPL_PLAIN_TYPE(float, f);
ATTR_WRITER_IMPL_PLAIN_TYPE(std::string, s);

#undef ATTR_WRITER_IMPL_PLAIN_TYPE

namespace details {
template <typename T>
void AppendToField(google::protobuf::RepeatedField<T>* field, const T& val) {
  field->Add(val);
}
template <typename T>
void AppendToField(google::protobuf::RepeatedPtrField<T>* field, const T& val) {
  *(field->Add()) = val;
}

}  // namespace details

#define ATTR_WRITER_IMPL_ARRAY_TYPE(T, FIELD_NAME)                          \
  namespace details {                                                       \
                                                                            \
  template <typename Iterator>                                              \
  struct SetArrayImpl<Iterator, T> {                                        \
    using VALUE_TYPE = typename std::iterator_traits<Iterator>::value_type; \
    Error __must_check operator()(AttributeMap* attrs,                      \
                                  const std::string& attributeName,         \
                                  Iterator begin, Iterator end,             \
                                  bool overwrite) {                         \
      static_assert(std::is_same<VALUE_TYPE, T>::value, "");                \
      auto it = attrs->find(attributeName);                                 \
      if (it != attrs->end() && !overwrite) {                               \
        return Error("Attribute %s has been set", attributeName.c_str());   \
      }                                                                     \
                                                                            \
      if (it != attrs->end() && overwrite) {                                \
        auto repeatedFieldPtr =                                             \
            it->second.mutable_list()->mutable_##FIELD_NAME();              \
        repeatedFieldPtr->erase(repeatedFieldPtr->begin(),                  \
                                repeatedFieldPtr->end());                   \
      }                                                                     \
      auto lst = (*attrs)[attributeName].mutable_list();                    \
      auto elems = lst->mutable_##FIELD_NAME();                             \
      auto distance = std::distance(begin, end);                            \
      if (std::is_integral<decltype(distance)>::value) {                    \
        elems->Reserve(distance);                                           \
      }                                                                     \
      for (; begin != end; ++begin) {                                       \
        AppendToField(elems, *begin);                                       \
      }                                                                     \
      return Error();                                                       \
    }                                                                       \
  };                                                                        \
  }

ATTR_WRITER_IMPL_ARRAY_TYPE(float, floats);
ATTR_WRITER_IMPL_ARRAY_TYPE(int, ints);
ATTR_WRITER_IMPL_ARRAY_TYPE(std::string, strings);

#undef ATTR_WRITER_IMPL_ARRAY_TYPE

}  // namespace framework
}  // namespace paddle
