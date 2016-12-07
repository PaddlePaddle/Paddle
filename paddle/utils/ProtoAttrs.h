#pragma once
/* Copyright (c) 2016 PaddlePaddle Authors, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include "Attribute.pb.h"
#include "Logging.h"
#include "TypeDefs.h"

namespace paddle {

namespace internal {
extern std::string gReaderErrorStub;
}

template <typename Container = google::protobuf::RepeatedPtrField<Attribute>>
class AttributeReader {
public:
  explicit inline AttributeReader(const Container& attrs) : attrs_(attrs) {}

  inline typename Container::const_iterator get(const std::string& name) const {
    return std::find_if(
        attrs_.begin(), attrs_.end(), [&name](const Attribute& attr) {
          return attr.name() == name;
        });
  }

  inline bool contain(const std::string& name) const {
    return this->get(name) != attrs_.end();
  }

  inline const Attribute& attr(const std::string& name,
                               bool* ok = nullptr) const {
    auto it = this->get(name);
    CHECK(it != attrs_.end() || ok != nullptr) << "cannot find " << name
                                               << " in attributes";
    if (ok != nullptr) {
      *ok = it != attrs_.end();
    }
    return *it;
  }

  inline Attribute::AttributeType getType(const std::string& name,
                                          bool* ok = nullptr) const {
    auto& tmp = attr(name, ok);
    if (ok != nullptr && !(*ok)) {  // cannot found.
      return Attribute_AttributeType_REAL;
    } else {
      return tmp.type();
    }
  }

  inline const std::string& getStr(const std::string& name,
                                   bool* ok = nullptr) const {
    auto& attr = this->attr(name, ok);
    if (ok != nullptr && !(*ok)) {
      return internal::gReaderErrorStub;
    }
    CHECK_EQ(attr.type(), Attribute_AttributeType_STRING);
    return attr.s_val();
  }

  inline real getReal(const std::string& name, bool* ok = nullptr) const {
    auto& attr = this->attr(name, ok);
    if (ok != nullptr && !(*ok)) {
      return .0;
    }
    CHECK_EQ(attr.type(), Attribute_AttributeType_REAL);
    return attr.r_val();
  }

  inline bool getBool(const std::string& name, bool* ok = nullptr) const {
    auto& attr = this->attr(name);
    if (ok != nullptr && !(*ok)) {
      return false;
    }
    CHECK_EQ(attr.type(), Attribute_AttributeType_BOOL);
    return attr.b_val();
  }

  inline int32_t getInt32(const std::string& name, bool* ok = nullptr) const {
    auto& attr = this->attr(name);
    if (ok != nullptr && !(*ok)) {
      return 0;
    }
    CHECK_EQ(attr.type(), Attribute_AttributeType_INT32);
    return attr.i_val();
  }

private:
  const Container& attrs_;
};

template <typename Container = google::protobuf::RepeatedPtrField<Attribute>>
class AttributeWriter {
public:
  explicit inline AttributeWriter(Container* attrs) : attrs_(attrs) {}

  inline bool contain(const std::string& name) const {
    return AttributeReader<Container>(*attrs_).contain(name);
  }

  inline Attribute* add(const std::string& name) {
    CHECK(!contain(name)) << name << " has been set";
    Attribute* attr = attrs_->Add();
    attr->set_name(name);
    return attr;
  }

  inline void addStr(const std::string& name, const std::string& value) {
    Attribute* attr = this->add(name);
    attr->set_type(Attribute_AttributeType_STRING);
    attr->set_s_val(value);
  }

  inline void addInt32(const std::string& name, int32_t value) {
    Attribute* attr = this->add(name);
    attr->set_type(Attribute_AttributeType_INT32);
    attr->set_i_val(value);
  }

  inline void addReal(const std::string& name, real value) {
    Attribute* attr = this->add(name);
    attr->set_type(Attribute_AttributeType_REAL);
    attr->set_r_val(value);
  }

  inline void addBool(const std::string& name, bool value) {
    Attribute* attr = add(name);
    attr->set_type(Attribute_AttributeType_BOOL);
    attr->set_b_val(value);
  }

private:
  Container* attrs_;
};
}  // namespace paddle
