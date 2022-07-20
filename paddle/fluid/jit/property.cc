/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/jit/property.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"

namespace paddle {
namespace jit {

int Property::Size() const { return property_.entrys_size(); }

void Property::SetFloat(const float &f) {
  auto type = proto::ValueProto::FLOAT;
  auto entry = property_.add_entrys();
  entry->set_type(type);
  entry->set_f(f);
  VLOG(3) << "Property: set_float " << f;
}

void Property::SetFloat(const std::string &name, const float &f) {
  auto type = proto::ValueProto::FLOAT;
  auto entry = property_.add_entrys();
  entry->set_name(name);
  entry->set_type(type);
  entry->set_f(f);
  VLOG(3) << "Property: set_float " << f << " name: " << name;
}

float Property::GetFloat(const std::string &name) const {
  for (int i = 0; i < Size(); i++) {
    auto e = property_.entrys(i);
    if (e.has_name() && e.name() == name) {
      return e.f();
    }
  }

  PADDLE_THROW(phi::errors::NotFound("name: %s not found", name));
  return 0;
}

float Property::GetFloat(const int &idx) const {
  PADDLE_ENFORCE_EQ(idx < Size() && idx >= 0,
                    true,
                    phi::errors::OutOfRange("idx out of range"));

  auto e = property_.entrys(idx);
  if (e.has_f()) {
    return e.f();
  }

  PADDLE_THROW(
      phi::errors::InvalidArgument("get_float: idx (%d) is not a float.", idx));
  return 0;
}

void Property::SetFloats(const std::vector<float> &v) {
  auto type = proto::ValueProto::FLOATS;
  auto entry = property_.add_entrys();
  entry->set_type(type);
  for (auto i : v) {
    entry->add_floats(i);
  }
  VLOG(3) << "Property: set_floats  with length: " << v.size();
}

void Property::SetFloats(const std::string &name, const std::vector<float> &v) {
  auto type = proto::ValueProto::FLOATS;
  auto entry = property_.add_entrys();
  entry->set_name(name);
  entry->set_type(type);
  for (auto i : v) {
    entry->add_floats(i);
  }
  VLOG(3) << "Property: set_floats  with length " << v.size()
          << " for name: " << name;
}

void Property::SetInt64(const int64_t &i) {
  auto type = proto::ValueProto::INT;
  auto entry = property_.add_entrys();
  entry->set_type(type);
  entry->set_i(i);
  VLOG(3) << "Property: set_int " << i;
}

void Property::SetInt64(const std::string &name, const int64_t &i) {
  auto type = proto::ValueProto::INT;
  auto entry = property_.add_entrys();
  entry->set_name(name);
  entry->set_type(type);
  entry->set_i(i);
  VLOG(3) << "Property: set_int " << i << " name: " << name;
}

void Property::SetInt64s(const std::vector<int64_t> &v) {
  auto type = proto::ValueProto::INTS;
  auto entry = property_.add_entrys();
  entry->set_type(type);
  for (auto e : v) {
    entry->add_ints(e);
  }
  VLOG(3) << "Property: set_ints " << v.size();
}

void Property::SetInt64s(const std::string &name,
                         const std::vector<int64_t> &v) {
  auto type = proto::ValueProto::INTS;
  auto entry = property_.add_entrys();
  entry->set_name(name);
  entry->set_type(type);
  for (auto i : v) {
    entry->add_ints(i);
  }
  VLOG(3) << "Property: set_ints " << v[0] << " name: " << name;
}

void Property::SetString(const std::string &s) {
  auto type = proto::ValueProto::STRING;
  auto entry = property_.add_entrys();
  entry->set_type(type);
  entry->set_s(s);
  VLOG(3) << "Property: set_string with value : " << s;
}

void Property::SetString(const std::string &name, const std::string &s) {
  auto type = proto::ValueProto::STRING;
  auto entry = property_.add_entrys();
  entry->set_name(name);
  entry->set_type(type);
  entry->set_s(s);
  VLOG(3) << "Property: set_string " << s << " name: " << name;
}

void Property::SetStrings(const std::vector<std::string> &v) {
  auto type = proto::ValueProto::STRINGS;
  auto entry = property_.add_entrys();
  entry->set_type(type);
  for (auto i : v) {
    entry->add_strings(i);
  }
  VLOG(3) << "Property: set_strings " << v.size();
}

void Property::SetStrings(const std::string &name,
                          const std::vector<std::string> &v) {
  auto type = proto::ValueProto::STRINGS;
  auto entry = property_.add_entrys();
  entry->set_name(name);
  entry->set_type(type);
  for (auto i : v) {
    entry->add_strings(i);
  }
  VLOG(3) << "Property: set_strings " << v[0] << " name: " << name;
}

}  // namespace jit
}  // namespace paddle
