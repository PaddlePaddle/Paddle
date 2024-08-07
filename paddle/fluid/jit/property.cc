/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <fstream>
#include <streambuf>
#include <string>

#include "glog/logging.h"

#include "paddle/common/errors.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/jit/property.h"
#include "paddle/phi/core/enforce.h"

namespace paddle::jit {

using Variable = paddle::framework::Variable;

void Property::DeserializationFromString(const std::string &str) {
  PADDLE_ENFORCE_EQ(
      this->Proto()->ParsePartialFromString(str),
      true,
      common::errors::InvalidArgument("Failed to parse pb from string"));
  return;
}

std::string Property::SerializationToString() {
  std::string retv;
  PADDLE_ENFORCE_EQ(this->Proto()->SerializePartialToString(&retv),
                    true,
                    common::errors::InvalidArgument(
                        "Failed to serialize input Desc to string."));
  return retv;
}

void Property::Deserialization(const std::string &path) {
  std::ifstream ifs(path, std::ios::binary | std::ios::in);
  std::string str((std::istreambuf_iterator<char>(ifs)),
                  std::istreambuf_iterator<char>());
  DeserializationFromString(str);
  ifs.close();
  return;
}

void Property::Serialization(const std::string &path) {
  std::string str = SerializationToString();
  std::ofstream ofs(path, std::ios::binary | std::ios::out);
  ofs << str;
  ofs.close();
  return;
}

int Property::Size() const { return property_.entrys_size(); }

std::vector<std::string> Property::Names() const {
  std::vector<std::string> res;
  for (int i = 0; i < Size(); i++) {
    auto entry = property_.entrys(i);
    if (entry.has_name()) {
      res.push_back(entry.name());
    } else {
      LOG(WARNING) << "JIT::Property entry " << i
                   << " not has name! Please check whether it is reasonable.";
    }
  }
  return res;
}

std::unordered_map<std::string, std::shared_ptr<Variable>> Property::Values() {
  std::unordered_map<std::string, std::shared_ptr<Variable>> res;
  using ValueProto = proto::ValueProto;
  for (int i = 0; i < Size(); i++) {
    auto entry = property_.entrys(i);
    if (entry.has_name()) {
      auto &n = entry.name();
      // remove Class Name suffix
      auto key = n.substr(n.find_first_of('.') + 1);
      std::shared_ptr<Variable> var(new Variable());
      auto type = entry.type();
      switch (type) {
        case ValueProto::FLOAT:
          *var->GetMutable<float>() = GetFloat(n);
          break;
        case ValueProto::INT:
          *var->GetMutable<int>() = static_cast<int>(GetInt64(n));
          break;
        case ValueProto::STRING:
          *var->GetMutable<paddle::framework::String>() = GetString(n);
          break;
        case ValueProto::FLOATS:  // NOLINT
          *var->GetMutable<std::vector<float>>() = GetFloats(n);
          break;
        case ValueProto::INTS:
          *var->GetMutable<std::vector<int>>() = GetInt64s(n);
          break;
        case ValueProto::STRINGS:
          *var->GetMutable<std::vector<std::string>>() = GetStrings(n);
          break;
        default:
          break;
      }
      res[key] = var;
      VLOG(3) << "read property: " << n << " to " << key;
    } else {
      LOG(WARNING) << "JIT::Property entry " << i
                   << " not has name! Please check whether it is reasonable.";
    }
  }
  return res;
}

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
      PADDLE_ENFORCE(e.has_type() && e.type() == proto::ValueProto::FLOAT,
                     common::errors::PreconditionNotMet(
                         "JIT::Property GetFloat: idx=%d type "
                         "is not float. Expect %d, but %d",
                         i,
                         proto::ValueProto::FLOAT,
                         e.type()));

      return e.f();
    }
  }

  PADDLE_THROW(common::errors::NotFound(
      "JIT::Property GetFloat: name: %s not found", name));
  return 0;
}

float Property::GetFloat(const int &idx) const {
  PADDLE_ENFORCE_EQ(
      idx < Size() && idx >= 0,
      true,
      common::errors::OutOfRange(
          "JIT::Property GetFloat: idx=%d out of range %d", idx, Size()));

  auto e = property_.entrys(idx);
  if (e.has_f()) {
    return e.f();
  }

  PADDLE_THROW(common::errors::InvalidArgument(
      "JIT::Property GetFloat: input idx (%d) element is not a float.", idx));
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

std::vector<float> Property::GetFloats(const std::string &name) {
  for (int i = 0; i < Size(); i++) {
    auto e = property_.entrys(i);

    if (e.has_name() && e.name() == name) {
      PADDLE_ENFORCE(
          e.has_type() && e.type() == proto::ValueProto::FLOATS,
          common::errors::PreconditionNotMet(
              "JIT::Property GetFloats: idx=%d type is not floats.", i));

      // auto items = e.floats();
      return std::vector<float>(e.floats().begin(), e.floats().end());
    }
  }

  PADDLE_THROW(common::errors::NotFound(
      "JIT::Property GetFloats: name: %s not found", name));
  return std::vector<float>();
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

int64_t Property::GetInt64(const std::string &name) {
  for (int i = 0; i < Size(); i++) {
    auto e = property_.entrys(i);

    if (e.has_name() && e.name() == name) {
      PADDLE_ENFORCE(e.has_type() && e.type() == proto::ValueProto::INT,
                     common::errors::PreconditionNotMet(
                         "JIT::Property GetInt64: idx=%d type is not int.", i));

      return e.i();
    }
  }

  PADDLE_THROW(common::errors::NotFound(
      "JIT::Property GetInt64: name: %s not found", name));
  return 0;
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

std::vector<int> Property::GetInt64s(const std::string &name) {
  for (int i = 0; i < Size(); i++) {
    auto e = property_.entrys(i);

    if (e.has_name() && e.name() == name) {
      PADDLE_ENFORCE(
          e.has_type() && e.type() == proto::ValueProto::INTS,
          common::errors::PreconditionNotMet(
              "JIT::Property GetInt64s: idx=%d type is not ints.", i));

      auto items = e.ints();
      std::vector<int> res;
      std::transform(items.begin(),
                     items.end(),
                     std::back_inserter(res),
                     [](const int64_t &v) { return static_cast<int>(v); });
      return res;
    }
  }

  PADDLE_THROW(common::errors::NotFound(
      "JIT::Property GetInt64s: name: %s not found", name));
  return {};
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

std::string Property::GetString(const std::string &name) {
  for (int i = 0; i < Size(); i++) {
    auto e = property_.entrys(i);

    if (e.has_name() && e.name() == name) {
      PADDLE_ENFORCE(
          e.has_type() && e.type() == proto::ValueProto::STRING,
          common::errors::PreconditionNotMet(
              "JIT::Property GetString: idx=%d type is not string.", i));
      return e.s();
    }
  }

  PADDLE_THROW(common::errors::NotFound(
      "JIT::Property GetString: name: %s not found", name));
  return {};
}

void Property::SetStrings(const std::vector<std::string> &v) {
  auto type = proto::ValueProto::STRINGS;
  auto entry = property_.add_entrys();
  entry->set_type(type);
  for (auto const &i : v) {
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
  for (auto const &i : v) {
    entry->add_strings(i);
  }
  VLOG(3) << "Property: set_strings " << v[0] << " name: " << name;
}

std::vector<std::string> Property::GetStrings(const std::string &name) {
  for (int i = 0; i < Size(); i++) {
    auto e = property_.entrys(i);

    if (e.has_name() && e.name() == name) {
      PADDLE_ENFORCE(
          e.has_type() && e.type() == proto::ValueProto::STRINGS,
          common::errors::PreconditionNotMet(
              "JIT::Property GetStrings: idx=%d type is not strings.", i));

      // auto items = e.strings();
      return std::vector<std::string>(e.strings().begin(), e.strings().end());
    }
  }

  PADDLE_THROW(common::errors::NotFound(
      "JIT::Property GetStrings: name: %s not found", name));
  return {};
}

}  // namespace paddle::jit
