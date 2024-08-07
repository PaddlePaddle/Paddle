// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <algorithm>
#include <atomic>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/jit/property.pb.h"

namespace paddle {
namespace framework {
class Variable;
}
namespace jit {

using Variable = paddle::framework::Variable;

class Property {
 public:
  Property() {}

  // Explicitly implement the copy constructor for auto parallel
  explicit Property(const Property &other)
      : property_(other.property_), original_id_(other.original_id_) {}

  Property &operator=(const Property &other) {
    property_ = other.property_;
    original_id_ = other.original_id_;
    return *this;
  }

  proto::PropertyVals *Proto() { return &property_; }

  const proto::PropertyVals *Proto() const { return &property_; }

  int Size() const;
  std::vector<std::string> Names() const;
  std::unordered_map<std::string, std::shared_ptr<Variable>> Values();

  void SetFloat(const float &f);
  void SetFloat(const std::string &name, const float &f);

  float GetFloat(const std::string &name) const;
  float GetFloat(const int &idx) const;

  void SetFloats(const std::vector<float> &v);
  void SetFloats(const std::string &name, const std::vector<float> &v);

  std::vector<float> GetFloats(const std::string &name);

  void SetInt64(const int64_t &i);
  void SetInt64(const std::string &name, const int64_t &i);

  int64_t GetInt64(const std::string &name);

  void SetInt64s(const std::vector<int64_t> &v);
  void SetInt64s(const std::string &name, const std::vector<int64_t> &v);

  std::vector<int> GetInt64s(const std::string &name);

  void SetString(const std::string &s);
  void SetString(const std::string &name, const std::string &s);

  std::string GetString(const std::string &name);

  void SetStrings(const std::vector<std::string> &v);
  void SetStrings(const std::string &name, const std::vector<std::string> &v);

  std::vector<std::string> GetStrings(const std::string &name);

  void Deserialization(const std::string &path);

  void Serialization(const std::string &path);

  // The Id() and OriginalId() are only used for auto parallel.
  uint64_t Id() const { return id_; }
  uint64_t OriginalId() const { return original_id_; }
  void SetOriginalId(uint64_t original_id) { original_id_ = original_id; }

 private:
  void DeserializationFromString(const std::string &str);

  std::string SerializationToString();

 private:
  proto::PropertyVals property_;

  // This thread-safe implementation seems to be redundant since the neural
  // networks are usually constructed in a single thread.
  static uint64_t GenerateId() {
    static std::atomic<std::uint64_t> uid{0};
    return ++uid;
  }

  // Note: the id_ is unique for all Property (only for auto parallel).
  uint64_t id_ = GenerateId();
  // Note: the original_id_ is used for referring to the original Property
  // that the current Property is built from (only for auto parallel).
  // The default original_id_ is same as the id_, which means the
  // current Property is not built from the other one.
  uint64_t original_id_ = id_;
};

}  // namespace jit
}  // namespace paddle
