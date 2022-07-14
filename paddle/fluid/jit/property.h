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
#include <vector>

#include "glog/logging.h"
#include "paddle/fluid/jit/property.pb.h"

namespace paddle {
namespace jit {

// convert between std::vector and protobuf repeated.
template <typename T>
inline std::vector<T> RepeatedToVector(
    const google::protobuf::RepeatedField<T> &repeated_field) {
  std::vector<T> ret;
  ret.reserve(repeated_field.size());
  std::copy(
      repeated_field.begin(), repeated_field.end(), std::back_inserter(ret));
  return ret;
}

template <typename T, typename RepeatedField>
inline void VectorToRepeated(const std::vector<T> &vec,
                             RepeatedField *repeated_field) {
  repeated_field->Clear();
  repeated_field->Reserve(vec.size());
  for (const auto &elem : vec) {
    *repeated_field->Add() = elem;
  }
}

// Specialize vector<bool>.
template <typename RepeatedField>
inline void VectorToRepeated(const std::vector<bool> &vec,
                             RepeatedField *repeated_field) {
  repeated_field->Clear();
  repeated_field->Reserve(vec.size());
  for (auto elem : vec) {
    *repeated_field->Add() = elem;
  }
}


class Property {
 public:
  explicit Property() {}

  // Explicitly implement the copy constructor for auto parallel
  Property(const Property &other)
      : property_(other.property_),
        original_id_(other.original_id_) {}

  Property &operator=(const Property &other) {
    property_ = other.property_;
    original_id_ = other.original_id_;
    return *this;
  }

  proto::PropertyVals *Proto() { return &property_; }

  const proto::PropertyVals *Proto() const { return &property_; }

  bool SetFloat(const float& f, std::string name="");
  bool SetFloats(const std::vector<float>& v, std::string name="");

  bool SetInt64(const int64_t& f, std::string name="");
  bool SetInt64s(const std::vector<int64_t>& v, std::string name="");

  bool SetString(const std::string& s, std::string name="");
  bool SetStrings(const std::vector<std::string>& v, std::string name="");

  // The Id() and OriginalId() are only used for auto parallel.
  uint64_t Id() const { return id_; }
  uint64_t OriginalId() const { return original_id_; }
  void SetOriginalId(uint64_t original_id) { original_id_ = original_id; }

private:
  void AddEntry();

private:
  proto::PropertyVals property_;
  
  // This thread-safe implementation seems to be redudent since the neural
  // networks are usually constructed in a single thread.
  static uint64_t GenerateId() {
    static std::atomic<std::uint64_t> uid{0};
    return ++uid;
  }

  // Note: the id_ is unique for all Property (only for auto parallel).
  uint64_t id_ = GenerateId();
  // Note: the orignal_id_ is used for referring to the original Property
  // that the current Property is built from (only for auto parallel).
  // The default original_id_ is same as the id_, which means the
  // current Property is not built from the other one.
  uint64_t original_id_ = id_;
};

bool operator==(const Property &left, const Property &right);

}  // namespace jit
}  // namespace paddle
