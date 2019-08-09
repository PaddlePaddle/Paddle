// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace paddle {
namespace lite {

/*
 * Compatible interfaces for all the different kinds of opdesc. All the OpDesc
 * classes should implement this.
 * NOTE Some interfaces are weried, we remain them unchanged to keep compatible
 * with framework::OpDesc in Fluid framework.
 */
class OpDescAPI {
 public:
  // The AttrType is used to make the proto::AttrType portable.
  enum class AttrType {
    INT = 0,
    FLOAT = 1,
    STRING = 2,
    INTS = 3,
    FLOATS = 4,
    STRINGS = 5,
    BOOLEAN = 6,
    BOOLEANS = 7,
    BLOCK = 8,
    LONG = 9,
    BLOCKS = 10,
    LONGS = 11,
    UNK,
  };

  virtual ~OpDescAPI() = default;

  /// Get operator's type.
  virtual std::string Type() const = 0;
  /// Set operator's type.
  virtual void SetType(const std::string& type) = 0;
  /// Get arguments given the parameter.
  virtual std::vector<std::string> Input(const std::string& param) const = 0;
  /// Get parameters.
  virtual std::vector<std::string> InputArgumentNames() const = 0;
  /// Get arguments given the parameter.
  virtual std::vector<std::string> Output(const std::string& param) const = 0;
  /// Get parameters.
  virtual std::vector<std::string> OutputArgumentNames() const = 0;
  /// Set a input given the parameter and arguments.
  virtual void SetInput(const std::string& param,
                        const std::vector<std::string>& args) = 0;
  virtual void SetOutput(const std::string& param,
                         const std::vector<std::string>& args) = 0;
  /// Tell whether this desc has an attribute.
  virtual bool HasAttr(const std::string& name) const = 0;

  /// Get the type of an attribute.
  virtual AttrType GetAttrType(const std::string& name) const = 0;

  virtual std::vector<std::string> AttrNames() const = 0;

  /// Set an attribute.
  template <typename T>
  void SetAttr(const std::string& name, const T& v);

  /// Get an attribute.
  template <typename T>
  T GetAttr(const std::string& name) const;

  std::string Repr() const {
    std::stringstream ss;
    ss << Type();
    ss << "(";
    for (auto& arg : InputArgumentNames()) {
      ss << arg << ":";
      for (auto val : Input(arg)) {
        ss << val << " ";
      }
    }
    ss << ") -> (";
    for (auto& arg : OutputArgumentNames()) {
      ss << arg << ":";
      for (auto val : Output(arg)) {
        ss << val << " ";
      }
    }
    ss << ")";
    return ss.str();
  }
};

}  // namespace lite
}  // namespace paddle
