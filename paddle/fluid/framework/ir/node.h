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

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include "paddle/fluid/platform/macros.h"
#include "paddle/fluid/platform/variant.h"

namespace paddle {
namespace framework {

class Node {
 public:
  enum class Type { kNone = -1, kOperation, kVariable };

  Node() {}
  virtual ~Node() {}

  int64_t ID() const { return id_; }

  std::string Name() const { return name_; }

  virtual std::string ToString() const {
    return Name() + "(" + std::to_string(ID()) + ")";
  }

  virtual std::string DebugString() const = 0;

  Type NodeType() const { return type_; }

  std::vector<Node *> inputs;
  std::vector<Node *> outputs;

 protected:
  std::map<std::string, std::vector<boost::any>> attrs_;
  int64_t id_ = 0;
  std::string name_;
  Type type_;

  DISABLE_COPY_AND_ASSIGN(Node);
};

}  // namespace framework
}  // namespace paddle
