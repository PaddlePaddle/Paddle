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

#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/utils/Error.h"

#include <paddle/framework/op_desc.pb.h>

namespace paddle {
namespace framework {

class OpContext {};

/**
 * @brief Operator is used to do some computation.
 *
 * We use a OpDesc proto Message to describe and create a operator.
 * Operator will get the Variables and computing resource from OpContext when Run.
 */
class Operator {
 public:
  explicit Operator(const OpDesc& desc);
  virtual ~Operator() {}

  Error InitializeAttrs(const std::vector<AttrDesc>& attrs);

  /**
   * InferShape is used to infer the shape of tensors related to this Operator.
   */
  virtual void InferShape() = 0;

  /**
   * Run take a OpContext as parameter.
   *
   * 1. it will get input/output variable from OpContext.scope
   * 2. It will get computing resource such as cpu/gpu from OpContext.
   */
  virtual void Run(OpContext *context) const = 0;
  const std::string DebugString() const {
    return op_desc_.ShortDebugString();
  }

protected:
  OpDesc op_desc_;
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
};

} // namespace framework
} // namespace paddle