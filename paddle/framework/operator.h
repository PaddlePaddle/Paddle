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

#include <boost/variant.hpp>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/framework/op_desc.pb.h"
#include "paddle/framework/scope.h"
#include "paddle/utils/Error.h"
#include "paddle/framework/attr_checker.h"

namespace paddle {
namespace framework {

class DeviceContext {};
class CPUContext : public DeviceContext {};
class GPUContext : public DeviceContext {};

/**
 * OpRunContext is the only parameter of Operator's Run function.
 * Run will get input/output variables, state such as momentum and
 * device resource such as CUDA stream, cublas handle, etc. from
 * OpRunContext. User should construct it before run the Operator.
 */
class OpRunContext {
 public:
  Scope* scope;
  DeviceContext* device_context;
};

/**
 * OperatorBase has the basic element that Net will call to do compute.
 * It have no construct function because CreateOperator(const& op_desc)
 * will parse op_desc and set the input/output/attr properly.
 */
class OperatorBase {
 public:
  virtual ~OperatorBase() {}

  void Init(const OpDesc& op_desc, AttributeMap& attrs);

  std::string type() const {
    return desc_.type();
  }

  Variable* Input(Scope* scope, int index) const;
  Variable* Output(Scope* scope, int index) const;

  Attribute GetAttr(std::string name);

  inline const AttributeMap attrs() const {
    return attrs_;
  }

  inline const std::vector<std::string> inputs() const {
    return inputs_;
  }

  inline const std::vector<std::string> outputs() const {
    return outputs_;
  }

  std::string DebugString() const;

  /// InferShape infer the size of Variables used by this Operator with
  /// information
  /// inside scope
  void InferShape(Scope* scope) const;

  /// when implement an Op, your should implement this function.
  virtual void Run(OpRunContext* context) const = 0;

 protected:
  OpDesc desc_;
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
  AttributeMap attrs_;
};

}  // namespace framework
}  // namespace paddle
