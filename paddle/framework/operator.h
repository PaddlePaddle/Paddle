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

namespace paddle {
namespace framework {

class DeviceContext {};
class CpuContext : public DeviceContext {};
class GpuContext : public DeviceContext {};

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

using Attribute =
    boost::variant<boost::blank, int, float, std::string, std::vector<int>,
                   std::vector<float>, std::vector<std::string>>;
using AttributeMap = std::unordered_map<std::string, Attribute>;

/**
 * OperatorBase has the basic element that Net will call to do compute.
 * It have no construct function because CreateOperator(const& op_desc)
 * will parse op_desc and set the input/output/attr properly.
 */
class OperatorBase {
 public:
  virtual ~OperatorBase() {}
  /// when implement an Op, your should implement this function.
  virtual void Run(OpRunContext* context) const = 0;

  /// InferShape infer the size of Variables used by this Operator with
  /// information
  /// inside scope
  virtual void InferShape(const Scope* scope) const = 0;

  std::string DebugString() const;

 public:
  std::string type_;
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
  AttributeMap attrs_;
};

}  // namespace framework
}  // namespace paddle
