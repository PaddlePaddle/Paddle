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
#include "paddle/framework/scope.h"
#include "paddle/framework/ddim.h"
#include "paddle/framework/op_desc.pb.h"

namespace paddle {
namespace framework {

class DeviceContext {};
class CpuContext : public DeviceContext {};
class GpuContext : public DeviceContext {};

/**
 * @brief Operator is used to do some computation.
 *
 * We use a OpDesc proto Message to describe and create a operator.
 * Operator will get the Variables from scope and computing resource from DeviceContext.
 */
class OperatorBase {
 public:
  explicit OperatorBase(const OpDesc& desc);
  virtual ~OperatorBase() {}

  /// initialize Attributes of this OP from proto message desc.attrs()
  /// you should derive this function to init the attr you need in OP.
  virtual void InitializeAttributes() = 0;

  virtual void InferShape(const Scope* scope) const = 0;

  /// when implement an Op, your should implement this function.
  virtual void Run(Scope* scope, DeviceContext* device_context) const = 0;

  std::string DebugString();
  Variable* input(Scope* scope, int index);
  Variable* output(Scope* scope, int index);

protected:
 const OpDesc desc_;
 std::vector<std::string> inputs_;
 std::vector<std::string> outputs_;
};

} // namespace framework
} // namespace paddle