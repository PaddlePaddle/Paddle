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

#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace framework {

class Executor {
 public:
  // TODO(dzhwinter) : Do not rely on this function, it will be removed
  explicit Executor(const platform::DeviceContext& device)
      : Executor(device.GetPlace()) {}

  explicit Executor(const platform::Place& place);

  /* @Brief
   * Runtime evaluation of the given ProgramDesc under certain Scope
   *
   * @param
   *  ProgramDesc
   *  Scope
   */
  void Run(const ProgramDesc&, Scope*, int, bool create_local_scope = true,
           bool create_vars = true);

  void Run(const ProgramDesc& program, Scope* scope,
           std::map<std::string, const LoDTensor*>& feed_targets,
           std::map<std::string, LoDTensor*>& fetch_targets,
           const std::string& feed_holder_name = "feed",
           const std::string& fetch_holder_name = "fetch");

 private:
  const platform::Place place_;
};

}  // namespace framework
}  // namespace paddle
