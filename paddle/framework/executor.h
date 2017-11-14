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

#include "paddle/framework/op_info.h"
#include "paddle/framework/program_desc.h"
#include "paddle/framework/scope.h"
#include "paddle/framework/tensor.h"

namespace paddle {
namespace framework {

class Executor {
 public:
  explicit Executor(const std::vector<platform::Place>& places);
  explicit Executor(const platform::DeviceContext& devices);
  ~Executor();

  /* @Brief
   * Runtime evaluation of the given ProgramDesc under certain Scope
   *
   * @param
   *  ProgramDesc
   *  Scope
   */
  void Run(const ProgramDescBind&, Scope*, int, bool create_local_scope = true);

 private:
  std::vector<const platform::DeviceContext*> device_contexts_;
  bool own_;
};

}  // namespace framework
}  // namespace paddle
