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

#include "paddle/framework/framework.pb.h"
#include "paddle/framework/op_info.h"
#include "paddle/framework/scope.h"
#include "paddle/framework/tensor.h"

namespace paddle {
namespace framework {

class Executor {
 public:
  explicit Executor(const std::vector<platform::Place>& places);
  ~Executor();

  /* @Brief
   * Runtime evaluation of the given ProgramDesc under certain Scope
   *
   * @param
   *  ProgramDesc
   *  Scope
   */
  void Run(const ProgramDesc&, Scope*, int);

 private:
  std::vector<platform::DeviceContext*> device_contexts_;
};

/* @Brief
 * Pruning the graph
 *
 * @param
 *  ProgramDesc
 *
 * @return
 *  vector<bool> Same size as ops. Indicates whether an op should be run.
 */
std::vector<bool> Prune(const ProgramDesc& pdesc, int block_id);

}  // namespace framework
}  // namespace paddle
