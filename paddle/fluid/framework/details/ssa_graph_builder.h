//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/details/op_handle_base.h"
#include "paddle/fluid/framework/details/var_handle.h"

#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/platform/place.h"

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace details {

// all variable in each devices.
// The outside vector is the device vector. Each element of this vector is a
// map from variable name to variables. The variables, who have the same name,
// will have a differsent version. The offset in the
// `std::vector<std::unique_ptr<VarHandle>>` is the version of varaibles.
typedef std::vector<
    std::unordered_map<std::string, std::vector<std::unique_ptr<VarHandle>>>>
    GraphVars;

// aux variables to represent dependency. Useful to resolve data hazard.
typedef std::unordered_set<std::unique_ptr<VarHandleBase>> GraphDepVars;

// all operators. NOTE that even we use a vector here, the operators is
// unordered.
typedef std::vector<std::unique_ptr<OpHandleBase>> GraphOps;

class SSAGraphBuilder : public ir::Pass {
 public:
  SSAGraphBuilder() {}
  virtual ~SSAGraphBuilder() {}

  virtual int GetVarDeviceID(const std::string &var_name) const = 0;

  DISABLE_COPY_AND_ASSIGN(SSAGraphBuilder);

 protected:
  /**
   * We only handle write after read(WAR), since it should not have a write
   * after write in program. If there are write after write operators, we need
   * prune them.
   *
   * https://en.wikipedia.org/wiki/Hazard_(computer_architecture)#Write_after_read_(WAR)
   */
  static void PolishGraphToSupportDataHazards(Graph *graph);

  static VarHandle *CreateOrGetLatestVarHandle(Graph *graph, ir::Node *node,
                                               const platform::Place &place,
                                               size_t place_offset);

  // Add an output variable (each_var_name, place, place_offset) to op_handle,
  // which belongs to graph
  static void CreateOpOutput(Graph *graph, OpHandleBase *op_handle,
                             ir::Node *new_node, const platform::Place &place,
                             size_t place_offset);

  static void AddOutputToLeafOps(Graph *graph);
};
}  // namespace details
}  // namespace framework
}  // namespace paddle
