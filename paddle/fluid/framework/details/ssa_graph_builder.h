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

#include "paddle/fluid/framework/details/ssa_graph.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace framework {
namespace details {

class SSAGraphBuilder {
 public:
  struct Context {
    std::vector<std::unordered_map<std::string,
                                   std::vector<std::unique_ptr<VarHandle>>>>
        vars_;
    // aux variables to represent dependency. Useful to resolve data hazard.
    std::vector<std::unique_ptr<VarHandleBase>> dep_vars_;
    std::vector<std::unique_ptr<OpHandleBase>> ops_;
  };

  SSAGraphBuilder() {}
  virtual ~SSAGraphBuilder() {}
  virtual std::unique_ptr<SSAGraph> Build(const ProgramDesc &program) const = 0;

  DISABLE_COPY_AND_ASSIGN(SSAGraphBuilder);

 protected:
  /**
   * We only handle write after read(WAR), since it should not have a write
   * after write in program. If there are write after write operators, we need
   * prune them.
   *
   * https://en.wikipedia.org/wiki/Hazard_(computer_architecture)#Write_after_read_(WAR)
   */
  static void PolishGraphToSupportDataHazards(Context *graph);

  static VarHandle *CreateOrGetLatestVarHandle(Context *graph,
                                               const std::string &each_var_name,
                                               const platform::Place &place,
                                               size_t place_offset);

  static void CreateOpOutput(Context *graph, OpHandleBase *op_handle,
                             const std::string &each_var_name,
                             const platform::Place &place, size_t place_offset);

  static void AddOutputToLeafOps(Context *graph);

  static void PrintGraphviz(const Context &graph, std::ostream &sout);

  static std::unique_ptr<SSAGraph> ContextToSSAGraph(
      std::unique_ptr<Context> &&graph);
};
}  // namespace details
}  // namespace framework
}  // namespace paddle
