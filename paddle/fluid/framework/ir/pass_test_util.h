// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include <utility>
#include <vector>

#include "paddle/common/ddim.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/phi/common/place.h"

namespace paddle {
namespace framework {
namespace ir {

// -------------------------- helper functions --------------------------------
namespace test {

/// The pair describing correlation between {input/output name, variable name}.
using InOutVarNamePair = std::pair<std::string, std::string>;
/// The pair describing number of occurrences of given op type.
using OpTypeCountPair = std::pair<std::string, int>;

///
/// @brief      Creates the specified operator and sets up its inputs/outputs.
///
/// @param      prog          The program descriptor to which we add new op.
/// @param[in]  op_type_name  The operator type name.
/// @param[in]  inputs        The vector of input pairs: {input_name, variable
///                           name}
/// @param[in]  outputs       The vector of output pairs {output_name, variable}
/// @param[in]  use_mkldnn    The flag deciding whether or not to set
///                           'use_mkldnn' attribute.
///
/// @return     Returns pointer to the created operator descriptor.
///
OpDesc* CreateOp(ProgramDesc* prog,
                 const std::string& op_type_name,
                 const std::vector<InOutVarNamePair>& inputs,
                 const std::vector<InOutVarNamePair>& outputs,
                 bool use_mkldnn = true);

///
/// @brief      Check whether node 'to' is reachable from node 'from' in graph.
///
/// @param[in]  graph  The graph we're checking for reachability.
/// @param[in]  from   The 'from' node name.
/// @param[in]  to     The 'to' node name.
///
/// @return     True if there is connection between nodes 'from' and 'to'.
///
bool TestIsReachable(const Graph& graph, std::string from, std::string to);

///
/// @brief      Search through graph and counts provided operator occurrences.
///
/// @param[in]  graph          The graph we search through.
/// @param[in]  op_type_count  The vector of pairs {op_type_name, op count}
///
/// @note       After going through all graph nodes this function asserts
///             whether counted number for each requested op is as expected.
///
/// @return     Returns true if occurrences of all ops is as expected.
///
bool AssertOpsCount(const Graph& graph,
                    std::vector<OpTypeCountPair> op_type_count);

///
/// @brief      Builds a program descriptor.
///
/// @param[in]  transient_vars   The vector of transient variables names.
/// @param[in]  persistent_vars  The vector of persistent variables names. Those
///                              will have persistable attribute set to true.
///
/// @return     The program descriptor object.
///
ProgramDesc BuildProgramDesc(const std::vector<std::string>& transient_vars,
                             const std::vector<std::string>& persistent_vars);

///
/// @brief      Execute pass on provided graph and perform checks.
///
/// @note       Check whether the balance of removed and added nodes after pass
///             is as expected.
///
/// @param      graph                The graph we run pass on.
/// @param[in]  from                 The name of a 'starting' node sequence in a
///                                  graph. This would be used to test for
///                                  correct node connections.
/// @param[in]  to                   The name of a 'ending' node sequence in a
///                                  graph. This would be used to test for
///                                  correct node connections.
/// @param[in]  removed_nodes_count  The number of nodes we expect will be
///                                  removed/fused after pass execution.
/// @param[in]  added_nodes_count    The number of nodes we expect will be added
///                                  after pass execution.
///
/// @return     Return true if all checks passed, otherwise false.
///
bool RunPassAndAssert(Graph* graph,
                      const std::string& pass_name,
                      const std::string& from,
                      const std::string& to,
                      int removed_nodes_count,
                      int added_nodes_count = 0);

///
/// @brief      Initializes the tensor memory holder.
///
/// @param[in]  scope     The scope that manages the variable.
/// @param[in]  place     The place where memory will be allocated.
/// @param[in]  var_name  The variable name.
/// @param[in]  dims      The dimensions of allocated tensor.
///
/// @tparam     T         phi::DenseTensor data type.
///
template <typename T>
void InitDenseTensorHolder(const Scope& scope,
                           const phi::Place& place,
                           const std::string& var_name,
                           const std::vector<int64_t>& dims,
                           const T* data = nullptr);

///
/// @brief      Retrieve operator descriptor from program.
///
/// @param[in]  prog             The program descriptor containing the op we
///                              search for.
/// @param[in]  op_type          The wanted operator type name.
/// @param[in]  output_name      The wanted operator output name.
/// @param[in]  output_arg_name  The wanted operator output argument name.
///
/// @return     The operator descriptor.
///
OpDesc* GetOp(const ProgramDesc& prog,
              const std::string& op_type,
              const std::string& output_name,
              const std::string& output_arg_name);

OpDesc* GetOp(const BlockDesc& block_desc,
              const std::string& op_type,
              const std::string& output_name,
              const std::string& output_arg_name);

}  // namespace test
}  // namespace ir
}  // namespace framework
}  // namespace paddle
