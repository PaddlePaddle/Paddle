// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/utils/registry.h"

#define CINN_REGISTER_PASS(name)                                        \
  CINN_REGISTRY_REGISTER(::cinn::hlir::framework::PassFunctionRegister, \
                         PassFunctionRegister,                          \
                         name)

namespace cinn {
namespace hlir {
namespace framework {

class PassFunctionRegister;
typedef std::function<void(Graph* g)> PassFunction;

/**
 * \brief Given an attribute of graph, find the pass that generates this
 * attribute.
 * @param attr_name Name of the graph attribute.
 * @return The pass that generates it.
 */
const PassFunctionRegister* FindPassDep(const std::string& attr_name);

class PassFunctionRegister
    : public FunctionRegEntryBase<PassFunctionRegister, PassFunction> {
 public:
  bool change_structure{false};
  //! dependencies on operator attributes
  std::vector<std::string> op_attr_dependency{};
  //! dependencies on attributes in the graph
  std::vector<std::string> graph_attr_dependency{};
  //! generated targets of graph attributes
  std::vector<std::string> graph_attr_targets{};

  /**
   * \brief Imply whether this pass will change the Graph's structure.
   * @param in A bool variable implying whether this pass will change the
   * Graph's structure.
   * @return Reference to self.
   */
  PassFunctionRegister& set_change_structure(bool in) {
    change_structure = in;
    return *this;
  }

  /**
   * \brief Declare that this pass will generate the given graph attribute name
   *        once it is applied on the graph.
   * @param attr_name Name of the graph attribute.
   * @return Reference to self.
   */
  PassFunctionRegister& provide_graph_attr(const std::string& attr_name) {
    graph_attr_targets.push_back(attr_name);
    return *this;
  }

  /**
   * \brief Declare this pass requires the given operator attribute to be
   *        available before being applied on the graph.
   * @param attr_name Name of the attribute.
   * @return Reference to self.
   */
  PassFunctionRegister& depend_op_attr(const std::string& attr_name) {
    op_attr_dependency.push_back(attr_name);
    return *this;
  }

  /**
   * \brief Declare this pass requires the given graph attribute to be
   *        available before being applied on the graph.
   * @param attr_name Name of the attribute.
   * @return Reference to self.
   */
  PassFunctionRegister& depend_graph_attr(const std::string& attr_name) {
    graph_attr_dependency.push_back(attr_name);
    return *this;
  }
};

const PassFunctionRegister* FindPassDep(const std::string& attr_name);

/**
 * \brief Apply a sequence of passes on a graph.
 * @param g The input graph to apply passes on.
 * @param passes The sequence of pass.
 * @return The graph after being modified by the passes.
 */
void ApplyPasses(Graph* g, const std::vector<std::string>& passes);

// Apply a single pass on a graph.
inline void ApplyPass(Graph* g, const std::string& pass) {
  return ApplyPasses(g, {pass});
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
