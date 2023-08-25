// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <unordered_set>
#include <vector>

#include "paddle/cinn/adt/equation.h"
#include "paddle/cinn/adt/equation_context.h"
#include "paddle/cinn/common/equation_graph_topo_walker.h"

namespace cinn::adt::equation {

// clang-format off
/*
Graph = ([Variale], [Function], [Edge Variable Function], [Edge Function Variable])
Edge T0 T1 = (T0, T1)
*/
// clang-format on
class Graph final {
 public:
  using V2Fs = std::unordered_map<const Variable, std::vector<const Function*>>;
  using F2Vs = std::unordered_map<const Function*, std::vector<const Variable>>;

  explicit Graph(const std::shared_ptr<std::vector<Equation>>& equations)
      : functions_(equations),
        variable2next_functions_(std::make_shared<V2Fs>()),
        function2in_variables_(std::make_shared<F2Vs>()),
        function2out_variables_(std::make_shared<F2Vs>()) {
    for (const Function& function : *functions_) {
      CollectVariablesAndEdges(function);
    }
  }

  using VariableVisitorT = std::function<void(const Variable)>;
  using FunctionVisitorT = std::function<void(const Function*)>;
  using F4VVisitor =
      std::function<void(const Variable, const FunctionVisitorT&)>;
  using V4FVisitor =
      std::function<void(const Function*, const VariableVisitorT&)>;

  static F4VVisitor Merge(const F4VVisitor& lhs, const F4VVisitor& rhs) {
    return [=](const Variable variable,
               const std::function<void(const Function*)>& Visit) {
      lhs(variable, Visit);
      rhs(variable, Visit);
    };
  }

  static V4FVisitor Merge(const V4FVisitor& lhs, const V4FVisitor& rhs) {
    return [=](const Function* function,
               const std::function<void(const Variable)>& Visit) {
      lhs(function, Visit);
      rhs(function, Visit);
    };
  }

  F4VVisitor GetNextFunctionsVisitor() const {
    return [=](const Variable variable,
               const std::function<void(const Function*)>& Visit) {
      const auto iter = variable2next_functions_->find(variable);
      if (iter == variable2next_functions_->end()) {
        return;
      }
      for (const Function* function : iter->second) {
        Visit(function);
      }
    };
  }

  V4FVisitor GetInputVariablesVisitor() const {
    return [=](const Function* function,
               const std::function<void(const Variable)>& Visit) {
      const auto iter = function2in_variables_->find(function);
      if (iter == function2in_variables_->end()) {
        return;
      }
      for (const Variable variable : iter->second) {
        Visit(variable);
      }
    };
  }

  V4FVisitor GetOutputVariablesVisitor() const {
    return [=](const Function* function,
               const std::function<void(const Variable)>& Visit) {
      const auto iter = function2out_variables_->find(function);
      if (iter == function2out_variables_->end()) {
        return;
      }
      for (const Variable variable : iter->second) {
        Visit(variable);
      }
    };
  }

  static EquationGraphTopoWalker<const Variable, const Function*>
  GetMergedWalker(const Graph& lhs, const Graph& rhs) {
    return EquationGraphTopoWalker<const Variable, const Function*>(
        /*NextFunctionsVisitor=*/Merge(lhs.GetNextFunctionsVisitor(),
                                       rhs.GetNextFunctionsVisitor()),
        /*InputVariablesVisitor=*/
        Merge(lhs.GetInputVariablesVisitor(), rhs.GetInputVariablesVisitor()),
        /*OutputVariablesVisitor=*/
        Merge(lhs.GetOutputVariablesVisitor(),
              rhs.GetOutputVariablesVisitor()));
  }

  EquationGraphTopoWalker<const Variable, const Function*> GetWalker() const {
    return EquationGraphTopoWalker<const Variable, const Function*>(
        /*NextFunctionsVisitor=*/GetNextFunctionsVisitor(),
        /*InputVariablesVisitor=*/GetInputVariablesVisitor(),
        /*OutputVariablesVisitor=*/GetOutputVariablesVisitor());
  }

 private:
  void CollectVariablesAndEdges(const Function& function) {
    std::unordered_set<Variable> in_variables;
    std::unordered_set<Variable> out_variables;

    // clang-format off
    function >> match {
      [&](const Identity<tOut<Iterator>, tIn<Iterator>>& identity) {
        const auto& [out_iter, in_iter] = identity.tuple();
        out_variables.emplace(out_iter.value());
        in_variables.emplace(in_iter.value());
      },
      [&](const Identity<tOut<Index>, tIn<Index>>& identity) {
        const auto& [out_index, in_index] = identity.tuple();
        out_variables.emplace(out_index.value());
        in_variables.emplace(in_index.value());
      },
      [&](const Dot<List<Stride>, tOut<Index>, tIn<List<Iterator>>>& dot) {
        const auto& [strides, out_index, in_iterators] = dot.tuple();
        out_variables.emplace(out_index.value());
        in_variables.emplace(in_iterators.value().begin(),
                             in_iterators.value().end());
      },
      [&](const UnDot<List<Stride>, tOut<List<Iterator>>, tIn<Index>>& undot) {
        const auto& [strides, out_iterators, in_index] = undot.tuple();
        out_variables.emplace(out_iterators.value().begin(),
                              out_iterators.value().end());
        in_variables.emplace(in_index.value());
      },
      [&](const ConstructFakeOpPlaceHolder<tOut<FakeOpPlaceHolder>,
                                           tIn<List<Index>>>& construct_fake) {
        const auto& [out_fake_op_placeholder, in_indexes] =
            construct_fake.tuple();
        out_variables.emplace(out_fake_op_placeholder.value());
        in_variables.emplace(in_indexes.value().begin(),
                             in_indexes.value().end());
      }
    };
    // clang-format on

    for (const Variable& variable : in_variables) {
      variables_.insert(variable);
      (*variable2next_functions_)[variable].push_back(&function);
      (*function2in_variables_)[&function].push_back(variable);
      v2f_edges_.emplace_back(std::pair{variable, &function});
    }
    for (const Variable& variable : out_variables) {
      variables_.insert(variable);
      (*function2out_variables_)[&function].push_back(variable);
      f2v_edges_.emplace_back(std::pair{&function, variable});
    }
  }

  std::shared_ptr<std::vector<Function>> functions_;
  // tNext [Function] <- Variable
  std::shared_ptr<V2Fs> variable2next_functions_;
  // tIn [Variable] <- Function
  std::shared_ptr<F2Vs> function2in_variables_;
  // tOut [Variable] <- Function
  std::shared_ptr<F2Vs> function2out_variables_;
  std::unordered_set<Variable> variables_;

  // For debug
  std::vector<std::pair<const Variable, const Function*>> v2f_edges_;
  std::vector<std::pair<const Function*, const Variable>> f2v_edges_;
};

}  // namespace cinn::adt::equation
