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
  using V2Fs =
      std::unordered_map<const Variable*, std::vector<const Function*>>;
  using F2Vs =
      std::unordered_map<const Function*, std::vector<const Variable*>>;

  explicit Graph(const std::shared_ptr<std::vector<Equation>>& equations)
      : functions_(equations),
        variable2next_functions_(std::make_shared<V2Fs>()),
        function2in_variables_(std::make_shared<F2Vs>()),
        function2out_variables_(std::make_shared<F2Vs>()) {
    for (const Function& function : *functions_) {
      CollectVariablesAndEdges(function);
    }
  }

  EquationGraphTopoWalker<const Variable*, const Function*> GetWalker() const {
    return EquationGraphTopoWalker<const Variable*, const Function*>(
        /*NextFunctionsVisitor=*/
        [=](const Variable* variable,
            const std::function<void(const Function*)>& Visit) {
          for (const Function* function :
               variable2next_functions_->at(variable)) {
            Visit(function);
          }
        },
        /*InputVariablesVisitor=*/
        [=](const Function* function,
            const std::function<void(const Variable*)>& Visit) {
          for (const Variable* variable :
               function2in_variables_->at(function)) {
            Visit(variable);
          }
        },
        /*OutputVariablesVisitor*/
        [=](const Function* function,
            const std::function<void(const Variable*)>& Visit) {
          for (const Variable* variable :
               function2out_variables_->at(function)) {
            Visit(variable);
          }
        });
  }

 private:
  void CollectVariablesAndEdges(const Function& function) {
    std::unordered_set<Variable> in_variables;
    std::unordered_set<Variable> out_variables;

    // clang-format off
    function >> match {
      [&](const Identity<tOut<Iterator>, tIn<Iterator>>& identity) {
        const auto& [out_iter, in_iter] = identity;
        out_variables.emplace(out_iter.value());
        in_variables.emplace(in_iter.value());
      },
      [&](const Identity<tOut<Index>, tIn<Index>>& identity) {
        const auto& [out_index, in_index] = identity;
        out_variables.emplace(out_index.value());
        in_variables.emplace(in_index.value());
      },
      [&](const Dot<List<Stride>, tOut<Index>, tIn<List<Iterator>>>& dot) {
        const auto& [strides, out_index, in_iterators] = dot;
        out_variables.emplace(out_index.value());
        in_variables.emplace(in_iterators.value().begin(),
                             in_iterators.value().end());
      },
      [&](const UnDot<List<Stride>, tOut<List<Iterator>>, tIn<Index>>& undot) {
        const auto& [strides, out_iterators, in_index] = undot;
        out_variables.emplace(out_iterators.value().begin(),
                              out_iterators.value().end());
        in_variables.emplace(in_index.value());
      },
      [&](const ConstructFakeOpPlaceHolder<tOut<FakeOpPlaceHolder>,
                                           tIn<List<Index>>>& construct_fake) {
        const auto& [out_fake_op_placeholder, in_indexes] =
            construct_fake;
        out_variables.emplace(out_fake_op_placeholder.value());
        in_variables.emplace(in_indexes.value().begin(),
                             in_indexes.value().end());
      }
    };
    // clang-format on

    for (const Variable& variable : in_variables) {
      const Variable* var_ptr = &*variables_.insert(variable).first;
      variable2next_functions_[var_ptr].push_back(&function);
      function2in_variables_[&function].push_back(var_ptr);
      v2f_edges_.emplace_back({var_ptr, &function});
    }
    for (const Variable& variable : out_variables) {
      const Variable* var_ptr = &*variables_.insert(variable).first;
      function2out_variables_[&function].push_back(var_ptr);
      f2v_edges_.emplace_back({&function, var_ptr});
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

  template <typename T0, typename T1>
  struct Edge final : public Tuple<T0, T1> {
    using Tuple<T0, T1>::Tuple;
  };

  // For debug
  std::vector<Edge<const Variable*, const Function*>> v2f_edges_;
  std::vector<Edge<const Function*, const Variable*>> f2v_edges_;
};

}  // namespace cinn::adt::equation
