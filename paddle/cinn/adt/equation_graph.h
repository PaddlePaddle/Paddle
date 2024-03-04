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

#include <memory>
#include <unordered_set>
#include <vector>

#include "paddle/cinn/adt/equation.h"
#include "paddle/cinn/common/equation_graph_topo_walker.h"

namespace cinn::adt {

template <typename VariableT, typename FunctionT>
struct GraphTrait;

template <>
struct GraphTrait<Variable, Function> {
  static std::pair<std::unordered_set<Variable>, std::unordered_set<Variable>>
  CollectInputAndOutputVariables(const Function& function) {
    std::unordered_set<Variable> in_variables;
    std::unordered_set<Variable> out_variables;

    function >>
        match{
            [&](const Identity<tOut<Iterator>, tIn<Iterator>>& identity) {
              const auto& [out_iter, in_iter] = identity.tuple();
              out_variables.emplace(Variable{out_iter.value()});
              in_variables.emplace(Variable{in_iter.value()});
            },
            [&](const Identity<tOut<Index>, tIn<Index>>& identity) {
              const auto& [out_index, in_index] = identity.tuple();
              out_variables.emplace(Variable{out_index.value()});
              in_variables.emplace(Variable{in_index.value()});
            },
            [&](const IndexDot<List<DimExpr>, tOut<Index>, tIn<List<Iterator>>>&
                    dot) {
              const auto& [dims, out_index, in_iterators] = dot.tuple();
              out_variables.emplace(Variable{out_index.value()});
              for (const auto& iterator : *in_iterators.value()) {
                in_variables.emplace(Variable{iterator});
              }
            },
            [&](const GetBroadcastedIterator<DimExpr,
                                             tOut<Iterator>,
                                             tIn<Iterator>>& broadcast) {
              const auto& [dim, out_iterator, in_iterator] = broadcast.tuple();
              out_variables.emplace(Variable{out_iterator.value()});
              in_variables.emplace(Variable{in_iterator.value()});
            },
            [&](const IndexUnDot<List<DimExpr>,
                                 tOut<List<Iterator>>,
                                 tIn<Index>>& undot) {
              const auto& [dims, out_iterators, in_index] = undot.tuple();
              for (const auto& iterator : *out_iterators.value()) {
                out_variables.emplace(Variable{iterator});
              }
              in_variables.emplace(Variable{in_index.value()});
            },
            [&](const InMsg2OutMsg<tOut<FakeOpPlaceHolder>,
                                   tOut<OpArgIndexes<std::optional<Index>>>,
                                   tIn<OpArgIndexes<Index>>>& in_msg2out_msg) {
              const auto& [op_placeholder, out_msg_indexes, in_msg_indexes] =
                  in_msg2out_msg.tuple();
              out_variables.emplace(Variable{op_placeholder.value()});
              const auto& [out_msg_in_indexes, out_msg_out_indexes] =
                  out_msg_indexes.value().tuple();
              const auto& [in_msg_in_indexes, in_msg_out_indexes] =
                  in_msg_indexes.value().tuple();
              for (const auto& index : *out_msg_in_indexes.value()) {
                out_variables.emplace(Variable{index});
              }
              for (const auto& index : *out_msg_out_indexes.value()) {
                if (index.has_value()) {
                  out_variables.emplace(Variable{index.value()});
                }
              }
              for (const auto& index : *in_msg_in_indexes.value()) {
                in_variables.emplace(Variable{index});
              }
              for (const auto& index : *in_msg_out_indexes.value()) {
                in_variables.emplace(Variable{index});
              }
            },
            [&](const ConstantFunction<tOut<Iterator>, tIn<Index>>&
                    constant_function) {
              const auto& [out_iterator, in_index, constant] =
                  constant_function.tuple();
              out_variables.emplace(Variable{out_iterator.value()});
              in_variables.emplace(Variable{in_index.value()});
            },
        };
    return std::make_pair(in_variables, out_variables);
  }
};

// clang-format off
/*
Graph = ([Variable], [Function], [Edge Variable Function], [Edge Function Variable])
Edge T0 T1 = (T0, T1)
*/
// clang-format on
template <typename VariableT, typename FunctionT>
class Graph final
    : public std::enable_shared_from_this<Graph<VariableT, FunctionT>> {
 public:
  using V2Fs = std::unordered_map<VariableT, std::vector<const FunctionT*>>;
  using F2Vs = std::unordered_map<const FunctionT*, std::vector<VariableT>>;
  using Functions = List<FunctionT>;

  static std::shared_ptr<Graph<VariableT, FunctionT>> New(
      const Functions& equations) {
    return std::shared_ptr<Graph<VariableT, FunctionT>>{
        new Graph<VariableT, FunctionT>{equations}};
  }

  using VariableVisitorT = std::function<void(const VariableT)>;
  using FunctionVisitorT = std::function<void(const FunctionT*)>;
  using F4VVisitor =
      std::function<void(const VariableT, const FunctionVisitorT&)>;
  using V4FVisitor =
      std::function<void(const FunctionT*, const VariableVisitorT&)>;

  static F4VVisitor Merge(const F4VVisitor& lhs, const F4VVisitor& rhs) {
    return [=](const VariableT variable,
               const std::function<void(const FunctionT*)>& Visit) {
      lhs(variable, Visit);
      rhs(variable, Visit);
    };
  }

  static V4FVisitor Merge(const V4FVisitor& lhs, const V4FVisitor& rhs) {
    return [=](const FunctionT* function,
               const std::function<void(const VariableT)>& Visit) {
      lhs(function, Visit);
      rhs(function, Visit);
    };
  }

  F4VVisitor GetNextFunctionsVisitor() const {
    auto self = this->shared_from_this();
    return [self](const VariableT variable,
                  const std::function<void(const FunctionT*)>& Visit) {
      const auto iter = self->variable2next_functions_->find(variable);
      if (iter == self->variable2next_functions_->end()) {
        return;
      }
      for (const FunctionT* function : iter->second) {
        Visit(function);
      }
    };
  }

  V4FVisitor GetInputVariablesVisitor() const {
    auto self = this->shared_from_this();
    return [self](const FunctionT* function,
                  const std::function<void(const VariableT)>& Visit) {
      const auto iter = self->function2in_variables_->find(function);
      if (iter == self->function2in_variables_->end()) {
        return;
      }
      for (const VariableT variable : iter->second) {
        Visit(variable);
      }
    };
  }

  V4FVisitor GetOutputVariablesVisitor() const {
    auto self = this->shared_from_this();
    return [self](const FunctionT* function,
                  const std::function<void(const VariableT)>& Visit) {
      const auto iter = self->function2out_variables_->find(function);
      if (iter == self->function2out_variables_->end()) {
        return;
      }
      for (const VariableT variable : iter->second) {
        Visit(variable);
      }
    };
  }

  static EquationGraphTopoWalker<VariableT, const FunctionT*> GetMergedWalker(
      const Graph<VariableT, FunctionT>& lhs,
      const Graph<VariableT, FunctionT>& rhs) {
    return EquationGraphTopoWalker<VariableT, const FunctionT*>(
        /*NextFunctionsVisitor=*/Merge(lhs.GetNextFunctionsVisitor(),
                                       rhs.GetNextFunctionsVisitor()),
        /*InputVariablesVisitor=*/
        Merge(lhs.GetInputVariablesVisitor(), rhs.GetInputVariablesVisitor()),
        /*OutputVariablesVisitor=*/
        Merge(lhs.GetOutputVariablesVisitor(),
              rhs.GetOutputVariablesVisitor()));
  }

  EquationGraphTopoWalker<VariableT, const FunctionT*> GetGraphView() const {
    return EquationGraphTopoWalker<VariableT, const FunctionT*>(
        /*NextFunctionsVisitor=*/GetNextFunctionsVisitor(),
        /*InputVariablesVisitor=*/GetInputVariablesVisitor(),
        /*OutputVariablesVisitor=*/GetOutputVariablesVisitor());
  }

  const std::unordered_set<VariableT>& GetVariables() const {
    return variables_;
  }

  const Functions& GetEquations() const { return functions_; }

 private:
  explicit Graph(const Functions& equations)
      : functions_(equations),
        variable2next_functions_(std::make_shared<V2Fs>()),
        function2in_variables_(std::make_shared<F2Vs>()),
        function2out_variables_(std::make_shared<F2Vs>()) {
    for (const FunctionT& function : *functions_) {
      CollectVariablesAndEdges(function);
    }
  }

  void CollectVariablesAndEdges(const FunctionT& function) {
    const auto& [in_variables, out_variables] =
        GraphTrait<VariableT, FunctionT>::CollectInputAndOutputVariables(
            function);

    for (const VariableT& variable : in_variables) {
      variables_.insert(variable);
      (*variable2next_functions_)[variable].push_back(&function);
      (*function2in_variables_)[&function].push_back(variable);
      v2f_edges_.emplace_back(std::pair{variable, &function});
    }
    for (const VariableT& variable : out_variables) {
      variables_.insert(variable);
      (*function2out_variables_)[&function].push_back(variable);
      f2v_edges_.emplace_back(std::pair{&function, variable});
    }
  }

  Functions functions_;
  // tNext [Function] <- Variable
  std::shared_ptr<V2Fs> variable2next_functions_;
  // tIn [Variable] <- Function
  std::shared_ptr<F2Vs> function2in_variables_;
  // tOut [Variable] <- Function
  std::shared_ptr<F2Vs> function2out_variables_;
  std::unordered_set<VariableT> variables_;

  // For debug
  std::vector<std::pair<const VariableT, const FunctionT*>> v2f_edges_;
  std::vector<std::pair<const FunctionT*, const VariableT>> f2v_edges_;
};

}  // namespace cinn::adt
