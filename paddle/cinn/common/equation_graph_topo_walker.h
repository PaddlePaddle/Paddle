// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include <array>
#include <functional>
#include <iostream>
#include <queue>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "paddle/cinn/common/bfs_walker.h"

namespace cinn {

template <typename VT, typename FT>
class EquationGraphTopoWalker final {
 public:
  using VariableVisitorT = std::function<void(VT)>;
  using FunctionVisitorT = std::function<void(FT)>;
  using F4VVisitor = std::function<void(VT, const FunctionVisitorT&)>;
  using V4FVisitor = std::function<void(FT, const VariableVisitorT&)>;

  EquationGraphTopoWalker(const F4VVisitor& NextFunctionsVisitor,
                          const V4FVisitor& InputVariablesVisitor,
                          const V4FVisitor& OutputVariablesVisitor)
      : VisitNextFunctions(NextFunctionsVisitor),
        VisitInputVariables(InputVariablesVisitor),
        VisitOutputVariables(OutputVariablesVisitor) {}
  ~EquationGraphTopoWalker() = default;

  static F4VVisitor Merge(const F4VVisitor& lhs, const F4VVisitor& rhs) {
    return [=](VT variable, const FunctionVisitorT& Visit) {
      lhs(variable, Visit);
      rhs(variable, Visit);
    };
  }

  static V4FVisitor Merge(const V4FVisitor& lhs, const V4FVisitor& rhs) {
    return [=](FT function, const VariableVisitorT& Visit) {
      lhs(function, Visit);
      rhs(function, Visit);
    };
  }

  EquationGraphTopoWalker Merge(const EquationGraphTopoWalker& that) const {
    return {Merge(this->VisitNextFunctions, that.VisitNextFunctions),
            Merge(this->VisitInputVariables, that.VisitInputVariables),
            Merge(this->VisitOutputVariables, that.VisitOutputVariables)};
  }

  void WalkVariable(VT start, const VariableVisitorT& VariableVisitor) const {
    std::array<VT, 1> starts{start};
    (*this)(starts.begin(), starts.end(), VariableVisitor, [&](FT) {});
  }

  template <typename VarIterT>
  void WalkVariable(VarIterT begin,
                    VarIterT end,
                    const VariableVisitorT& VariableVisitor) const {
    (*this)(begin, end, VariableVisitor, [&](FT) {});
  }

  void WalkFunction(VT start, const FunctionVisitorT& FunctionVisitor) const {
    std::array<VT, 1> starts{start};
    (*this)(
        starts.begin(), starts.end(), [&](VT) {}, FunctionVisitor);
  }

  template <typename VarIterT>
  void WalkFunction(VarIterT begin,
                    VarIterT end,
                    const FunctionVisitorT& FunctionVisitor) const {
    (*this)(
        begin, end, [&](VT) {}, FunctionVisitor);
  }

  void BfsWalkFunction(VT variable,
                       const FunctionVisitorT& FunctionVisitor) const {
    std::array<VT, 1> array{variable};
    BfsWalkFunction(array.begin(), array.end(), FunctionVisitor);
  }

  template <typename VarIterT>
  void BfsWalkFunction(VarIterT begin,
                       VarIterT end,
                       const FunctionVisitorT& FunctionVisitor) const {
    using F4FVisitor = std::function<void(FT, const FunctionVisitorT&)>;
    F4FVisitor BfsVisitNextFunction = [&](FT f,
                                          const FunctionVisitorT& DoEach) {
      VisitInputVariables(
          f, [&](VT variable) { VisitNextFunctions(variable, DoEach); });
      VisitOutputVariables(
          f, [&](VT variable) { VisitNextFunctions(variable, DoEach); });
    };
    std::vector<FT> starts{};
    for (VarIterT iter = begin; iter != end; ++iter) {
      VisitNextFunctions(*iter, [&](FT f) { starts.emplace_back(f); });
    }
    cinn::common::BfsWalker<FT> bfs_walker{BfsVisitNextFunction};
    bfs_walker(starts.begin(), starts.end(), FunctionVisitor);
  }

  template <typename VarIterT>
  void operator()(VarIterT begin,
                  VarIterT end,
                  const VariableVisitorT& VariableVisitor,
                  const FunctionVisitorT& FunctionVisitor) const {
    std::queue<VT> variables_queue{};
    std::unordered_set<VT> queued_variables{};
    std::queue<FT> functions_queue{};
    std::unordered_set<FT> queued_functions{};
    const auto& TryEnqueueVariable = [&](VT variable) {
      if (queued_variables.count(variable) == 0) {
        variables_queue.push(variable);
        queued_variables.insert(variable);
      }
    };
    const auto& TryEnqueueFunction = [&](FT function) {
      if (queued_functions.count(function) == 0) {
        functions_queue.push(function);
        queued_functions.insert(function);
      }
    };
    for (VarIterT iter = begin; iter != end; ++iter) {
      TryEnqueueVariable(*iter);
    }
    while (!functions_queue.empty() || !variables_queue.empty()) {
      if (!functions_queue.empty()) {
        FT function = functions_queue.front();
        functions_queue.pop();
        FunctionVisitor(function);
        VisitOutputVariables(function, TryEnqueueVariable);
      }
      if (!variables_queue.empty()) {
        VT variable = variables_queue.front();
        variables_queue.pop();
        VariableVisitor(variable);
        VisitNextFunctions(variable, [&](FT function) {
          size_t num_unfinished_inputs = 0;
          VisitInputVariables(function, [&](VT in_variable) {
            num_unfinished_inputs +=
                (queued_variables.count(in_variable) > 0 ? 0 : 1);
          });
          if (num_unfinished_inputs == 0) {
            TryEnqueueFunction(function);
          }
        });
      }
    }
  }

  // tNext [Function] <- Variable
  F4VVisitor VisitNextFunctions;
  // tIn [Variable] <- Function
  V4FVisitor VisitInputVariables;
  // tOut [Variable] <- Function
  V4FVisitor VisitOutputVariables;
};

}  // namespace cinn
