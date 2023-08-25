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
#include <iostream>
#include <queue>
#include <tuple>
#include <unordered_set>

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
      : : NextFunctionsVisitor_(NextFunctionsVisitor),
          InputVariablesVisitor_(InputVariablesVisitor),
          OutputVariablesVisitor_(OutputVariablesVisitor) {}
  ~EquationGraphTopoWalker() = default;

  void operator()(VT start, const VariableVisitorT& VariableVisitor) const {
    std::array<VT, 1> starts{start};
    (*this)(starts.begin(), starts.end(), VariableVisitor, [&](FT) {});
  }

  template <typename VarIterT>
  void operator()(VarIterT begin,
                  VarIterT end,
                  const VariableVisitorT& VariableVisitor) const {
    (*this)(begin, end, VariableVisitor, [&](FT) {});
  }

  void operator()(VT start, const FunctionVisitorT& FunctionVisitor) const {
    std::array<VT, 1> starts{start};
    (*this)(
        starts.begin(), starts.end(), [&](VT) {}, FunctionVisitor);
  }

  template <typename VarIterT>
  void operator()(VarIterT begin,
                  VarIterT end,
                  const FunctionVisitorT& FunctionVisitor) const {
    (*this)(
        begin, end, [&](VT) {}, FunctionVisitor);
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
    const TryEnqueueVaraible = [&](VT variable) {
      if (queued_variables.count(variable) == 0) {
        variables_queue.push_back(variable);
        queued_variables.insert(variable);
      }
    };
    const TryEnqueueFunction = [&](FT function) {
      if (queued_functions.count(function) == 0) {
        functions_queue.push_back(function);
        queued_functions.insert(function);
      }
    };
    for (VarIterT iter = begin; iter != end; ++iter) {
      TryEnqueueVaraible(*iter);
    }
    while (!functions_queue.empty() || !variables_queue.empty()) {
      if (!functions_queue.empty()) {
        FT function = functions_queue.front();
        functions_queue.pop();
        FunctionVisitor(function);
        OutputVariablesVisitor_(function, TryEnqueueVaraible);
      }
      if (!variables_queue.empty()) {
        VT variable = variables_queue.front();
        variables_queue.pop();
        VariableVisitor(variable);
        NextFunctionsVisitor_(variable, [&](FT function) {
          size_t num_unfinished_inputs = 0;
          InputVariablesVisitor_(function, [&](VT in_variable) {
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

 private:
  // tNext [Function] <- Variable
  F4VVisitor NextFunctionsVisitor_;
  // tIn [Variable] <- Function
  V4FVisitor InputVariablesVisitor_;
  // tOut [Variable] <- Function
  V4FVisitor OutputVariablesVisitor_;
};

}  // namespace cinn
