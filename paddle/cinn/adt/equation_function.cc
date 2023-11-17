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

#include "paddle/cinn/adt/equation_function.h"

namespace cinn::adt {

std::pair<std::unordered_set<Variable>, std::unordered_set<Variable>>
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
          [&](const IndexDot<List<Dim>, tOut<Index>, tIn<List<Iterator>>>&
                  dot) {
            const auto& [dims, out_index, in_iterators] = dot.tuple();
            out_variables.emplace(Variable{out_index.value()});
            for (const auto& iterator : *in_iterators.value()) {
              in_variables.emplace(Variable{iterator});
            }
          },
          [&](const GetBroadcastedIterator<Dim, tOut<Iterator>, tIn<Iterator>>&
                  broadcast) {
            const auto& [dim, out_iterator, in_iterator] = broadcast.tuple();
            out_variables.emplace(Variable{out_iterator.value()});
            in_variables.emplace(Variable{in_iterator.value()});
          },
          [&](const IndexUnDot<List<Dim>, tOut<List<Iterator>>, tIn<Index>>&
                  undot) {
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

std::string GetFunctionTypeName(const Function& function) {
  return function >>
         match{
             [&](const Identity<tOut<Iterator>, tIn<Iterator>>& identity) {
               return "Identity";
             },
             [&](const Identity<tOut<Index>, tIn<Index>>& identity) {
               return "Identity";
             },
             [&](const IndexDot<List<Dim>, tOut<Index>, tIn<List<Iterator>>>&
                     dot) { return "IndexDot"; },
             [&](const GetBroadcastedIterator<Dim,
                                              tOut<Iterator>,
                                              tIn<Iterator>>& broadcast) {
               return "GetBroadcastedIterator";
             },
             [&](const IndexUnDot<List<Dim>, tOut<List<Iterator>>, tIn<Index>>&
                     undot) { return "IndexUnDot"; },
             [&](const InMsg2OutMsg<tOut<FakeOpPlaceHolder>,
                                    tOut<OpArgIndexes<std::optional<Index>>>,
                                    tIn<OpArgIndexes<Index>>>& in_msg2out_msg) {
               return "InMsg2OutMsg";
             },
             [&](const ConstantFunction<tOut<Iterator>, tIn<Index>>&
                     constant_function) { return "ConstantFunction"; },
         };
}

const void* GetFunctionDataPtr(const Function& function) {
  return function >>
         match{
             [&](const Identity<tOut<Iterator>, tIn<Iterator>>& identity)
                 -> const void* { return &identity.tuple(); },
             [&](const Identity<tOut<Index>, tIn<Index>>& identity)
                 -> const void* { return &identity.tuple(); },
             [&](const IndexDot<List<Dim>, tOut<Index>, tIn<List<Iterator>>>&
                     dot) -> const void* { return &dot.tuple(); },
             [&](const GetBroadcastedIterator<Dim,
                                              tOut<Iterator>,
                                              tIn<Iterator>>& broadcast)
                 -> const void* { return &broadcast.tuple(); },
             [&](const IndexUnDot<List<Dim>, tOut<List<Iterator>>, tIn<Index>>&
                     undot) -> const void* { return &undot.tuple(); },
             [&](const InMsg2OutMsg<tOut<FakeOpPlaceHolder>,
                                    tOut<OpArgIndexes<std::optional<Index>>>,
                                    tIn<OpArgIndexes<Index>>>& in_msg2out_msg)
                 -> const void* { return &in_msg2out_msg.tuple(); },
             [&](const ConstantFunction<tOut<Iterator>, tIn<Index>>&
                     constant_function) -> const void* {
               return &constant_function.tuple();
             },
         };
}

}  // namespace cinn::adt
