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

std::string GetFunctionTypeName(const Function& function) {
  return function >>
         match{
             [&](const Identity<tOut<Iterator>, tIn<Iterator>>& identity) {
               return "Identity";
             },
             [&](const Identity<tOut<Index>, tIn<Index>>& identity) {
               return "Identity";
             },
             [&](const IndexDot<List<DimExpr>,
                                tOut<Index>,
                                tIn<List<Iterator>>>& dot) {
               return "IndexDot";
             },
             [&](const GetBroadcastedIterator<DimExpr,
                                              tOut<Iterator>,
                                              tIn<Iterator>>& broadcast) {
               return "GetBroadcastedIterator";
             },
             [&](const IndexUnDot<List<DimExpr>,
                                  tOut<List<Iterator>>,
                                  tIn<Index>>& undot) { return "IndexUnDot"; },
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
             [&](const IndexDot<List<DimExpr>,
                                tOut<Index>,
                                tIn<List<Iterator>>>& dot) -> const void* {
               return &dot.tuple();
             },
             [&](const GetBroadcastedIterator<DimExpr,
                                              tOut<Iterator>,
                                              tIn<Iterator>>& broadcast)
                 -> const void* { return &broadcast.tuple(); },
             [&](const IndexUnDot<List<DimExpr>,
                                  tOut<List<Iterator>>,
                                  tIn<Index>>& undot) -> const void* {
               return &undot.tuple();
             },
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
