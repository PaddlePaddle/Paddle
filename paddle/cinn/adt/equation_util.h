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
#include <optional>
#include <unordered_map>
#include <unordered_set>

#include "paddle/cinn/adt/equation.h"
#include "paddle/cinn/common/equation_graph_topo_walker.h"
#include "paddle/cinn/common/topo_walker.h"

namespace cinn::adt {

template <typename VT, typename FT>
common::TopoWalker<FT> GetDAGTopoWalker(
    const EquationGraphTopoWalker<VT, FT>& eg_walker, const VT& start) {
  auto var2solo_producer =
      std::make_shared<std::unordered_map<VT, std::optional<FT>>>();
  var2solo_producer->emplace_back(start, std::nullopt);
  eg_walker(start, [&](FT function) {
    eg_walker.VisitOutputVariables(function, [&](VT out_variable) {
      var2solo_producer->emplace_back(out_variable, function);
    });
  });
  const auto& VisitPrevNodes = [var2solo_producer, eg_walker](
                                   FT function,
                                   const std::function<void(FT)>& Visit) {
    eg_walker.VisitInputVariables(function, [&](VT in_variable) {
      const auto& opt_producer = var2solo_producer->at(in_variable);
      if (opt_producer.has_value()) {
        Visit(opt_producer.value());
      }
    });
  };
  const auto& VisitNextNodes = [var2solo_producer, eg_walker](
                                   FT function,
                                   const std::function<void(FT)>& Visit) {
    eg_walker.VisitOutputVariables(function, [&](VT out_variable) {
      const auto& opt_producer = var2solo_producer->at(out_variable);
      if (opt_producer.has_value() && opt_producer.value() == function) {
        eg_walker.VisitNextFunctions(out_variable, Visit);
      } else {
        // Do nothing
      }
    });
  };
  return common::TopoWalker<FT>(VisitPrevNodes, VisitNextNodes);
}

template <typename VT, typename FT>
EquationGraphTopoWalker<VT, FT> GetSubgraph(
    const EquationGraphTopoWalker<VT, FT>& graph,
    const std::function<bool(FT)>& IsSelected) {
  const auto& VisitNextFunctions =
      [graph, IsSelected](VT variable, const std::function<void(FT)>& Visit) {
        graph.VisitNextFunctions(variable, [&](FT out_function) {
          if (IsSelected(out_function)) {
            Visit(out_function);
          }
        });
      };
  const auto& VisitInputVariables =
      [graph, IsSelected](FT function, const std::function<void(VT)>& Visit) {
        CHECK(IsSelected(function));
        graph.VisitInputVariables(function, Visit);
      };
  const auto& VisitOutputVariables =
      [graph, IsSelected](FT function, const std::function<void(VT)>& Visit) {
        CHECK(IsSelected(function));
        graph.VisitOutputVariables(function, Visit);
      };
  return EquationGraphTopoWalker<VT, FT>(
      VisitNextFunctions, VisitInputVariables, VisitOutputVariables);
}

inline List<Stride> MakeStrides(std::size_t num_strides) {
  List<Stride> ret{};
  for (std::size_t i = 0; i < num_strides; ++i) {
    ret->emplace_back(UniqueId::New());
  }
  return ret;
}

template <typename DoEachT>
void IdentityConnect(const Index& out, const Index& in, const DoEachT& DoEach) {
  DoEach(Identity<tOut<Index>, tIn<Index>>{out, in});
}

inline void IdentityConnect(const Index& out,
                            const Index& in,
                            Equations* equations) {
  IdentityConnect(out, in, [&](const auto& equation) {
    (*equations)->push_back(equation);
  });
}

template <typename DoEachT>
void Equal(const Index& lhs, const Index& rhs, const DoEachT& DoEach) {
  IdentityConnect(lhs, rhs, DoEach);
  IdentityConnect(rhs, lhs, DoEach);
}

inline void Equal(const Index& lhs, const Index& rhs, Equations* equations) {
  Equal(lhs, rhs, [&](const auto& equation) {
    (*equations)->emplace_back(equation);
  });
}

template <typename DoEachT>
void GenerateDotEquation(const List<Iterator>& iterators,
                         const List<Stride>& strides,
                         const Index& index,
                         const DoEachT& DoEach) {
  DoEach(Dot<List<Stride>, tOut<Index>, tIn<List<Iterator>>>{
      strides, index, iterators});
  DoEach(UnDot<List<Stride>, tOut<List<Iterator>>, tIn<Index>>{
      strides, iterators, index});
}

template <typename DoEachT>
Index MakeDot(const List<Iterator>& iterators,
              const List<Stride>& strides,
              const DoEachT& DoEach) {
  Index ret{UniqueId::New()};
  GenerateDotEquation(iterators, strides, ret, DoEach);
  return ret;
}

inline Index MakeDot(const List<Iterator>& iterators,
                     const List<Stride>& strides,
                     Equations* equations) {
  return MakeDot(iterators, strides, [&](const auto& equation) {
    (*equations)->emplace_back(equation);
  });
}

inline List<Iterator> MakeIterators(std::size_t num_iterators) {
  List<Iterator> ret{};
  for (std::size_t i = 0; i < num_iterators; ++i) {
    ret->emplace_back(UniqueId::New());
  }
  return ret;
}

template <typename DoEachT>
List<Iterator> MakeUnDot(const Index& index,
                         const List<Stride>& strides,
                         const DoEachT& DoEach) {
  List<Iterator> ret{};
  GenerateDotEquation(ret, strides, index, DoEach);
  return ret;
}

inline List<Iterator> MakeUnDot(const Index& index,
                                const List<Stride>& strides,
                                Equations* equations) {
  return MakeUnDot(index, strides, [&](const auto& equation) {
    (*equations)->emplace_back(equation);
  });
}

}  // namespace cinn::adt
