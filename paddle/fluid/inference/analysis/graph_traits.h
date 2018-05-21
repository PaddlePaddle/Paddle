/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

/*
 * This file defines the GraphTraits<X> template class that should be specified
 * by classes that want to be iteratable by generic graph iterators.
 *
 * This file also defines the marker class Inverse that is used to iterate over
 * graphs in a graph defined, inverse ordering...
 */

#pragma once

#include "paddle/fluid/inference/analysis/helper.h"

namespace paddle {
namespace inference {
namespace analysis {

/*
 * This class should be specialized by different graph types...
 * That's why the base class is empty.
 */
template <typename GraphType>
struct GraphTraits {
  // using NodesBFSIterator = xxx

  // NodesBFSIterator nodes_begin();
  // NodesBFSIterator nodes_end();
};

/*
 * Inverse - This class is used as a marker class to tell the graph iterator to
 * iterate in a graph defined Inverse order.
 */
template <typename GraphType>
struct Inverse {
  const GraphType &graph;

  explicit Inverse(const GraphType &graph) : graph(graph) {}
};

/*
 * Provide a partial specialization of GraphTraits so that the inverse of an
 * inverse turns into the original graph.
 */
template <typename GraphType>
struct GraphTraits<Inverse<Inverse<GraphType>>> : GraphTraits<GraphType> {};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
