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
#include "glog/logging.h"

#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/hlir/framework/node.h"

namespace cinn::adt::adapter {

struct Tensor final {
  const hlir::framework::NodeData* node_data;
  const hlir::framework::Graph* graph;

  bool operator==(const Tensor& other) const {
    return this->node_data == other.node_data && this->graph == other.graph;
  }

  std::size_t GetRank() const {
    const auto& shape_dict =
        graph->GetAttrs<absl::flat_hash_map<std::string, utils::ShapeType>>(
            "infershape");
    CHECK(shape_dict.count(node_data->id()))
        << "Can't find " << node_data->id() << " 's shape!";
    return shape_dict.at(node_data->id()).size();
  }

  const std::vector<int32_t>& GetShape() const {
    const auto& shape_dict =
        graph->GetAttrs<absl::flat_hash_map<std::string, utils::ShapeType>>(
            "infershape");
    CHECK(shape_dict.count(node_data->id()))
        << "Can't find " << node_data->id() << " 's shape!";
    return shape_dict.at(node_data->id());
  }

  std::size_t GetNumel() const {
    const auto& shape_dict =
        graph->GetAttrs<absl::flat_hash_map<std::string, utils::ShapeType>>(
            "infershape");
    CHECK(shape_dict.count(node_data->id()))
        << "Can't find " << node_data->id() << " 's shape!";
    std::vector<int32_t> shape = shape_dict.at(node_data->id());
    std::size_t ret = 1;
    for (int32_t dim_size : shape) {
      ret = ret * dim_size;
    }
    return ret;
  }
};

inline std::size_t GetHashValueImpl(const Tensor& tensor) {
  return hash_combine(
      std::hash<const hlir::framework::NodeData*>()(tensor.node_data),
      std::hash<const hlir::framework::Graph*>()(tensor.graph));
}

}  // namespace cinn::adt::adapter
