// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include <algorithm>
#include <map>
#include <memory>
#include <vector>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/inference/anakin/engine.h"

#include "framework/core/net/net.h"
#include "framework/core/types.h"
#include "framework/graph/graph.h"
#include "framework/graph/graph_global_mem.h"
#include "saber/saber_types.h"

using anakin::saber::Shape;
using anakin::AK_FLOAT;
using anakin::AK_INT8;
using anakin::PBlock;

namespace paddle {
namespace inference {
namespace anakin {

std::unique_ptr<framework::LoDTensor> tensor_from_var(
    const framework::Variable& var, const platform::Place& place);

template <typename TargetT, ::anakin::Precision PrecisionT>
PBlock<TargetT>* pblock_from_tensor(const framework::LoDTensor& tensor,
                                    std::vector<int> shape_vec,
                                    AnakinEngine<TargetT, PrecisionT>* engine) {
  while (shape_vec.size() < 4) {
    shape_vec.insert(shape_vec.begin(), 1);
  }
  Shape shape(shape_vec);
  PBlock<TargetT>* weight = new PBlock<TargetT>(shape, AK_FLOAT);
  engine->RegistBlock(weight);
  float* cpu_data = static_cast<float*>(weight->h_tensor().mutable_data());
  std::copy_n(tensor.data<float>(), tensor.numel(), cpu_data);
  weight->d_tensor().set_shape(shape);
  weight->d_tensor().copy_from(weight->h_tensor());
  return weight;
}

template <typename TargetT, ::anakin::Precision PrecisionT>
PBlock<TargetT>* pblock_from_vector(const std::vector<float>& vec,
                                    std::vector<int> shape_vec,
                                    AnakinEngine<TargetT, PrecisionT>* engine) {
  while (shape_vec.size() < 4) {
    shape_vec.insert(shape_vec.begin(), 1);
  }
  Shape shape(shape_vec);
  PBlock<TargetT>* weight = new PBlock<TargetT>(shape, AK_FLOAT);
  engine->RegistBlock(weight);
  auto* weight_data = static_cast<float*>(weight->h_tensor().mutable_data());
  std::copy(std::begin(vec), std::end(vec), weight_data);
  weight->d_tensor().set_shape(shape);
  weight->d_tensor().copy_from(weight->h_tensor());
  return weight;
}

template <typename TargetT, ::anakin::Precision PrecisionT>
PBlock<TargetT>* pblock_from_vector(const std::vector<float>& vec,
                                    AnakinEngine<TargetT, PrecisionT>* engine) {
  int size = vec.size();
  return pblock_from_vector<TargetT, PrecisionT>(
      vec, std::vector<int>({1, 1, 1, size}), engine);
}

template <typename TargetT, ::anakin::Precision PrecisionT>
PBlock<TargetT>* pblock_from_var(const framework::Variable& var,
                                 AnakinEngine<TargetT, PrecisionT>* engine) {
  auto tensor = tensor_from_var(var, platform::CPUPlace());
  auto shape = framework::vectorize2int(tensor->dims());
  return pblock_from_tensor<TargetT, PrecisionT>(*tensor, shape, engine);
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle
