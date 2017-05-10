/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "TensorMeta.h"

namespace paddle {
namespace topology {
namespace meta {

TensorMeta &TensorMeta::addShape(size_t dims,
                                 Constraints<std::vector<int>> **constraints) {
  auto &cons = addAttribute<std::vector<int>>("shape", "The shape of tensor")
                   .mustSet()
                   .dimsEq(dims);
  if (constraints != nullptr) *constraints = &cons;
  return *this;
}

TensorMeta &TensorMeta::addSequenceType(
    const std::unordered_set<SequenceType, std::hash<int>> &supportedTypes,
    Constraints<SequenceType> **constraints) {
  auto &cons = addAttribute<SequenceType>("sequence_type",
                                          "The sequence types of tensor")
                   .mustSet()
                   .in(supportedTypes);
  if (constraints != nullptr) {
    *constraints = &cons;
  }
  return *this;
}

TensorMeta &TensorMeta::addDataType(
    const std::unordered_set<DataType, std::hash<int>> &supportedTypes) {
  addAttribute<DataType>("data_type", "The data types of tensor")
      .mustSet()
      .in(supportedTypes);
  return *this;
}

TensorMeta &TensorMeta::addArgType(
    int defaultArgType, const std::unordered_set<int> &supportedTypes) {
  std::unordered_set<int> tmp;
  const std::unordered_set<int> *ptr = &supportedTypes;
  if (supportedTypes.empty()) {
    tmp = {defaultArgType};
    ptr = &tmp;
  }
  addAttribute<int>("arg_type", "The argument type of tensor")
      .defaultValue(defaultArgType)
      .in(*ptr);
  return *this;
}

}  // namespace meta
}  // namespace topology
}  // namespace paddle
