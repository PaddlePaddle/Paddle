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

const Set<int> DefaultSequenceType = {SEQUENCE, NO_SEQUENCE, NESTED_SEQUENCE};

TensorMeta &TensorMeta::setShapeDimension(
    size_t dims, Constraints<std::vector<int>> **constraints) {
  auto &cons = addAttribute<std::vector<int>>("shape", "The shape of tensor")
                   .mustSet()
                   .dimsEq(dims);
  if (constraints != nullptr) *constraints = &cons;
  return *this;
}

TensorMeta &TensorMeta::setShapeWithConstraints(const std::vector<int> &shape) {
  Constraints<std::vector<int>> *ptr;
  this->setShapeDimension(shape.size(), &ptr);
  ptr->addConstraint([shape](std::vector<int> *attr, bool) {
    for (size_t i = 0; i < shape.size(); ++i) {
      if (shape[i] != kTensorShape_BATCH_SIZE) {
        if (shape[i] != attr->at(i)) {
          return Error("Shape mismatch %d, expect %d, actual %d",
                       i,
                       shape[i],
                       attr->at(i));
        }
      }
    }
    return Error();
  });
  return *this;
}

TensorMeta &TensorMeta::supportSequenceTypes(const Set<int> &supportedTypes,
                                             Constraints<int> **constraints) {
  auto &cons =
      addAttribute<int>("sequence_type", "The sequence types of tensor")
          .mustSet()
          .in(supportedTypes);
  if (constraints != nullptr) {
    *constraints = &cons;
  }
  return *this;
}

TensorMeta &TensorMeta::supportDataTypes(const Set<int> &supportedTypes) {
  addAttribute<int>("data_type", "The data types of tensor")
      .mustSet()
      .in(supportedTypes);
  return *this;
}

TensorMeta &TensorMeta::supportArgType(int defaultArgType,
                                       const Set<int> &supportedTypes) {
  std::unordered_set<int> tmp;
  const Set<int> *ptr = &supportedTypes;
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
