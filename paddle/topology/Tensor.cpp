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
#include "Tensor.h"
namespace paddle {
namespace topology {

Tensor &Tensor::setDataType(int type) {
  attributes["data_type"] = type;
  return *this;
}

Tensor &Tensor::setSequenceType(int type) {
  attributes["sequence_type"] = type;
  return *this;
}

Tensor &Tensor::setShape(const std::vector<size_t> &shape) {
  attributes["shape"] = shape;
  return *this;
}

std::vector<size_t> &Tensor::shape() {
  return attributes.get<std::vector<size_t>>("shape");
}
SequenceType Tensor::sequenceType() const {
  return (SequenceType)attributes.get<int>("sequence_type");
}

DataType Tensor::dataType() const {
  return (DataType)attributes.get<int>("data_type");
}

}  // namespace topology
}  // namespace paddle
