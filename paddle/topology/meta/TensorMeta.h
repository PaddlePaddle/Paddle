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

#pragma once
#include <unordered_set>
#include "AttributeMeta.h"
namespace paddle {
namespace topology {
namespace meta {

enum DataType { DENSE = 0, SPARSE_INTEGER, SPARSE, INTEGER };
enum SequenceType { NO_SEQUENCE = 0, SEQUENCE, NESTED_SEQUENCE };

const int kTensorShape_BATCH_SIZE = -1;

class TensorMeta : public WithAttributeMeta {
public:
  TensorMeta() : WithAttributeMeta("Tensor") {}

  TensorMeta& setShapeDimension(
      size_t dims, Constraints<std::vector<int>>** constraints = nullptr);

  TensorMeta& setShapeWithConstraints(const std::vector<int>& shape);

  TensorMeta& supportSequenceTypes(
      const std::unordered_set<SequenceType, std::hash<int>>& supportedTypes =
          {NO_SEQUENCE, SEQUENCE, NESTED_SEQUENCE},
      Constraints<SequenceType>** constraints = nullptr);

  TensorMeta& supportDataTypes(
      const std::unordered_set<DataType, std::hash<int>>& supportedTypes);

  TensorMeta& supportArgType(
      int defaultArgType, const std::unordered_set<int>& supportedTypes = {});
};

typedef std::shared_ptr<TensorMeta> TensorMetaPtr;

}  // namespace meta
}  // namespace topology
}  // namespace paddle
