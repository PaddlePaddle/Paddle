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
#include "AttributeMap.h"
#include "AttributeMeta.h"
namespace paddle {
namespace topology {
namespace meta {

enum DataType { DENSE = 0, SPARSE_INTEGER, SPARSE, INTEGER };
enum SequenceType { NO_SEQUENCE = 0, SEQUENCE, NESTED_SEQUENCE };

extern const Set<int> DefaultSequenceType;

const size_t kTensorShape_BATCH_SIZE = -1UL;
const size_t kTensorShape_NOT_SPECIFIC = -2UL;

/**
 * @brief A tensor meta save the meta information of a input/output of a layer
 * or a function.
 *
 * Tensor is a AttributeMetaMap, which contains many meta information of
 * Attribute.
 */
class TensorMeta : public AttributeMetaMap {
public:
  TensorMeta() : AttributeMetaMap("Tensor") {}

  /**
   * @brief setShapeDimension add constraints about the tensor shape dimension.
   * @param dims expected dimension
   * @param [out] constraints the raw Constraints object, used to add more
   * constraints.
   * @return *this
   */
  TensorMeta& setShapeDimension(
      size_t dims, Constraints<std::vector<size_t>>** constraints = nullptr);

  /**
   * @brief setShape add constrains about the tensor shape. If some elment of
   * this shape is dynamic and as same as batch size, set it to
   * kTensorShape_BATCH_SIZE.
   *
   * For example, if the tensor is a matrix, height = batch_size, width = 1, the
   * meta information of shape should be [kTensorShape_BATCH_SIZE, 1]
   *
   * @return *this
   */
  TensorMeta& setShape(const std::vector<size_t>& shape);

  /**
   * @brief Make tensor support sequence, default support all sequence.
   * @return *this
   */
  TensorMeta& supportSequenceTypes(
      const Set<int>& supportedTypes = DefaultSequenceType,
      Constraints<int>** constraints = nullptr);

  /**
   * @brief supportDataTypes set supported data type
   * @return *this
   */
  TensorMeta& supportDataTypes(const Set<int>& supportedTypes);

  /**
   * @brief Set default argument type. Used by paddle::Function.
   * @return *this;
   */
  TensorMeta& supportArgType(int defaultArgType,
                             const Set<int>& supportedTypes = {});

  /**
   * @brief if this tensor is optional for Function/Layer or not.
   */
  bool isOptional() const {
    return metaAttributes_.get<bool>("optional", /*default = */ false);
  }

  /**
   * @brief setOptional set this tensor is optional or not for Funciton/Layer
   * @return *this;
   */
  TensorMeta& setOptional(bool optional = true) {
    metaAttributes_.set<bool>("optional", optional, /*overwrite*/ true).check();
    return *this;
  }

  TensorMeta& setDescription(const std::string& doc) {
    metaAttributes_.set("description", doc).check();
    return *this;
  }

  const std::string& description() const {
    return metaAttributes_.get<std::string>("description");
  }

private:
  AttributeMap metaAttributes_;
};

typedef std::shared_ptr<TensorMeta> TensorMetaPtr;

}  // namespace meta
}  // namespace topology
}  // namespace paddle
