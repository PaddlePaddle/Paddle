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
#include <paddle/utils/Any.h>
#include <functional>
#include <string>
#include <unordered_map>
#include "../Tensor.h"
#include "AttributeMap.h"
#include "AttributeMeta.h"
#include "TensorMeta.h"

namespace paddle {
namespace topology {
namespace meta {
namespace details {
enum FunctionTensorType { INPUT = 0, OUTPUT };
}  // namespace details

using details::FunctionTensorType;

class FunctionMeta;
typedef std::shared_ptr<FunctionMeta> FunctionMetaPtr;

/**
 * @brief Meta information for Function.
 *
 * Function is a concept mapping a Tensor to another Tensor. It represents the
 * computation logic in Paddle. It uses internally in Paddle to implement Layer.
 *
 * Function is a AttributeMetaMap, which contains many meta information of
 * Attributes. Function also contains meta information of inputs and outputs(See
 * TensorMeta for details).
 *
 * A FunctionMeta also contains some attribute itself, such as ShapeInferer
 * function. They stores in metaAttributes_, and can be read/write outside.
 */
class FunctionMeta : public AttributeMetaMap {
public:
  typedef std::vector<TensorMetaPtr> TensorMetas;
  typedef std::function<Error(std::vector<TensorPtr>&, std::vector<TensorPtr>&)>
      TenserShapeInferer;

private:
  std::string name_;
  TensorMetas inputs_;
  TensorMetas outputs_;

public:
  explicit FunctionMeta(const std::string& name)
      : AttributeMetaMap("function(" + name + ")'s"), name_(name) {}

  // register
  static FunctionMetaPtr registerFuncMeta(const std::string& name);

  // get registered meta
  static FunctionMetaPtr get(const std::string& name);

  /**
   * Add (input or output) tensor
   *
   * See TensorMeta for details
   */
  template <FunctionTensorType type>
  TensorMetaPtr& addTensor() {
    TensorMetas* arr = nullptr;
    if (type == FunctionTensorType::INPUT) {
      arr = &inputs_;
    } else {
      arr = &outputs_;
    }
    arr->emplace_back(new TensorMeta());
    return arr->back();
  }

  template <FunctionTensorType type>
  FunctionMeta& addTensor(size_t dim,
                          int argType = -1,
                          const Set<int>& dataTypes = {DataType::DENSE},
                          const Set<int>& seqTypes = DefaultSequenceType) {
    TensorMetaPtr& meta = addTensor<type>();
    meta->supportDataTypes(dataTypes)
        .supportSequenceTypes(seqTypes)
        .setShapeDimension(dim);
    if (argType != -1) meta->supportArgType(argType);
    return *this;
  }

  template <FunctionTensorType type>
  FunctionMeta& addTensor(const std::vector<size_t>& shape,
                          int argType = -1,
                          const Set<int>& dataTypes = {DataType::DENSE},
                          const Set<int>& seqTypes = DefaultSequenceType) {
    TensorMetaPtr& meta = addTensor<type>();
    meta->supportDataTypes(dataTypes).supportSequenceTypes(seqTypes).setShape(
        shape);
    if (argType != -1) meta->supportArgType(argType);
    return *this;
  }

  // Some alias for addTensor, make user interface simple.
  FunctionMeta& addInput(size_t dim,
                         const Set<int>& dataTypes = {DataType::DENSE},
                         const Set<int>& seqTypes = DefaultSequenceType) {
    return addTensor<FunctionTensorType::INPUT>(dim, -1, dataTypes, seqTypes);
  }

  FunctionMeta& addInput(const std::vector<size_t>& shape,
                         const Set<int>& dataTypes = {DataType::DENSE},
                         const Set<int>& seqTypes = DefaultSequenceType) {
    return addTensor<FunctionTensorType::INPUT>(shape, -1, dataTypes, seqTypes);
  }

  FunctionMeta& addOutput(size_t dim,
                          int argType,
                          const Set<int>& dataTypes = {DataType::DENSE},
                          const Set<int>& seqTypes = DefaultSequenceType) {
    return addTensor<FunctionTensorType::OUTPUT>(
        dim, argType, dataTypes, seqTypes);
  }

  FunctionMeta& addOutput(const std::vector<size_t>& shape,
                          int argType,
                          const Set<int>& dataTypes = {DataType::DENSE},
                          const Set<int>& seqTypes = DefaultSequenceType) {
    return addTensor<FunctionTensorType::OUTPUT>(
        shape, argType, dataTypes, seqTypes);
  }

  /**
   * @brief set shape inferer function. The shape inferer is a function, which
   * uses input tensor shapes as input, inference the output tensor should be
   * what shape.
   * @param inferer
   * @return *this
   */
  FunctionMeta& setShapeInferer(TenserShapeInferer inferer);

  TenserShapeInferer getShapeInferer() const;

  const std::vector<TensorMetaPtr>& inputs() const { return inputs_; }
  const std::vector<TensorMetaPtr>& outputs() const { return outputs_; }

  AttributeMap metaAttributes_;
};

}  // namespace meta
}  // namespace topology
}  // namespace paddle
