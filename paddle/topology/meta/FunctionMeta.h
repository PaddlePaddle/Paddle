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

class FunctionMeta : public WithAttributeMeta {
public:
  typedef std::vector<TensorMetaPtr> TensorMetas;
  typedef std::function<Error(std::vector<TensorPtr>&, std::vector<TensorPtr>&)>
      TenserShapeInferer;

private:
  std::string name;
  TensorMetas inputs_;
  TensorMetas outputs_;

public:
  explicit FunctionMeta(const std::string& name)
      : WithAttributeMeta("function(" + name + ")'s"), name(name) {}

  static FunctionMetaPtr registerFuncMeta(const std::string& name);

  static FunctionMetaPtr get(const std::string& name);

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
  FunctionMeta& addTensor(const std::vector<int>& shape,
                          int argType = -1,
                          const Set<int>& dataTypes = {DataType::DENSE},
                          const Set<int>& seqTypes = DefaultSequenceType) {
    TensorMetaPtr& meta = addTensor<type>();
    meta->supportDataTypes(dataTypes)
        .supportSequenceTypes(seqTypes)
        .setShapeWithConstraints(shape);
    if (argType != -1) meta->supportArgType(argType);
    return *this;
  }

  FunctionMeta& setShapeInferer(TenserShapeInferer inferer);

  TenserShapeInferer getShapeInferer() const;

  const std::vector<TensorMetaPtr>& inputs() const { return inputs_; }
  const std::vector<TensorMetaPtr>& outputs() const { return outputs_; }

  AttributeMap metaAttributes_;
};

}  // namespace meta
}  // namespace topology
}  // namespace paddle
