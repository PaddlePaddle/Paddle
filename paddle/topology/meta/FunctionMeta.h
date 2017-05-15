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
#include <string>
#include <unordered_map>
#include "../Tensor.h"
#include "AttributeMap.h"
#include "AttributeMeta.h"
#include "TensorMeta.h"

namespace paddle {
namespace topology {
namespace meta {

class FunctionMeta;
typedef std::shared_ptr<FunctionMeta> FunctionMetaPtr;

class FunctionMeta : public WithAttributeMeta {
public:
  typedef std::vector<TensorMetaPtr> TensorMetas;
  typedef std::function<Error(std::vector<TensorPtr>&, std::vector<TensorPtr>&)>
      TenserShapeInferer;

private:
  const static Set<int> defaultSeqTypes;

private:
  std::string name;
  TensorMetas inputs_;
  TensorMetas outputs_;

public:
  explicit FunctionMeta(const std::string& name)
      : WithAttributeMeta("function(" + name + ")'s"), name(name) {}

  static paddle::Error __must_check
  registerFuncMeta(const std::string& name,
                   std::function<paddle::Error(FunctionMetaPtr&)> func);

  static FunctionMetaPtr get(const std::string& name);

  TensorMetaPtr& addInput() {
    inputs_.emplace_back(new TensorMeta());
    return inputs_.back();
  }

  FunctionMeta& addInput(size_t dim,
                         const Set<int>& dataTypes = {DataType::DENSE},
                         const Set<int>& seqTypes = defaultSeqTypes) {
    addInput()
        ->supportDataTypes(dataTypes)
        .supportSequenceTypes(seqTypes)
        .setShapeDimension(dim);
    return *this;
  }

  FunctionMeta& addInput(const std::vector<int>& shape,
                         const Set<int>& dataTypes = {DataType::DENSE},
                         const Set<int>& seqTypes = defaultSeqTypes) {
    addInput()
        ->supportDataTypes(dataTypes)
        .supportSequenceTypes(seqTypes)
        .setShapeWithConstraints(shape);
    return *this;
  }

  TensorMetaPtr& addOutput() {
    outputs_.emplace_back(new TensorMeta());
    return outputs_.back();
  }

  FunctionMeta& addOutput(int argType,
                          size_t dim,
                          const Set<int>& dataTypes = {DataType::DENSE},
                          const Set<int>& seqTypes = defaultSeqTypes) {
    addOutput()
        ->supportDataTypes(dataTypes)
        .supportSequenceTypes(seqTypes)
        .setShapeDimension(dim)
        .supportArgType(argType);
    return *this;
  }

  FunctionMeta& addOutput(int argType,
                          const std::vector<int>& shape,
                          const Set<int>& dataTypes = {DataType::DENSE},
                          const Set<int>& seqTypes = defaultSeqTypes) {
    addOutput()
        ->supportDataTypes(dataTypes)
        .supportSequenceTypes(seqTypes)
        .setShapeWithConstraints(shape)
        .supportArgType(argType);
    return *this;
  }

  void setShapeInferer(TenserShapeInferer inferer) {
    metaAttributes_.set("shapeInferer", inferer).check();
  }

  TenserShapeInferer getShapeInferer() const {
    const TenserShapeInferer* func;
    metaAttributes_.get("shapeInferer", &func).check();
    return *func;
  }

  const std::vector<TensorMetaPtr>& inputs() const { return inputs_; }
  const std::vector<TensorMetaPtr>& outputs() const { return outputs_; }

  template <typename T>
  Error regAttributeParser(
      const std::function<Error(const AttributeMap&, T*)>& callback) {
    return this->metaAttributes_.set("attribute_parser", callback, false);
  }

  template <typename T>
  Error parseAttribute(const AttributeMap& attrs, T* attr) const {
    const std::function<Error(const AttributeMap&, T*)>* callback;
    auto err = this->metaAttributes_.get("attribute_parser", &callback);
    if (!err.isOK()) {
      return err;
    }
    return (*callback)(attrs, attr);
  }

  AttributeMap metaAttributes_;
};

}  // namespace meta
}  // namespace topology
}  // namespace paddle
