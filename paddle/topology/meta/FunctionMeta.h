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
  std::string name;
  static std::unordered_map<std::string, FunctionMetaPtr> gFuncMetas;
  std::unordered_map<std::string, any> metaInfos;

  TensorMetas inputs_;
  TensorMetas outputs_;

public:
  explicit FunctionMeta(const std::string& name)
      : WithAttributeMeta("function(" + name + ")'s"), name(name) {}

  template <typename T>
  paddle::Error __must_check addMeta(const std::string& name, const T& val) {
    if (metaInfos.find(name) != metaInfos.end()) {
      return paddle::Error("Duplicated meta infos %s", name.c_str());
    }
    metaInfos[name] = val;
    return paddle::Error();
  }

  template <typename T>
  paddle::Error __must_check getMeta(const std::string& name,
                                     const T** val) const {
    auto it = metaInfos.find(name);
    if (it == metaInfos.end()) {
      return paddle::Error("Cannot find meta %s", name.c_str());
    }
    const any* ptr = &it->second;
    *val = any_cast<T>(ptr);
    if (val == nullptr) {
      return paddle::Error("Cannot cast to type %s", typeid(T).name());
    }
    return paddle::Error();
  }

  static paddle::Error __must_check
  registerFuncMeta(const std::string& name,
                   std::function<paddle::Error(FunctionMetaPtr&)> func);

  static FunctionMetaPtr get(const std::string& name) {
    auto it = gFuncMetas.find(name);
    if (it != gFuncMetas.end()) {
      return it->second;
    } else {
      return nullptr;
    }
  }

  TensorMetaPtr& addInput() {
    inputs_.emplace_back(new TensorMeta());
    return inputs_.back();
  }

  TensorMetaPtr& addOutput() {
    outputs_.emplace_back(new TensorMeta());
    return outputs_.back();
  }

  void setShapeInferer(TenserShapeInferer inferer) {
    addMeta("shapeInferer", inferer).check();
  }

  const TenserShapeInferer& getShapeInferer() const {
    const TenserShapeInferer* func;
    getMeta("shapeInferer", &func).check();
    return *func;
  }

  const std::vector<TensorMetaPtr>& inputs() const { return inputs_; }
  const std::vector<TensorMetaPtr>& outputs() const { return outputs_; }
};

}  // namespace meta
}  // namespace topology
}  // namespace paddle
