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
#include "AttributeMeta.h"
namespace paddle {
namespace topology {
namespace meta {

class FunctionMeta;
typedef std::shared_ptr<FunctionMeta> FunctionMetaPtr;

class FunctionMeta {
private:
  std::string name;
  std::unordered_map<std::string, AttributeMetaPtr> attributes;
  static std::unordered_map<std::string, FunctionMetaPtr> gFuncMetas;
  std::unordered_map<std::string, any> metaInfos;

public:
  FunctionMeta(const std::string& name) : name(name) {}

  const std::unordered_map<std::string, AttributeMetaPtr>& getAttributes()
      const {
    return attributes;
  }

  paddle::Error __must_check
  addAttribute(const AttributeMetaPtr& attributeMeta) {
    if (attributeMeta == nullptr) {
      return paddle::Error("NULL Pointer Error");
    }
    auto attrName = attributeMeta->name;
    if (this->attributes.find(attrName) != this->attributes.end()) {
      return paddle::Error("function(%s)'s attribute %s has been setted",
                           this->name.c_str(),
                           attrName.c_str());
    }
    this->attributes[attrName] = attributeMeta;
    return paddle::Error();
  }

  template <typename T>
  Constraints<T>& addAttribute(const std::string& name,
                               const std::string& description) {
    auto metaPtr = AttributeMeta::create<T>(name, description);
    auto err = addAttribute(metaPtr);
    err.check();
    return *(metaPtr->template constraintBuilder<T>());
  }

  template <typename T>
  paddle::Error __must_check addMeta(const std::string& name, const T& val) {
    if (metaInfos.find(name) != metaInfos.end()) {
      return paddle::Error("Duplicated meta infos %s", name.c_str());
    }
    metaInfos[name] = val;
    return paddle::Error();
  }

  template <typename T>
  paddle::Error __must_check getMeta(const std::string& name, T* val) {
    auto it = metaInfos.find(name);
    if (it == metaInfos.end()) {
      return paddle::Error("Cannot find meta %s", name.c_str());
    }
    val = any_cast<T>(&it->second);
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
};

}  // namespace meta
}  // namespace topology
}  // namespace paddle
