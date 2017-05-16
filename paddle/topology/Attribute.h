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
#include <memory>
#include <string>
#include <unordered_map>
#include "AttributeMap.h"
#include "details/AttributeParser.h"
#include "meta/AttributeMeta.h"
#include "meta/FunctionMeta.h"

namespace paddle {
namespace topology {
class Attribute {
public:
  bool useGPU;

  virtual ~Attribute();

protected:
  template <typename T, typename SubClass>
  static meta::Constraints<T>& regAttr(T SubClass::*memPtr,
                                       const std::string& name,
                                       const std::string& description) {
    details::gCurParser->append(name, memPtr);
    return details::gCurFuncMeta->addAttribute<T>(name, description);
  }

  static void parentRegAttrs() {
    regAttr(&Attribute::useGPU, "useGPU", "Use GPU or not").mustSet();
  }
};

#define REGISTER_FUNC_ATTRIBUTE()                                      \
  static void registerFunctionAttribute(                               \
      const paddle::topology::meta::FunctionMetaPtr metaPtr) {         \
    paddle::topology::details::FunctionMetaScope scope(metaPtr.get()); \
    parentRegAttrs();                                                  \
    registerFunctionAttribute__impl__();                               \
  }                                                                    \
  static void registerFunctionAttribute__impl__()

}  // namespace topology

}  // namespace paddle
