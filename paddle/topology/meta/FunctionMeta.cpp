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
#include "FunctionMeta.h"
#include "../Attribute.h"
namespace paddle {
namespace topology {
namespace meta {
static Map<std::string, FunctionMetaPtr> gFuncMetas;

FunctionMetaPtr FunctionMeta::registerFuncMeta(const std::string &name) {
  CHECK(gFuncMetas.find(name) == gFuncMetas.end()) << "Function " << name
                                                   << " has been registered";
  auto metaPtr = std::make_shared<FunctionMeta>(name);
  gFuncMetas[name] = metaPtr;
  return metaPtr;
}

FunctionMetaPtr FunctionMeta::get(const std::string &name) {
  auto it = gFuncMetas.find(name);
  if (it != gFuncMetas.end()) {
    return it->second;
  } else {
    return nullptr;
  }
}

Error FunctionMeta::parseAttribute(const AttributeMap &attrs,
                                   Attribute *out) const {
  auto parserFunction = metaAttributes_.template get<
      std::function<Error(const topology::AttributeMap &, Attribute *)>>(
      "attribute_parser");
  return parserFunction(attrs, out);
}

}  // namespace meta
}  // namespace topology
}  // namespace paddle
