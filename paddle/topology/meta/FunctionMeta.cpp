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

namespace paddle {
namespace topology {
namespace meta {
static Map<std::string, FunctionMetaPtr> gFuncMetas;

const Set<int> FunctionMeta::defaultSeqTypes = {SequenceType::NO_SEQUENCE,
                                                SequenceType::SEQUENCE,
                                                SequenceType::NESTED_SEQUENCE};

Error FunctionMeta::registerFuncMeta(
    const std::string &name, std::function<Error(FunctionMetaPtr &)> func) {
  if (gFuncMetas.find(name) != gFuncMetas.end()) {
    return paddle::Error("Function %s has been registered", name.c_str());
  }
  auto metaPtr = std::make_shared<FunctionMeta>(name);
  auto err = func(metaPtr);
  if (err.isOK()) {
    gFuncMetas[name] = metaPtr;
  }
  return err;
}

FunctionMetaPtr FunctionMeta::get(const std::string &name) {
  auto it = gFuncMetas.find(name);
  if (it != gFuncMetas.end()) {
    return it->second;
  } else {
    return nullptr;
  }
}

}  // namespace meta
}  // namespace topology
}  // namespace paddle
