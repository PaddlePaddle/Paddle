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

#include "Validator.h"
#include "FunctionMeta.h"
namespace paddle {
namespace topology {
namespace meta {

Error AttributeValidator::validate(
    std::unordered_map<std::string, any>* attrs) const {
  for (auto it = attrs->begin(); it != attrs->end(); ++it) {
    auto metaIt = this->metas_.find(it->first);
    if (metaIt == this->metas_.end()) {
      return Error("No such attribute meta %s", it->first.c_str());
    }
    auto err = metaIt->second->check(&it->second, true /*setted*/);
    if (!err.isOK()) {
      return err;
    }
  }

  for (auto metaIt = metas_.begin(); metaIt != metas_.end(); ++metaIt) {
    auto it = attrs->find(metaIt->first);
    if (it != attrs->end()) continue;
    (*attrs)[metaIt->first] = any();
    any* tmp = &(*attrs)[metaIt->first];
    auto err = metaIt->second->check(tmp, false /*setted*/);
    if (!err.isOK()) {
      return err;
    }
  }
  return Error();
}

Error validate(Function& func) {
  auto meta = FunctionMeta::get(func.type);
  if (meta == nullptr) {
    return Error("No such function type %s", func.type.c_str());
  }
  AttributeValidator validator(meta->getAttributes());
  return validator.validate(&func.attributes);
}
}  // namespace meta
}  // namespace topology
}  // namespace paddle
