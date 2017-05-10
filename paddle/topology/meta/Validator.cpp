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
      return Error("Meta Error(%s)", err.msg());
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

Error validateAndInferShape(Function& func, bool validOutput) {
  auto meta = FunctionMeta::get(func.type);
  if (meta == nullptr) {
    return Error("No such function type %s", func.type.c_str());
  }
  auto err = validate(*meta, func.attributes);
  if (!err.isOK()) {
    return Error("AttrErr(%s)", err.msg());
  }
  auto& inMetas = meta->inputs();
  auto& outMetas = meta->outputs();

  if (inMetas.size() != func.inputs.size()) {
    return Error("Input size mismatch");
  }
  if (outMetas.size() != func.outputs.size()) {
    return Error("Output size mismatch");
  }
  for (size_t i = 0; i < inMetas.size(); ++i) {
    err = validate(*inMetas[i], func.inputs[i]->attributes);
    if (!err.isOK()) return Error("Input %d error %s", i, err.msg());
  }
  if (validOutput) {
    for (size_t i = 0; i < outMetas.size(); ++i) {
      err = validate(*outMetas[i], func.outputs[i]->attributes);
      if (!err.isOK()) return err;
    }
  }
  err = meta->getShapeInferer()(func.inputs, func.outputs);
  if (!err.isOK()) return err;
  for (size_t i = 0; i < outMetas.size(); ++i) {
    err = validate(*outMetas[i], func.outputs[i]->attributes);
    if (!err.isOK()) return Error("Output %d error %s", i, err.msg());
  }
  return Error();
}

Error validate(const WithAttributeMeta& meta, Attribute& attr) {
  AttributeValidator validator(meta.getAttributes());
  return validator.validate(&attr);
}

}  // namespace meta
}  // namespace topology
}  // namespace paddle
